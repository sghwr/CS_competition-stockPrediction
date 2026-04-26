import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from market_prior import AdaptiveWeightNet

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CrossStockAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, stock_features):
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        output = self.norm(stock_features + self.dropout(attended))
        return output

class FeatureAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attention_weights = self.attention(x)
        attended = torch.sum(x * attention_weights, dim=1)
        return self.dropout(attended)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks
        
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        
        self.cross_stock_attention = CrossStockAttention(config['d_model'], config['nhead'], config['dropout'])
        
        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )
        
        self._init_weights()
        
        self.adaptive_net = AdaptiveWeightNet(input_dim=5, hidden_dim=8)
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, src, prior_bias=None, market_state=None):
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        
        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)
        
        temporal_features = self.temporal_encoder(src_proj)
        
        aggregated_features = self.feature_attention(temporal_features)
        
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)
        
        interactive_features = self.cross_stock_attention(stock_features)
        
        interactive_features = interactive_features.view(batch_size * num_stocks, -1)
        
        ranking_features = self.ranking_layers(interactive_features)
        
        scores = self.score_head(ranking_features)
        
        output = scores.view(batch_size, num_stocks)
        
        if prior_bias is not None and market_state is not None:
            state_onehot = market_state[:, :4]
            vol_scalar = market_state[:, 4]
            w = self.adaptive_net(state_onehot, vol_scalar)
            output = output + w.unsqueeze(-1) * prior_bias
        
        return output