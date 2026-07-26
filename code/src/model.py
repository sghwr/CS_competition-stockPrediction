"""Models: MacroIndustryAlphaBetaModel (industry-level) + LinearRegressionModel (stock-level).

  - MacroIndustryAlphaBetaModel: HS300 + 11 industries x 60d x 3 features
    → 输出 industry_alpha (B, 11) + industry_beta (B, 11)
  - LinearRegressionModel: 10 dim reversal features (无 RSI, 跟 Tree 不重复)

设计原则:
  - regime 用确定性规则 (regime_classifier), 不训
  - log_volume, RSI 等跟 Tree/Linear 重复的 features 删掉
  - Macro 只输出 industry alpha + beta, 跟 Tree 职责清晰分离
"""
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """正弦位置编码 (Transformer 共享)."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
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


class MacroIndustryAlphaBetaModel(nn.Module):
    """MacroIndustryAlphaBetaModel — 行业层面预测 (只训 alpha + beta).

    输入: (B, 12, 60, 3)
      - channels: 12 = [HS300] + 11 industries
      - features (3, open-based, dimensionless):
        0. log_return_1d   log(open[t]/open[t-1])
        1. vol_20d         基于 1d log return 的 std
        2. open_mom_5d     log(open[t]/open[t-5])

    输出:
      industry_alpha:  (B, 11)  per-industry 5d excess return (rotation)
      industry_beta:   (B, 11)  per-industry rolling 60d corr vs HS300 (tanh [-1, 1])

    Loss: 0.4 * Huber(alpha) + 0.4 * Huber(beta) + 0.2 * pairwise_rank(alpha)
    """
    def __init__(self, n_channels=12, n_features=3, seq_len=60,
                 d_model=128, nhead=4, num_layers=3,
                 n_industries=11, dropout=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.n_industries = n_industries
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_features = n_features

        self.channel_embed = nn.Embedding(n_channels, d_model)
        self.industry_embed = nn.Embedding(n_industries, d_model)

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, max_len=n_channels * seq_len + 100,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=512, dropout=dropout,
            norm_first=True, activation='gelu',
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.industry_cross_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True,
        )
        self.industry_norm = nn.LayerNorm(d_model)

        # Output heads: industry alpha + beta
        self.industry_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        """x: (B, 12, 60, 3)

        Returns:
          industry_alpha: (B, 11) per-industry 5d excess return
          industry_beta:  (B, 11) per-industry rolling 60d corr vs HS300 (tanh [-1, 1])
        """
        B, C, T, F_ = x.shape
        x = self.input_proj(x)
        channel_ids = torch.arange(C, device=x.device)
        ch_emb = self.channel_embed(channel_ids)
        x = x + ch_emb.view(1, C, 1, self.d_model)
        x = x.view(B, C * T, self.d_model)
        x = self.pos_encoder(x)
        z = self.temporal_encoder(x)
        z = z.view(B, C, T, self.d_model)

        z_last = z[:, :, -1, :]
        ind_z_last = z_last[:, 1:, :]

        ind_ids = torch.arange(self.n_industries, device=x.device)
        ind_emb = self.industry_embed(ind_ids).unsqueeze(0)
        ind_z_in = ind_z_last + ind_emb
        ind_z_out, _ = self.industry_cross_attn(ind_z_in, ind_z_in, ind_z_in)
        ind_z = self.industry_norm(ind_z_last + ind_z_out)

        ind_z_full = z[:, 1:, :, :]
        ind_short = ind_z_full[:, :, -5:, :].mean(dim=2)
        ind_long = ind_z_full.mean(dim=2)
        ind_mom = ind_short - ind_long
        ind_multi = torch.cat([ind_short, ind_long, ind_mom], dim=-1)

        ind_alpha = self.industry_head(ind_multi).squeeze(-1)
        ind_beta = torch.tanh(self.beta_head(ind_multi).squeeze(-1))  # [-1, 1]

        return ind_alpha, ind_beta


# 保留旧类名作为 alias, 避免破坏其他文件导入
MacroTransformerV4 = MacroIndustryAlphaBetaModel


class LinearRegressionModel(nn.Module):
    """简单线性回归做个股预测.

    score = clip(features @ W + b, -0.5, 0.5)
    训练: MSE loss + L2 regularization (weight_decay).
    加 tanh 限制 output 范围, 防止 val/test 数值发散.
    """
    def __init__(self, n_features, output_clip=0.3):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)
        self.output_clip = output_clip

    def forward(self, x):
        """x: (B, n_features) or (N, n_features) -> (B,) or (N,)"""
        out = self.linear(x).squeeze(-1)
        if self.output_clip is not None:
            out = out.clamp(-self.output_clip, self.output_clip)
        return out


def _rsi_wilder(close, period=14):
    """Wilder's RSI (14-day default). Returns 0-100, 50 = neutral.

    不再用于 build_macro_features (跟 Tree/Linear 重复), 保留供其他用途.
    """
    import pandas as pd
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0).values


def build_macro_features(df, industry_map, index_csv, seq_len=60):
    """3 维特征 MacroIndustryAlphaBetaModel 输入 (log_return, open-based).

    Features per channel (3, open-based, dimensionless):
      0. log_return_1d    log(open[t]/open[t-1]), 跨行业可比
      1. vol_20d          基于 1d log return 的 std
      2. open_mom_5d      log(open[t]/open[t-5]), 5d 动量
    """
    import pandas as pd

    # 1. HS300 data
    idx = pd.read_csv(index_csv, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    idx['log_ret_1d'] = np.log(idx['open'] / idx['open'].shift(1)).fillna(0).clip(-0.1, 0.1)
    idx['vol_20d'] = idx['log_ret_1d'].rolling(20).std().fillna(0.01)
    idx['open_mom_5d'] = np.log(idx['open'] / idx['open'].shift(5)).fillna(0).clip(-0.2, 0.2)
    hs300_by_date = {}
    for d, lr, vol, mom in zip(
        idx['date'], idx['log_ret_1d'], idx['vol_20d'], idx['open_mom_5d'],
    ):
        hs300_by_date[d.date()] = (float(lr), float(vol), float(mom))

    # 2. Industry mapping
    industries = sorted(set(industry_map.values()))
    n_ind = len(industries)
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}

    # 3. Per-day industry (log-return based)
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df['industry_idx'] = df['股票代码'].map(
        lambda c: industry_to_idx.get(industry_map.get(int(c), ''), 0)
    )
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    df['log_ret_1d'] = df.groupby('股票代码').apply(
        lambda g: np.log(g['开盘'] / g['开盘'].shift(1)).fillna(0).clip(-0.1, 0.1)
    ).reset_index(level=0, drop=True)
    df['vol_20d'] = df.groupby('股票代码')['log_ret_1d'].transform(
        lambda s: s.rolling(20).std()
    ).fillna(0.02)
    df['open_mom_5d'] = df.groupby('股票代码').apply(
        lambda g: np.log(g['开盘'] / g['开盘'].shift(5)).fillna(0).clip(-0.2, 0.2)
    ).reset_index(level=0, drop=True)

    ind_daily = df.groupby(['日期', 'industry_idx']).agg({
        'log_ret_1d': 'mean',
        'vol_20d': 'mean',
        'open_mom_5d': 'mean',
    }).reset_index()
    ind_daily = ind_daily.sort_values(['industry_idx', '日期']).reset_index(drop=True)

    # 4. 排序所有日期
    all_dates = sorted(df['日期'].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # 5. HS300 序列 (3 dim)
    hs300_seq = np.zeros((len(all_dates), 3), dtype=np.float32)
    for d in all_dates:
        d_py = d.date() if hasattr(d, 'date') else d
        v = hs300_by_date.get(d_py, (0, 0.01, 0))
        hs300_seq[date_to_idx[d]] = v

    # 6. 11 industries 序列 (向量化: pivot table, 3 features)
    ind_seq = np.zeros((len(all_dates), n_ind, 3), dtype=np.float32)
    ind_pivot = ind_daily.pivot_table(
        index='日期', columns='industry_idx',
        values=['log_ret_1d', 'vol_20d', 'open_mom_5d'],
        aggfunc='first',
    )
    for feat_idx, feat_name in enumerate(['log_ret_1d', 'vol_20d', 'open_mom_5d']):
        if (feat_name,) in ind_pivot.columns:
            sub = ind_pivot[feat_name]
            for ii in range(n_ind):
                if ii in sub.columns:
                    vals = sub[ii].reindex(all_dates).values
                    di_mask = ~pd.isna(vals)
                    ind_seq[di_mask, ii, feat_idx] = vals[di_mask]

    # 7. 拼接
    raw_seq = np.concatenate([hs300_seq[:, None, :], ind_seq], axis=1)
    T = len(all_dates)

    # 8. 滑动窗口
    if T < seq_len:
        raise ValueError(f"need T >= seq_len, got T={T}, seq_len={seq_len}")
    n_f = raw_seq.shape[-1]
    macro_X = np.zeros((T - seq_len, 12, seq_len, n_f), dtype=np.float32)
    for t in range(seq_len, T):
        window = raw_seq[t - seq_len:t].transpose(1, 0, 2)
        macro_X[t - seq_len] = window

    # 9. 输出日期
    macro_dates = [str(all_dates[t].date()) if hasattr(all_dates[t], 'date') else str(all_dates[t])
                   for t in range(seq_len, T)]

    return macro_X, macro_dates, industries
