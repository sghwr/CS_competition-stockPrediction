"""
集成模型类
实现加权平均分数集成
"""
import numpy as np
import torch
import joblib
import json
from pathlib import Path
import sys

# 添加必要的路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'LGBM'))

from .config import ENSEMBLE_CONFIG, FEATURE_COLUMNS_MAP


class EnsembleModel:
    """LGBM和Transformer的加权平均集成模型"""
    
    def __init__(self, transformer_model, lgbm_model, weight=0.5, config=None):
        """
        初始化集成模型
        
        Args:
            transformer_model: Transformer模型
            lgbm_model: LGBM模型
            weight: Transformer权重（LGBM权重为1-weight）
            config: 配置字典
        """
        self.transformer = transformer_model
        self.lgbm = lgbm_model
        self.weight = weight  # transformer权重
        self.config = config or ENSEMBLE_CONFIG
        
    @classmethod
    def from_config(cls, config=None):
        """从配置加载模型"""
        config = config or ENSEMBLE_CONFIG
        
        # 加载Transformer模型
        transformer_model = cls._load_transformer_model(config)
        
        # 加载LGBM模型
        lgbm_model = cls._load_lgbm_model(config)
        
        # 加载集成权重（如果有保存的最佳权重）
        weight = config.get('default_weight', 0.5)
        
        # 尝试加载最佳权重
        model_dir = Path(config['model_dir'])
        best_config_path = model_dir / 'best_ensemble_config.json'
        if best_config_path.exists():
            try:
                with open(best_config_path, 'r') as f:
                    best_config = json.load(f)
                weight = best_config.get('best_weight', weight)
                print(f"加载最佳权重: {weight:.3f}")
            except Exception as e:
                print(f"加载最佳权重失败: {e}, 使用默认权重: {weight}")
        
        return cls(transformer_model, lgbm_model, weight, config)
    
    @staticmethod
    def _load_transformer_model(config):
        """加载Transformer模型"""
        import torch
        from code.src.model import StockTransformer
        from code.src.config import config as transformer_config
        
        model_path = Path(config['model_paths']['transformer']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Transformer模型文件不存在: {model_path}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 获取模型参数
        feature_num = config['feature_num']
        feature_columns = FEATURE_COLUMNS_MAP[feature_num]
        input_dim = len(feature_columns) - 1  # 减去instrument列
        
        # 估计股票数量（可以从数据中获取）
        num_stocks = 300  # 默认值，实际可以从数据中获取
        
        # 创建模型
        model = StockTransformer(
            input_dim=input_dim,
            config=transformer_config,
            num_stocks=num_stocks
        )
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print(f"Transformer模型已加载: {model_path}")
        return model
    
    @staticmethod
    def _load_lgbm_model(config):
        """加载LGBM模型"""
        model_path = Path(config['model_paths']['lgbm']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"LGBM模型文件不存在: {model_path}")
        
        model = joblib.load(model_path)
        print(f"LGBM模型已加载: {model_path}")
        return model
    
    def predict(self, transformer_data, lgbm_data):
        """
        集成预测
        
        Args:
            transformer_data: Transformer输入数据
            lgbm_data: LGBM输入数据
            
        Returns:
            dict: 包含集成分数、原始分数和股票ID的字典
        """
        # 获取Transformer预测
        transformer_scores = self._predict_transformer(transformer_data)
        
        # 获取LGBM预测
        lgbm_scores = self._predict_lgbm(lgbm_data)
        
        # 对齐股票ID
        aligned_scores = self._align_scores(
            transformer_scores, lgbm_scores,
            transformer_data['stock_ids'], lgbm_data['stock_ids']
        )
        
        # 标准化分数
        norm_transformer = self._normalize_scores(aligned_scores['transformer'])
        norm_lgbm = self._normalize_scores(aligned_scores['lgbm'])
        
        # 加权平均
        ensemble_scores = (
            self.weight * norm_transformer + 
            (1 - self.weight) * norm_lgbm
        )
        
        return {
            'ensemble': ensemble_scores,
            'transformer': aligned_scores['transformer'],
            'lgbm': aligned_scores['lgbm'],
            'stock_ids': aligned_scores['stock_ids'],
            'transformer_raw': transformer_scores,
            'lgbm_raw': lgbm_scores
        }
    
    def _predict_transformer(self, data):
        """Transformer预测"""
        with torch.no_grad():
            device = next(self.transformer.parameters()).device
            x = data['sequences'].to(device)
            scores = self.transformer(x).squeeze(0).cpu().numpy()
        
        return scores
    
    def _predict_lgbm(self, data):
        """LGBM预测"""
        scores = self.lgbm.predict(data['X'])
        return scores
    
    def _align_scores(self, transformer_scores, lgbm_scores, 
                     transformer_stocks, lgbm_stocks):
        """对齐两个模型的股票分数"""
        # 找出共同股票
        common_stocks = set(transformer_stocks) & set(lgbm_stocks)
        common_stocks = sorted(common_stocks)
        
        if len(common_stocks) == 0:
            raise ValueError("两个模型没有共同的股票，无法集成")
        
        print(f"找到 {len(common_stocks)} 只共同股票进行集成")
        
        # 创建映射
        transformer_dict = dict(zip(transformer_stocks, transformer_scores))
        lgbm_dict = dict(zip(lgbm_stocks, lgbm_scores))
        
        # 提取共同股票的分数
        aligned_transformer = np.array([transformer_dict[s] for s in common_stocks])
        aligned_lgbm = np.array([lgbm_dict[s] for s in common_stocks])
        
        return {
            'transformer': aligned_transformer,
            'lgbm': aligned_lgbm,
            'stock_ids': common_stocks
        }
    
    def _normalize_scores(self, scores):
        """z-score标准化分数"""
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-8:
            return scores - mean  # 如果标准差太小，只去中心化
        return (scores - mean) / std
    
    def set_weight(self, weight):
        """设置集成权重"""
        self.weight = max(0.0, min(1.0, weight))
        print(f"设置集成权重: Transformer={self.weight:.3f}, LGBM={1-self.weight:.3f}")
    
    def save(self, path):
        """保存集成模型配置"""
        import json
        config = {
            'weight': self.weight,
            'transformer_path': str(self.config['model_paths']['transformer']['model_path']),
            'lgbm_path': str(self.config['model_paths']['lgbm']['model_path']),
            'feature_num': self.config['feature_num'],
            'sequence_length': self.config['sequence_length']
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"集成模型配置已保存: {path}")
    
    @classmethod
    def load(cls, path, config=None):
        """从保存的配置加载集成模型"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"集成模型配置文件不存在: {path}")
        
        with open(path, 'r') as f:
            saved_config = json.load(f)
        
        # 创建配置
        load_config = config or ENSEMBLE_CONFIG.copy()
        load_config['model_paths']['transformer']['model_path'] = Path(saved_config['transformer_path'])
        load_config['model_paths']['lgbm']['model_path'] = Path(saved_config['lgbm_path'])
        
        # 加载模型
        model = cls.from_config(load_config)
        model.set_weight(saved_config['weight'])
        
        return model
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'transformer_weight': self.weight,
            'lgbm_weight': 1 - self.weight,
            'transformer_device': str(next(self.transformer.parameters()).device),
            'lgbm_type': type(self.lgbm).__name__,
            'feature_num': self.config['feature_num'],
            'sequence_length': self.config['sequence_length']
        }