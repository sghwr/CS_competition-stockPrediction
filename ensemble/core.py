import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import multiprocessing as mp

_PROJECT_ROOT = Path(__file__).parent.parent
_SRC = str(_PROJECT_ROOT / 'code' / 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import config as transformer_config
from model import StockTransformer
from utils import engineer_features_158plus39

from .config import ENSEMBLE_CONFIG, FEATURE_COLUMNS


class Ensemble:
    def __init__(self, cfg=None):
        self.cfg = cfg or ENSEMBLE_CONFIG
        self.device = self._resolve_device()
        self.transformer = self._load_transformer()
        self.lgbm = self._load_lgbm()

    @staticmethod
    def _resolve_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _load_transformer(self):
        model = StockTransformer(
            input_dim=len(FEATURE_COLUMNS),
            config=transformer_config,
            num_stocks=1,
        )
        state = torch.load(self.cfg['transformer_model'], map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _load_lgbm(self):
        return joblib.load(self.cfg['lgbm_model'])

    # ── 公开接口 ──────────────────────────────────

    def feature_engineer(self, raw_df):
        groups = [g for _, g in raw_df.groupby('股票代码', sort=False)]
        nproc = min(8, mp.cpu_count())
        with mp.Pool(processes=nproc) as pool:
            from tqdm import tqdm
            processed = list(tqdm(pool.imap_unordered(engineer_features_158plus39, groups), total=len(groups), desc='ensemble 特征工程'))
        processed = pd.concat(processed).reset_index(drop=True)
        processed['日期'] = pd.to_datetime(processed['日期'])
        return processed

    def load_scaler(self):
        return joblib.load(self.cfg['transformer_scaler'])

    def predict_date(self, processed, features, date, scaler=None):
        date = pd.to_datetime(date) if not isinstance(date, pd.Timestamp) else date
        date_data = processed[processed['日期'] == date]
        if len(date_data) == 0:
            return None

        stock_ids = date_data['股票代码'].values.tolist()

        tf_scores = self._tf_predict(processed, features, stock_ids, date, scaler)
        lgbm_scores = self._lgbm_predict(date_data, features)

        common = sorted(set(tf_scores.keys()) & set(lgbm_scores.keys()))
        if len(common) == 0:
            return None

        tf_arr = np.array([tf_scores[s] for s in common], dtype=np.float64)
        lgbm_arr = np.array([lgbm_scores[s] for s in common], dtype=np.float64)

        tf_norm = (tf_arr - tf_arr.mean()) / (tf_arr.std() + 1e-8)
        lgbm_norm = (lgbm_arr - lgbm_arr.mean()) / (lgbm_arr.std() + 1e-8)

        w_tf = self.cfg['weight_transformer']
        w_lgbm = self.cfg['weight_lgbm']
        ensemble = w_tf * tf_norm + w_lgbm * lgbm_norm

        return {
            'scores': ensemble,
            'stock_ids': common,
            'transformer_scores': tf_arr,
            'lgbm_scores': lgbm_arr,
        }

    # ── 内部预测方法 ──────────────────────────────

    def _tf_predict(self, processed, features, stock_ids, date, scaler=None):
        seq_len = self.cfg['sequence_length']
        sequences, valid_ids = [], []
        for sid in stock_ids:
            hist = processed[(processed['股票代码'] == sid) & (processed['日期'] <= date)]
            hist = hist.sort_values('日期').tail(seq_len)
            if len(hist) < seq_len:
                continue
            row = hist[features].values.astype(np.float32)
            sequences.append(row)
            valid_ids.append(sid)
        if len(sequences) == 0:
            return {}

        x = np.stack(sequences, axis=0)
        if scaler is not None:
            orig_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
            x_flat = pd.DataFrame(x_flat, columns=features)
            x_flat = scaler.transform(x_flat)
            if hasattr(x_flat, 'values'):
                x = x_flat.values.reshape(orig_shape)
            else:
                x = x_flat.reshape(orig_shape)

        if self.cfg.get('use_mdrp', False):
            scores = self._tf_predict_with_mdrp(x, valid_ids, date)
        else:
            with torch.no_grad():
                t = torch.from_numpy(x).unsqueeze(0).to(self.device)
                out = self.transformer(t).squeeze(0).cpu().numpy()
            scores = out

        return dict(zip(valid_ids, scores))

    def _tf_predict_with_mdrp(self, x_np, stock_ids, date):
        from market_prior import IndustryPriorComputer, MarketStateExtractor

        prior_computer = IndustryPriorComputer(
            industry_csv=self.cfg['industry_csv'],
            stock_data_csv=self.cfg['stock_data_csv'],
            lookback=transformer_config.get('mdrp_lookback', 5),
        )
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
        prior = prior_computer.get_prior_returns(date_str, stock_ids)
        prior_bias = torch.from_numpy(prior).float().unsqueeze(0).to(self.device)

        state_extractor = MarketStateExtractor(index_csv=self.cfg['index_csv'])
        onehot = state_extractor.get_state_onehot(date_str)
        vol = state_extractor.get_volatility(date_str)
        market_state = torch.from_numpy(
            np.concatenate([onehot, [vol]]).astype(np.float32)
        ).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            t = torch.from_numpy(x_np).unsqueeze(0).to(self.device)
            out = self.transformer(t, prior_bias=prior_bias, market_state=market_state)
            out = out.squeeze(0).cpu().numpy()
        return out

    def _lgbm_predict(self, date_data, features):
        X = date_data[features].values.astype(np.float32)
        scores = self.lgbm.predict(X)
        return dict(zip(date_data['股票代码'].values, scores))

    @staticmethod
    def _normalize(scores):
        return (scores - scores.mean()) / (scores.std() + 1e-8)
