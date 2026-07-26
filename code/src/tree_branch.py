"""LightGBM Tree 分支 (Phase B)
- 187 dim 因子特征 (97 Tier1 + 89 Tier2 + 1 industry_id)
- LGBMRanker (lambdarank) with group=date
- 5d forward return as target
- 3 regime-specific models (bull/sideways/bear)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import lightgbm as lgb
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from utils import (
     engineer_features_158plus39,
     enrich_window_factors,
     TIER1_LAST_ONLY,
     TIER2_ENRICHED,
     TIER3_INDUSTRY,
 )
from industry_mapping import load_industry_map
from paths import INTEGRATED_DIR, MODEL_INTEGRATED_DIR, TRAIN_CSV


def add_industry_features(df, industry_map):
    """为 df 添加 10 维行业特征 (向量化 groupby).

    输入 df 必须有 '日期', '股票代码' 列和下列基础列:
      ROC5, ROC10, ROC20, STD20, BETA5, volume_ratio

    添加列 (TIER3_INDUSTRY):
      ind_rel_ROC5, ind_rel_ROC10, ind_rel_ROC20,
      ind_rel_STD20, ind_rel_BETA5, ind_rel_volume,
      ind_avg_ROC5, ind_5d_rank, ind_mom_persist, industry_id
    """
    df = df.copy()
    # 1. 行业 id (categorical)
    industries = sorted(set(industry_map.values()))
    industry_to_idx = {ind: i for i, ind in enumerate(industries)}
    df['industry_id'] = df['股票代码'].map(
        lambda c: industry_to_idx.get(industry_map.get(int(c), ''), 0)
    ).astype('int32')

    # 2. industry-relative 特征 (6): 个股 - 行业均值
    rel_specs = [
        ('ROC5',         'ind_rel_ROC5'),
        ('ROC10',        'ind_rel_ROC10'),
        ('ROC20',        'ind_rel_ROC20'),
        ('STD20',        'ind_rel_STD20'),
        ('BETA5',        'ind_rel_BETA5'),
        ('volume_ratio', 'ind_rel_volume'),
    ]
    for base_col, new_col in rel_specs:
        if base_col in df.columns:
            ind_mean = df.groupby(['日期', 'industry_id'])[base_col].transform('mean')
            df[new_col] = df[base_col] - ind_mean
        else:
            df[new_col] = 0.0

    # 3. industry_avg_ROC5: 行业 5d 平均
    if 'ROC5' in df.columns:
        df['ind_avg_ROC5'] = df.groupby(['日期', 'industry_id'])['ROC5'].transform('mean')
    else:
        df['ind_avg_ROC5'] = 0.0

    # 4. ind_5d_rank: 行业 5d 收益在 11 行业中的 rank (per day)
    #    先 groupby 到 (date, industry) 一行, 然后按日期 rank, 再 merge 回
    ind_5d = df.groupby(['日期', 'industry_id'])['ROC5'].mean().reset_index(name='_ind_5d_mean')
    ind_5d['ind_5d_rank'] = ind_5d.groupby('日期')['_ind_5d_mean'].rank(pct=True) - 0.5
    df = df.merge(ind_5d[['日期', 'industry_id', 'ind_5d_rank']], on=['日期', 'industry_id'], how='left')

    # 5. ind_mom_persist: 行业 5d 与 20d 收益符号一致性
    if 'ROC20' in df.columns:
        ind_20d = df.groupby(['日期', 'industry_id'])['ROC20'].mean().reset_index(name='_ind_20d_mean')
        ind_5d_20d = ind_5d.merge(ind_20d, on=['日期', 'industry_id'], how='left')
        ind_5d_20d['ind_mom_persist'] = np.sign(ind_5d_20d['_ind_5d_mean']) * np.sign(ind_5d_20d['_ind_20d_mean'])
        df = df.merge(ind_5d_20d[['日期', 'industry_id', 'ind_mom_persist']],
                      on=['日期', 'industry_id'], how='left')
    else:
        df['ind_mom_persist'] = 0.0

    # 填 NaN (无行业映射的股票)
    for c in TIER3_INDUSTRY:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df


def split_train_val(df, val_start=None, val_end=None, test_start=None, test_end=None, seq_len=60):
    """按 config.SPLITS 切分: train + val (9月, 早停+stack) + test (3月, strict OOS).

    Args:
        val_start: val 开始 (默认 config 'val_start')
        val_end:   val 结束 (默认 config 'val_end')
        test_start: test 开始 (默认 config 'test_start')
        test_end:   test 结束 (默认 config 'test_end')
    """
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)
    if val_start is None:
        val_start = config['val_start']
    if val_end is None:
        val_end = config['val_end']
    if test_start is None:
        test_start = config['test_start']
    if test_end is None:
        test_end = config['test_end']
    val_start_dt = pd.Timestamp(val_start)
    val_end_dt = pd.Timestamp(val_end)
    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end)
    val_context_start = val_start_dt - pd.tseries.offsets.BDay(seq_len - 1)
    # train: 全部 < val_start (不含 val)
    train_df = df[df['日期'] < val_start_dt].copy()
    # val + test context: 用于 early stop 和 OOF 评估
    val_df = df[(df['日期'] >= val_context_start) & (df['日期'] <= test_end_dt + pd.tseries.offsets.BDay(5))].copy()
    print(f"[Split] train: {train_df['日期'].min().date()} -> {train_df['日期'].max().date()}", flush=True)
    print(f"[Split] val: {val_start_dt.date()} -> {val_end_dt.date()}", flush=True)
    print(f"[Split] test: {test_start_dt.date()} -> {test_end_dt.date()}", flush=True)
    print(f"[Split] val+test context (含 60d): {val_df['日期'].min().date()} -> {val_df['日期'].max().date()}", flush=True)
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')
    return train_df, val_df, val_start_dt


def build_per_stock_windows(per_stock_df, seq_len=60, future_gap=5):
    """对单只股票生成 (end_date, stock_code, features_dict, label) 列表.
    end_date T: 窗口为 [T-59, T], 标签 = (open[T+5] - open[T+1]) / open[T+1]
    """
    out = []
    n = len(per_stock_df)
    if n < seq_len + future_gap:
        return out

    dates = per_stock_df['日期'].astype(str).values
    opens = per_stock_df['开盘'].astype(float).values
    stock_codes = per_stock_df['股票代码'].values

    cols_to_use = TIER1_LAST_ONLY + TIER2_ENRICHED + TIER3_INDUSTRY  # 187 dim (97+89+1)
    cols_present = [c for c in cols_to_use if c in per_stock_df.columns]

    arr = per_stock_df[cols_present].values.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 向量化: 一次算所有 valid labels
    valid_starts = np.arange(seq_len, n - future_gap + 1)
    open_t1 = opens[valid_starts]
    open_t5 = opens[valid_starts + future_gap - 1]
    valid_mask = open_t1 > 1e-4
    valid_starts = valid_starts[valid_mask]
    open_t1 = open_t1[valid_mask]
    open_t5 = open_t5[valid_mask]
    labels = np.clip((open_t5 - open_t1) / (open_t1 + 1e-12), -0.5, 0.5)

    for k, i in enumerate(valid_starts):
        end_idx = i - 1  # T
        window = arr[i - seq_len:i]  # (60, 186)
        feats = enrich_window_factors_np(window, cols_present, mode='tiered')
        out.append((dates[end_idx], stock_codes[end_idx], feats, float(labels[k])))
    return out


def enrich_window_factors_np(window_arr, cols, mode='tiered'):
    """对 numpy 数组形式的 60d 窗口做 enrich (向量化: 只取末值, 无 slope).
    LightGBM 设计: 不与 Transformer (7 原始量价) 和 MD-SRP (3 动量 rank) 重复,

    也不需要时序斜率信息, 只用 186 维因子末值 (97 Tier1 + 89 Tier2)
    + 10 维行业 (Tier3) + 3 维行业 beta (Tier3β).

    Args:
        window_arr: (60, K) numpy array
        cols: 对应的列名
    Returns:
        dict {feature_name: value}
    """
    out = {}
    t1 = [c for c in TIER1_LAST_ONLY if c in cols]
    t2 = [c for c in TIER2_ENRICHED if c in cols]
    t3 = [c for c in TIER3_INDUSTRY if c in cols]
    t3b = []  # 删 Tier3β (跟 MacroTransformer 重复)
    t1_idx = [cols.index(c) for c in t1]
    t2_idx = [cols.index(c) for c in t2]
    t3_idx = [cols.index(c) for c in t3]
    t3b_idx = [cols.index(c) for c in t3b]

    # Tier1: 末值 (向量化)
    if t1_idx:
        last_vals = window_arr[-1, t1_idx]
        for c, v in zip(t1, last_vals):
            out[c] = float(v)

    # Tier2: 末值 (无 slope_20)
    if t2_idx:
        t2_last = window_arr[-1, t2_idx]
        for c, v in zip(t2, t2_last):
            out[c] = float(v)

    # Tier3: 行业特征 (末值, 已是 day-level 算好的)
    if t3_idx:
        t3_last = window_arr[-1, t3_idx]
        for c, v in zip(t3, t3_last):
            out[c] = float(v)

    # Tier3β: 行业 beta/alpha/resid_std (末值)
    if t3b_idx:
        t3b_last = window_arr[-1, t3b_idx]
        for c, v in zip(t3b, t3b_last):
            out[c] = float(v)

    return out


def process_one_stock(sid_group):
    sid, group = sid_group
    group = group.sort_values('日期').reset_index(drop=True)
    return build_per_stock_windows(group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default=str(MODEL_INTEGRATED_DIR / 'tree'),
                        help='模型权重目录 (model_{r}.txt)')
    parser.add_argument('--output_dir', type=str,
                        default=str(INTEGRATED_DIR / 'tree'),
                        help='推理产物目录 (oof_scores, val_meta, scaler 等)')
    parser.add_argument('--val_meta_path', type=str, default=None,
                        help='val_meta.json 路径 (默认: 自动从 sibling transformer/val_meta.json 查找)')
    parser.add_argument('--smoke', action='store_true', help='冒烟测试: 仅用 30 只股票 + 1/10 数据')
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'=' * 80}\n>>> [LightGBM] starting\n", flush=True)
    print(f"[Config] model_dir={args.model_dir}", flush=True)
    print(f"[Config] output_dir={args.output_dir}, smoke={args.smoke}, "
          f"val={config['val_start']}~{config['val_end']} (from config.SPLITS)", flush=True)

    # 数据
    df = pd.read_csv(TRAIN_CSV)
    print(f"[Data] {len(df):,} rows, {df['股票代码'].nunique()} stocks")

    if args.smoke:
        # 冒烟: 仅用 30 只股票 + 最后 500 个交易日
        sample_stocks = sorted(df['股票代码'].unique())[:30]
        df = df[df['股票代码'].isin(sample_stocks)].copy()
        all_dates = sorted(df['日期'].unique())
        df = df[df['日期'].isin(all_dates[-500:])].copy()
        print(f"[Smoke] {len(df):,} rows, {df['股票代码'].nunique()} stocks, {df['日期'].nunique()} dates")

    train_df_raw, val_df_raw, val_start = split_train_val(
        df, val_start=config['val_start'], val_end=config['val_end'],
        test_start=config['test_start'], test_end=config['test_end'],
    )
    all_stocks = sorted(df['股票代码'].unique())
    stockid2idx = {s: i for i, s in enumerate(all_stocks)}
    idx2stock = {i: s for s, i in stockid2idx.items()}

    # val_meta 与 Transformer 对齐
    val_meta_paths = []
    if args.val_meta_path:
        val_meta_paths.append(args.val_meta_path)
    # MacroTransformer 写 val_meta.json
    sibling_macro = os.path.join(os.path.dirname(args.output_dir), 'macro_transformer', 'val_meta.json')
    val_meta_paths.append(sibling_macro)
    val_meta_paths.append(str(INTEGRATED_DIR / 'macro_transformer' / 'val_meta.json'))
    # 兼容旧路径
    val_meta_paths.append(str(INTEGRATED_DIR / 'transformer' / 'val_meta.json'))
    val_meta = None
    for p in val_meta_paths:
        if os.path.exists(p):
            with open(p) as f:
                val_meta = json.load(f)
            break
    if val_meta is None:
        raise FileNotFoundError("val_meta.json not found, run Phase A (MacroTransformer) first")
    val_target_dates = set(val_meta['dates'])
    print(f"[ValMeta] {len(val_meta['dates'])} val dates")

    # 特征工程 (全量) - 用 ThreadPoolExecutor 避免 mp.Pool 的 spawn+IPC 内存膨胀
    print("\n[Engineer] Running engineer_features_158plus39 on full data...", flush=True)
    t0 = time.time()
    all_groups = [g for _, g in df.groupby('股票代码', sort=False)]
    n_thread = min(8, (os.cpu_count() or 1) * 2)
    processed_list = [None] * len(all_groups)
    with ThreadPoolExecutor(max_workers=n_thread) as ex:
        futures = {ex.submit(engineer_features_158plus39, g): i for i, g in enumerate(all_groups)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="engineer", mininterval=2.0):
            processed_list[futures[fut]] = fut.result()
    processed = pd.concat(processed_list).reset_index(drop=True)
    print(f"[Engineer] done in {time.time()-t0:.1f}s, {len(processed):,} rows, {len(processed.columns)} cols", flush=True)

    # 清理 inf/nan
    cols_to_clean = TIER1_LAST_ONLY + TIER2_ENRICHED
    cols_present = [c for c in cols_to_clean if c in processed.columns]
    for c in cols_present:
        processed[c] = processed[c].replace([np.inf, -np.inf], 0).fillna(0)
    print(f"[Engineer] cleaned {len(cols_present)} feature cols", flush=True)

    # Tree 只用 industry_id (1 dim categorical), 删 TIER3 (9 dim) + Tier3β (3 dim) + TIER3_MACRO (5 dim)
    # 理由: 跟 MacroTransformer 职责分离
    #   - TIER3 (industry alpha/momentum/persistence) 跟 MacroTransformer industry_alpha 重复
    #   - Tier3β (industry beta) 跟 MacroTransformer industry_beta 重复
    #   - TIER3_MACRO (macro bias/rotation/regime) 跟 MacroTransformer 输出重复
    # Tree 专注 stock-level 因子 (97 Tier1 + 89 Tier2 + 1 industry_id = 187 dim)
    print("\n[Industry] Adding 1 dim industry_id (跟 Macro 职责分离)...", flush=True)
    t0 = time.time()
    industry_map = load_industry_map()
    processed = add_industry_features(processed, industry_map)
    print(f"[Industry] industry_id done in {time.time()-t0:.1f}s", flush=True)

    # 构建 (date, stock_code, features, label) 样本 - ThreadPoolExecutor 避免 IPC
    print("\n[BuildSamples] Building per-stock windows (thread)...", flush=True)
    t0 = time.time()
    stock_groups = list(processed.groupby('股票代码', sort=False))
    results_per_stock = [None] * len(stock_groups)
    with ThreadPoolExecutor(max_workers=n_thread) as ex:
        futures = {ex.submit(process_one_stock, sg): i for i, sg in enumerate(stock_groups)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="windows", mininterval=2.0):
            results_per_stock[futures[fut]] = fut.result()
    print(f"[BuildSamples] done in {time.time()-t0:.1f}s", flush=True)
    flat = [item for sub in results_per_stock for item in sub]
    print(f"[BuildSamples] {len(flat):,} samples", flush=True)

    # 转为 DataFrame, 保留 (date, stock_code) 用于 OOF 对齐
    print("\n[DataFrame] Assembling feature matrix...")
    feature_names = list(flat[0][2].keys())
    print(f"[DataFrame] {len(feature_names)} feature columns")
    dates_arr = np.array([x[0] for x in flat])
    stock_arr = np.array([x[1] for x in flat])
    instr_arr = np.array([stockid2idx[s] for s in stock_arr], dtype=np.int32)
    feats_arr = np.array([[x[2][k] for k in feature_names] for x in flat], dtype=np.float32)
    labels_arr = np.array([x[3] for x in flat], dtype=np.float32)
    df_all = pd.DataFrame({
        'date': dates_arr,
        'stock_code': stock_arr,
        'instrument': instr_arr,
    })
    print(f"[DataFrame] features={feats_arr.shape}, labels range [{labels_arr.min():.3f}, {labels_arr.max():.3f}]")

    # 切分 train / val / test
    val_start_str = val_start.strftime('%Y-%m-%d')
    val_end_str = pd.Timestamp(config['val_end']).strftime('%Y-%m-%d')
    test_start_str = pd.Timestamp(config['test_start']).strftime('%Y-%m-%d')
    test_end_str = pd.Timestamp(config['test_end']).strftime('%Y-%m-%d')
    train_mask = dates_arr < val_start_str
    val_mask = (dates_arr >= val_start_str) & (dates_arr <= val_end_str)
    val_mask = np.isin(dates_arr, list(val_target_dates))
    print(f"[Split] train: {train_mask.sum():,}, val: {val_mask.sum():,}", flush=True)

    X_train_raw = feats_arr[train_mask]
    X_val_raw = feats_arr[val_mask]  # val 仅用于 OOF + early stop (不参与训练)
    X_inner_val_raw = feats_arr[val_mask]  # val 用于 LGBM early stop
    y_train = labels_arr[train_mask]
    y_val = labels_arr[val_mask]
    y_inner_val = labels_arr[val_mask]
    train_dates_lgb = dates_arr[train_mask]
    val_dates_lgb = dates_arr[val_mask]
    inner_val_dates_lgb = dates_arr[val_mask]
    train_instr = instr_arr[train_mask]
    val_instr = instr_arr[val_mask]
    inner_val_instr = instr_arr[val_mask]
    train_stock = stock_arr[train_mask]
    val_stock = stock_arr[val_mask]
    inner_val_stock = stock_arr[val_mask]

    # StandardScaler
    print("\n[Scaler] Fitting StandardScaler on train...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(args.output_dir, 'feature_names.pkl'))

    # LGB group
    print("\n[Group] Computing date groups...")
    train_groups = pd.Series(train_dates_lgb).value_counts(sort=False).sort_index()
    val_groups = pd.Series(val_dates_lgb).value_counts(sort=False).sort_index()
    inner_val_groups = pd.Series(inner_val_dates_lgb).value_counts(sort=False).sort_index()
    # 必须按 date 升序排 groups
    train_group_sizes = train_groups.values
    val_group_sizes = val_groups.values
    inner_val_group_sizes = inner_val_groups.values
    print(f"[Group] train: {len(train_group_sizes)} groups, sizes min={train_group_sizes.min()} max={train_group_sizes.max()}")
    print(f"[Group] inner_val: {len(inner_val_group_sizes)} groups, sizes min={inner_val_group_sizes.min()} max={inner_val_group_sizes.max()}")
    print(f"[Group] outer_val: {len(val_group_sizes)} groups, sizes min={val_group_sizes.min()} max={val_group_sizes.max()}")

    # Load test_meta
    test_meta_paths = [
        os.path.join(os.path.dirname(args.output_dir), 'macro_transformer', 'test_meta.json'),
        str(INTEGRATED_DIR / 'macro_transformer' / 'test_meta.json'),
    ]
    test_meta = None
    for p in test_meta_paths:
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                test_meta = json.load(f)
            break
    if test_meta is None:
        print("[TestMeta] WARN: no test_meta found, train+val only", flush=True)
    test_target_dates = set(test_meta['dates']) if test_meta else set()
    test_mask_full = np.isin(dates_arr, list(test_target_dates)) if test_meta else np.zeros(len(dates_arr), dtype=bool)

    # 训 3 个 regime-specific Tree (bull_tree, sideways_tree, bear_tree)
    # 每个 model 训在 train + val (按 regime 分) 的数据
    # 推理: 3 models 各自 predict, integrated_model 按 regime_pred 选
    print("\n" + "=" * 60)
    print("[Tree] Training 3 regime-specific LightGBM (bull/sideways/bear)")
    print("=" * 60, flush=True)
    from regime_classifier import RegimeClassifier
    rc = RegimeClassifier()
    all_dates_arr = dates_arr  # 全 dates

    # 算 regime label for all dates (train + val + test)
    regime_all = rc.predict(all_dates_arr.tolist())[0]
    print(f"[Regime] total dates: {len(regime_all)}, "
          f"bear={(regime_all==0).sum()}, side={(regime_all==1).sum()}, bull={(regime_all==2).sum()}",
          flush=True)

    REGIME_NAMES = ['bear', 'sideways', 'bull']
    val_oofs = {}  # regime -> val_oof
    test_oofs = {}  # regime -> test_oof
    train_oofs_all = {}  # regime -> train_oof

    # Save val_meta.json (for backward compat with integrated_model)
    val_dates_unique = sorted(set(val_dates_lgb))
    val_meta_out = {
        'dates': val_dates_unique,
        'stock_indices': [list(range(300)) for _ in val_dates_unique],
        'n_stocks': 300,
        'seq_len': 60,
    }
    with open(os.path.join(args.output_dir, 'val_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(val_meta_out, f, ensure_ascii=False, indent=2)
    # Save test_meta.json
    if test_meta is not None:
        with open(os.path.join(args.output_dir, 'test_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(test_meta, f, ensure_ascii=False, indent=2)
    # Save train_meta.json (for integrated_model stack training)
    train_dates_unique = sorted(set(dates_arr[train_mask]))
    train_meta_out = {
        'dates': train_dates_unique,
        'stock_indices': [list(range(300)) for _ in train_dates_unique],
        'n_stocks': 300,
        'seq_len': 60,
    }
    with open(os.path.join(args.output_dir, 'train_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(train_meta_out, f, ensure_ascii=False, indent=2)

    for r in [0, 1, 2]:
        r_name = REGIME_NAMES[r]
        print(f"\n--- [Tree_{r_name}] training (regime={r}) ---", flush=True)

        # 训 data: 仅 TRAIN (按 regime 分)
        # VAL 绝对不参与训练 — 仅用于 OOF (给 stack) 和 early stop
        # TEST 仅用于最终评估
        train_dates = set(all_dates_arr[train_mask])
        val_dates = set(all_dates_arr[val_mask])
        test_dates = set(all_dates_arr[test_mask_full])
        # 训集: 仅 TRAIN 里 regime=r 的
        regime_r_train_dates = {d for d in train_dates
                                if d in all_dates_arr and regime_all[np.where(all_dates_arr == d)[0][0]] == r}
        # val 里 regime=r 的 (用于 OOF)
        regime_r_val_dates = {d for d in val_dates
                              if d in all_dates_arr and regime_all[np.where(all_dates_arr == d)[0][0]] == r}
        # test 里 regime=r 的
        regime_r_test_dates = {d for d in test_dates
                                if d in all_dates_arr and regime_all[np.where(all_dates_arr == d)[0][0]] == r}

        # 训 mask: 仅 train_mask 里 regime=r
        regime_r_mask = np.isin(dates_arr, list(regime_r_train_dates))
        # early stop mask: VAL 段 regime=r 的数据
        regime_r_eval_mask = np.isin(dates_arr, list(regime_r_val_dates))

        n_train = regime_r_mask.sum()
        n_eval = regime_r_eval_mask.sum()
        print(f"[Tree_{r_name}] train samples: {n_train:,}, eval samples: {n_eval:,}", flush=True)

        if n_train < 1000:
            print(f"[Tree_{r_name}] WARN: too few samples, skip", flush=True)
            continue

        # Build data
        X_r_train = feats_arr[regime_r_mask]
        y_r_train = labels_arr[regime_r_mask]
        dates_r_train = dates_arr[regime_r_mask]
        instr_r_train = instr_arr[regime_r_mask]

        train_groups_r = pd.Series(dates_r_train).value_counts(sort=False).sort_index().values
        cat_features = [feature_names.index('industry_id')] if 'industry_id' in feature_names else 'auto'

        train_data_r = lgb.Dataset(
            X_r_train, label=y_r_train, group=train_groups_r,
            feature_name=feature_names, categorical_feature=cat_features, free_raw_data=False,
        )

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': config.get('lgb_learning_rate', 0.05),
            'num_leaves': config.get('lgb_num_leaves', 63),
            'max_depth': config.get('lgb_max_depth', 8),
            'min_data_in_leaf': config.get('lgb_min_data_in_leaf', 50),
            'feature_fraction': config.get('lgb_feature_fraction', 0.8),
            'bagging_fraction': config.get('lgb_bagging_fraction', 0.8),
            'bagging_freq': config.get('lgb_bagging_freq', 5),
            'verbose': -1,
        }

        # 早停: 用 val 段对应 regime 的数据 (n_eval 太少就用 fixed)
        if n_eval >= 100:
            X_r_eval = feats_arr[regime_r_eval_mask]
            y_r_eval = labels_arr[regime_r_eval_mask]
            dates_r_eval = dates_arr[regime_r_eval_mask]
            eval_groups_r = pd.Series(dates_r_eval).value_counts(sort=False).sort_index().values
            eval_data_r = lgb.Dataset(
                X_r_eval, label=y_r_eval, group=eval_groups_r,
                feature_name=feature_names, categorical_feature=cat_features,
                reference=train_data_r, free_raw_data=False,
            )
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)]
            valid_sets = [train_data_r, eval_data_r]
            valid_names = ['train', 'eval']
        else:
            callbacks = [lgb.log_evaluation(period=0)]
            valid_sets = [train_data_r]
            valid_names = ['train']
            print(f"[Tree_{r_name}] no eval data, train fixed 200 rounds", flush=True)

        t0 = time.time()
        model_r = lgb.train(
            params, train_data_r,
            num_boost_round=200 if n_eval < 100 else config.get('lgb_n_estimators', 500),
            valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
        )
        print(f"[Tree_{r_name}] done in {time.time()-t0:.1f}s, best_iter={model_r.best_iteration}", flush=True)

        # 保存 model
        model_r.save_model(os.path.join(args.model_dir, f'model_{r_name}.txt'))

        # 特征重要性
        importance = pd.DataFrame({
            'feature': feature_names,
            'gain': model_r.feature_importance(importance_type='gain'),
            'split': model_r.feature_importance(importance_type='split'),
        }).sort_values('gain', ascending=False)
        importance.to_csv(os.path.join(args.output_dir, f'feature_importance_{r_name}.csv'), index=False)
        print(f"[Tree_{r_name}] Top-5 by gain: {importance.head(5)['feature'].tolist()}", flush=True)

        # OOF: predict on val (全 val, integrated_model 按 regime 取)
        print(f"[OOF] Tree_{r_name} predicting on val (full 6 月) ...", flush=True)
        val_pred = model_r.predict(X_val)
        val_oof_r = np.full((len(val_dates_unique), 300), np.nan, dtype=np.float32)
        pred_map = {}
        for d, iidx, p in zip(val_dates_lgb, val_instr, val_pred):
            pred_map[(d, int(iidx))] = float(p)
        for d_idx, d in enumerate(val_dates_unique):
            for iidx in range(300):
                if (d, iidx) in pred_map:
                    val_oof_r[d_idx, iidx] = pred_map[(d, iidx)]
        val_oofs[r] = val_oof_r
        np.save(os.path.join(args.output_dir, f'val_oof_{r_name}.npy'), val_oof_r)
        print(f"[OOF] val_{r_name} saved, shape={val_oof_r.shape}, NaN={np.isnan(val_oof_r).mean():.3f}",
              flush=True)

        # OOF: train (for stack model training — 在 TRAIN 上训 stack)
        print(f"[OOF] Tree_{r_name} predicting on train ...", flush=True)
        train_pred = model_r.predict(X_train_raw)
        train_dates_unique = sorted(set(dates_arr[train_mask]))
        train_oof_r = np.full((len(train_dates_unique), 300), np.nan, dtype=np.float32)
        pred_map_tr = {}
        for d, iidx, p in zip(train_dates_lgb, train_instr, train_pred):
            pred_map_tr[(d, int(iidx))] = float(p)
        for d_idx, d in enumerate(train_dates_unique):
            for iidx in range(300):
                if (d, iidx) in pred_map_tr:
                    train_oof_r[d_idx, iidx] = pred_map_tr[(d, iidx)]
        np.save(os.path.join(args.output_dir, f'train_oof_{r_name}.npy'), train_oof_r)
        print(f"[OOF] train_{r_name} saved, shape={train_oof_r.shape}, NaN={np.isnan(train_oof_r).mean():.3f}",
              flush=True)

        train_oofs_all[r] = train_oof_r

        # OOF: test
        if test_meta is not None and test_mask_full.sum() > 0:
            X_test_raw = feats_arr[test_mask_full]
            test_pred = model_r.predict(X_test_raw)
            test_dates_lgb_r = dates_arr[test_mask_full]
            test_instr_r = instr_arr[test_mask_full]
            test_oof_r = np.full((len(test_meta['dates']), 300), np.nan, dtype=np.float32)
            pred_map_t = {}
            for d, iidx, p in zip(test_dates_lgb_r, test_instr_r, test_pred):
                pred_map_t[(d, int(iidx))] = float(p)
            for d_idx, d in enumerate(test_meta['dates']):
                for iidx in test_meta['stock_indices'][d_idx]:
                    if (d, iidx) in pred_map_t:
                        test_oof_r[d_idx, iidx] = pred_map_t[(d, iidx)]
            test_oofs[r] = test_oof_r
            np.save(os.path.join(args.output_dir, f'oof_scores_{r_name}.npy'), test_oof_r)
            print(f"[OOF] test_{r_name} saved, shape={test_oof_r.shape}, NaN={np.isnan(test_oof_r).mean():.3f}",
                  flush=True)

    # 兼容老代码: 输出 val_oof.npy + oof_scores.npy (用 sideways model fallback)
    fallback_r = 1  # sideways
    if fallback_r in val_oofs:
        np.save(os.path.join(args.output_dir, 'val_oof.npy'), val_oofs[fallback_r])
        print(f"[Compat] val_oof.npy = sideways Tree OOF (fallback)", flush=True)
    if fallback_r in test_oofs:
        np.save(os.path.join(args.output_dir, 'oof_scores.npy'), test_oofs[fallback_r])
        print(f"[Compat] oof_scores.npy = sideways Tree OOF (fallback)", flush=True)

    # ========== 评估指标 (per regime Tree IC) ==========
    print("\n[Eval] Per-regime Tree IC on val (5d forward return):", flush=True)
    from scipy.stats import pearsonr
    true_map = {}
    for d, iidx, lbl in zip(val_dates_lgb, val_instr, y_val):
        true_map[(d, int(iidx))] = float(lbl)
    for r, val_oof_r in val_oofs.items():
        r_name = REGIME_NAMES[r]
        ics = []
        for d_idx, d in enumerate(val_dates_unique):
            preds, trues = [], []
            for iidx in range(300):
                if not np.isnan(val_oof_r[d_idx, iidx]) and (d, iidx) in true_map:
                    preds.append(val_oof_r[d_idx, iidx])
                    trues.append(true_map[(d, iidx)])
            if len(preds) < 30:
                continue
            preds, trues = np.array(preds), np.array(trues)
            if np.std(preds) > 1e-9 and np.std(trues) > 1e-9:
                ic, _ = pearsonr(preds, trues)
                if not np.isnan(ic):
                    ics.append(ic)
        if ics:
            print(f"  Tree_{r_name:9s}: IC mean={np.mean(ics):+.4f}, std={np.std(ics):.4f}, "
                  f"n_days={len(ics)}, pos={(np.array(ics)>0).sum()}/{len(ics)}", flush=True)

    print("\n[Phase B] 3 regime-specific LightGBM OOF done. Ready for Phase D ensemble.")


if __name__ == '__main__':
    main()
