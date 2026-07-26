"""项目根目录与常用路径常量.

所有路径均基于此模块推导, 避免硬编码. 调用:
    from paths import PROJECT_ROOT, DATA_DIR, OUTPUT_DIR, STOCK_DATA_CSV
"""
from pathlib import Path

# code/src/paths.py -> 3 级上溯 = 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'   # 推理产物 (predictions, scores, backtest, result.csv)
MODEL_DIR = PROJECT_ROOT / 'model'     # 训练产出的模型 weights
CODE_DIR = PROJECT_ROOT / 'code'
SRC_DIR = CODE_DIR / 'src'

# 唯一数据源: data/stock_data.csv (原始完整数据, 2015-01-05 ~ 2026-06-26, 300 股)
# 严格无重叠 3 段切分见 config.SPLITS:
#   train 2015-01-05 ~ 2025-06-30
#   val   2025-07-01 ~ 2025-12-31
#   test  2026-01-01 ~ 2026-06-26
STOCK_DATA_CSV = DATA_DIR / 'stock_data.csv'
INDEX_CSV = DATA_DIR / 'index_data.csv'
INDUSTRY_XLSX = DATA_DIR / '行业分类.xlsx'
HS300_LIST_CSV = DATA_DIR / 'hs300_stock_list.csv'

# 集成学习目录约定:
#   MODEL_INTEGRATED_DIR = 训练出的模型 weights (.pth / .txt / .pkl)
#   INTEGRATED_DIR       = 推理产物 (predictions, scores, backtest, result.csv)
MODEL_INTEGRATED_DIR = MODEL_DIR / 'integrated_v1'
INTEGRATED_DIR = OUTPUT_DIR / 'integrated_v1'

# 向后兼容别名 (旧代码可能仍引用)
TRAIN_CSV = STOCK_DATA_CSV
TEST_CSV = STOCK_DATA_CSV
RAW_CSV = STOCK_DATA_CSV


def ensure_dir(p):
    """创建目录 (如不存在), 接受 Path 或 str."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

