"""A 股股票的中证一级行业映射

主数据源：data/hs300_stock_list.csv (300 只, 含 industry 列)
  - 列：updateDate / code (sh.600000) / code_name / industry (中证一级)
  - 跟 stock_data.csv 1:1 对应 (300/300 完全覆盖)
  - 由 scripts/append_industry_to_hs300.py 从 xlsx 同步生成

备用数据源：data/行业分类.xlsx (中证指数公司官方, 5627 只全 A 股)
  - 用于 hs300 缺失 industry 时的兜底
"""

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, INDUSTRY_XLSX, HS300_LIST_CSV


def normalize_code(code):
    """sh.600000 → 600000, sz.000001 → 1"""
    code = str(code).strip()
    if "." in code:
        code = code.split(".")[1]
    return code.lstrip("0") or "0"


def _load_from_hs300(hs300_path):
    """从 hs300_stock_list.csv 加载 (含 industry 列)

    返回 dict[code_int → industry]，覆盖 300/300 stock_data 股票。
    """
    df = pd.read_csv(hs300_path)
    if "industry" not in df.columns:
        return {}
    ind_map = {}
    for _, row in df.iterrows():
        code = normalize_code(row["code"])
        ind = str(row.get("industry", "")).strip()
        if ind and ind not in ("", "nan"):
            ind_map[int(code) if code.isdigit() else code] = ind
    return ind_map


def _load_from_xlsx(xlsx_path):
    """从行业分类.xlsx 加载 A 股完整行业映射 (5627 只, 备用)"""
    df = pd.read_excel(xlsx_path, sheet_name="行业分类")
    df["证券代码"] = df["证券代码"].astype(str).str.strip()
    mask_ashare = df["证券代码"].str.match(r"^\d{6}$")
    df = df[mask_ashare].copy()
    df["证券代码"] = df["证券代码"].str.zfill(6)
    df["code_int"] = df["证券代码"].astype(int)
    ind_map = {row["code_int"]: str(row["中证一级行业分类简称"]).strip()
               for _, row in df.iterrows()
               if str(row["中证一级行业分类简称"]).strip() not in ("", "nan")}
    return ind_map


def load_industry_map():
    """加载股票 → 中证一级行业 映射

    优先 hs300 list (300 只, 跟 stock_data 1:1), 缺失 fallback xlsx (5627 只).

    Returns:
        dict: {stock_code (int): industry_name}
    """
    hs300_path = HS300_LIST_CSV
    if hs300_path.exists():
        ind_map = _load_from_hs300(hs300_path)
        if ind_map:
            return ind_map

    xlsx_path = INDUSTRY_XLSX
    if xlsx_path.exists():
        return _load_from_xlsx(xlsx_path)

    raise FileNotFoundError("未找到 hs300_stock_list.csv 或 行业分类.xlsx")


if __name__ == "__main__":
    m = load_industry_map()
    print(f"总股票数: {len(m)}")
    from collections import Counter
    cnt = Counter(m.values())
    print(f"行业数: {len(cnt)}")
    for ind, n in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {ind}: {n}")
