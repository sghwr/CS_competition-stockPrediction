import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="按日期区间将股票数据切分为 train.csv 和 test.csv"
	)
	parser.add_argument(
		"--input",
		type=str,
		default="data/stock_data.csv",
		help="原始数据文件路径，默认 data/stock_data.csv",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="data",
		help="输出目录，默认 data",
	)
	parser.add_argument(
		"--val-windows",
		type=int,
		default=60,
		help="验证集滑动窗口数量（步长1），默认 60",
	)
	parser.add_argument(
		"--future-days",
		type=int,
		default=5,
		help="标签需要的前向天数（T+1~T+future_days），默认 5",
	)
	return parser.parse_args()


def _to_timestamp(date_str: str, name: str) -> pd.Timestamp:
	ts = pd.to_datetime(date_str, errors="coerce")
	if pd.isna(ts):
		raise ValueError(f"参数 {name} 的日期格式无效: {date_str}")
	return ts.normalize()


def _validate_columns(df: pd.DataFrame) -> None:
	required = {"股票代码", "日期"}
	missing = required - set(df.columns)
	if missing:
		raise ValueError(f"输入文件缺少必要列: {sorted(missing)}")


def _filter_by_date(
	df: pd.DataFrame,
	start_date: pd.Timestamp,
	end_date: pd.Timestamp,
) -> pd.DataFrame:
	if start_date > end_date:
		raise ValueError(f"开始日期晚于结束日期: {start_date.date()} > {end_date.date()}")

	mask = (df["日期"] >= start_date) & (df["日期"] <= end_date)
	out = df.loc[mask].copy()
	out = out.sort_values(["股票代码", "日期"]).reset_index(drop=True)
	out["日期"] = out["日期"].dt.strftime("%Y-%m-%d")
	return out


def main() -> None:
	args = parse_args()

	input_path = Path(args.input)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	val_windows = args.val_windows
	future_days = args.future_days

	df = pd.read_csv(input_path)
	_validate_columns(df)

	df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
	if df["日期"].isna().any():
		bad_rows = int(df["日期"].isna().sum())
		raise ValueError(f"原始数据中存在无法解析的日期，共 {bad_rows} 行")

	# 获取所有交易日（去重排序）
	all_dates = sorted(df["日期"].unique())
	source_min_date = all_dates[0].date()
	source_max_date = all_dates[-1].date()
	n_trade_days = len(all_dates)

	# 可作为窗口结束日的日期：需要之后有 future_days 个交易日
	valid_window_dates = all_dates[: n_trade_days - future_days]
	n_valid = len(valid_window_dates)

	if val_windows > n_valid:
		raise ValueError(
			f"验证集窗口数 {val_windows} 超过可用窗口数 {n_valid}（总交易日 {n_trade_days} - 前向 {future_days} 天）"
		)

	# 验证集窗口结束日：最后 val_windows 个有效窗口日
	val_window_end_dates = valid_window_dates[-val_windows:]
	# 训练集窗口结束日：验证集之前的所有有效窗口日
	train_last_window_date = valid_window_dates[-val_windows - 1]

	# train.csv: 从数据起点到 train_last_window_date + future_days（包含标签上下文）
	train_end = all_dates[all_dates.index(train_last_window_date) + future_days]
	# test.csv: 验证集窗口需要 sequence_length-1 的历史上下文 + 窗口日 + future_days 前向
	# 这里取验证集第一个窗口日往前 90 个交易日（足够 60 序列上下文），到最后一天
	val_context_start_idx = max(0, all_dates.index(val_window_end_dates[0]) - 90)
	val_context_start = all_dates[val_context_start_idx]

	train_df = _filter_by_date(df, all_dates[0], train_end)
	test_df = _filter_by_date(df, val_context_start, pd.Timestamp(source_max_date))

	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"

	train_df.to_csv(train_path, index=False)
	test_df.to_csv(test_path, index=False)

	print(f"原始数据: {n_trade_days} 个交易日, {source_min_date} ~ {source_max_date}")
	print(f"有效窗口日: {n_valid} 个（排除最后 {future_days} 天无前向标签）")
	print(f"验证集: {val_windows} 个滑动窗口（步长1）")
	print(f"  窗口结束日范围: {val_window_end_dates[0].date()} ~ {val_window_end_dates[-1].date()}")
	print(f"训练集窗口结束日: ... ~ {train_last_window_date.date()}")
	print()
	print(f"train.csv: {len(train_df)} 行, {train_df['股票代码'].nunique()} 只股票, "
		  f"{train_df['日期'].min()} ~ {train_df['日期'].max()}")
	print(f"test.csv:  {len(test_df)} 行, {test_df['股票代码'].nunique()} 只股票, "
		  f"{test_df['日期'].min()} ~ {test_df['日期'].max()}")


if __name__ == "__main__":
	main()
