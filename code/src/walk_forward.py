"""Walk-forward weekly training and portfolio evaluation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from eval_metrics import evaluate_portfolio, load_benchmark_open_prices, load_stock_open_prices, summarize_metrics
from experiment_dates import iter_non_overlapping_signal_dates, load_trading_dates, make_evaluation_window
from paths import INDEX_CSV, MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT, STOCK_DATA_CSV


SRC_DIR = Path(__file__).resolve().parent


def _child_path(path: Path) -> str:
    """Prefer project-relative paths for child scripts to avoid native Unicode path issues."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _stream_run(cmd: list[str]) -> None:
    print("\n[Run] " + " ".join(cmd), flush=True)
    start = time.time()
    env = os.environ.copy()
    env.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _safe_write(line)
    code = proc.wait()
    print(f"[Run] exit={code}, elapsed={time.time() - start:.1f}s", flush=True)
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def _safe_write(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    sys.stdout.flush()


def _fold_name(signal_date: str) -> str:
    return signal_date.replace("-", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "walk_forward"))
    parser.add_argument("--artifact-root", default=str(OUTPUT_DIR / "walk_forward_models"))
    parser.add_argument("--model-root", default=str(MODEL_DIR / "walk_forward_models"))
    parser.add_argument("--policy", default=str(PROJECT_ROOT / "code" / "config" / "manual_policy.json"))
    parser.add_argument("--selection", choices=["pure_topk", "industry_diversified", "single"], default="industry_diversified")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--val-days", type=int, default=21)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--macro-epochs", type=int, default=100)
    parser.add_argument("--macro-patience", type=int, default=20)
    parser.add_argument("--macro-infer-batch-size", type=int, default=128)
    parser.add_argument("--linear-epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    trading_dates = load_trading_dates(STOCK_DATA_CSV)
    signal_dates = iter_non_overlapping_signal_dates(
        trading_dates,
        args.start_date,
        args.end_date,
        step=args.horizon,
        horizon=args.horizon,
    )
    if args.smoke and args.max_folds is None:
        signal_dates = signal_dates[:1]
    elif args.max_folds is not None:
        signal_dates = signal_dates[:args.max_folds]
    if not signal_dates:
        raise ValueError("No evaluable signal dates in requested range")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    open_prices = load_stock_open_prices(STOCK_DATA_CSV)
    index_open = load_benchmark_open_prices(INDEX_CSV)
    rows = []
    metrics_by_variant = {"base": [], "manual": []}

    print(f"[WalkForward] folds={len(signal_dates)}, selection={args.selection}", flush=True)
    for signal_date in signal_dates:
        window = make_evaluation_window(trading_dates, signal_date, horizon=args.horizon)
        fold = _fold_name(window.signal_date)
        fold_output = Path(args.artifact_root) / fold
        fold_model = Path(args.model_root) / fold
        predict_dir = fold_output / f"prediction_{args.selection}"

        train_cmd = [
            sys.executable, str(SRC_DIR / "train_weekly_model.py"),
            "--signal-date", window.signal_date,
            "--output-root", str(args.artifact_root),
            "--model-root", str(args.model_root),
            "--val-days", str(args.val_days),
            "--horizon", str(args.horizon),
            "--macro-epochs", str(args.macro_epochs),
            "--macro-patience", str(args.macro_patience),
            "--macro-infer-batch-size", str(args.macro_infer_batch_size),
            "--linear-epochs", str(args.linear_epochs),
            "--seeds", *[str(seed) for seed in args.seeds],
        ]
        if args.train_start:
            train_cmd.extend(["--train-start", args.train_start])
        if args.smoke:
            train_cmd.append("--smoke")
        if args.reuse:
            train_cmd.append("--skip-existing")
        _stream_run(train_cmd)

        _stream_run([
            sys.executable, str(SRC_DIR / "predict_asof.py"),
            "--signal-date", window.signal_date,
            "--base-dir", _child_path(fold_output),
            "--model-dir", _child_path(fold_model),
            "--output-dir", _child_path(predict_dir),
            "--policy", args.policy,
            "--selection", args.selection,
        ])

        for variant, file_name in [("base", "portfolio_base.csv"), ("manual", "portfolio_manual.csv")]:
            portfolio = pd.read_csv(predict_dir / file_name, dtype={"stock_id": str})
            metric = evaluate_portfolio(
                portfolio,
                signal_date=window.signal_date,
                buy_date=window.buy_date,
                sell_date=window.sell_date,
                open_prices=open_prices,
                index_open=index_open,
                compute_ndcg=True,
            )
            metrics_by_variant[variant].append(metric)
            row = asdict(metric)
            row["variant"] = variant
            row["selection"] = args.selection
            rows.append(row)
            print(f"[Eval] {window.signal_date} {variant}: excess={metric.excess_return:+.4f}, "
                  f"win={int(metric.excess_win)}, ndcg5={metric.ndcg5}", flush=True)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    summary = {
        variant: summarize_metrics(items)
        for variant, items in metrics_by_variant.items()
    }
    summary["config"] = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "selection": args.selection,
        "horizon": args.horizon,
        "val_days": args.val_days,
        "folds": signal_dates,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Save] {output_dir / 'fold_metrics.csv'}", flush=True)
    print(f"[Save] {output_dir / 'summary.json'}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
