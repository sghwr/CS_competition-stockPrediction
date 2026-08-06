"""Train one weekly walk-forward fold."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_dates import (
    load_trading_dates,
    make_evaluation_window,
    previous_or_same_trade_date,
    shift_trade_date,
)
from paths import MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT, STOCK_DATA_CSV


SRC_DIR = Path(__file__).resolve().parent


def _child_path(path: Path) -> str:
    """Prefer project-relative paths for child scripts to avoid LightGBM Unicode path issues."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _split_args(split: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in split.items():
        args.extend([f"--{key}", value])
    return args


def _stream_run(cmd: list[str], cwd: Path) -> None:
    print("\n[Run] " + " ".join(cmd), flush=True)
    start = time.time()
    env = os.environ.copy()
    env.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
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
    elapsed = time.time() - start
    print(f"[Run] exit={code}, elapsed={elapsed:.1f}s", flush=True)
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def _safe_write(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
    sys.stdout.flush()


def build_weekly_split(
    signal_date: str,
    train_start: str | None,
    val_days: int,
    horizon: int,
    allow_unrevealed: bool = False,
) -> dict:
    trading_dates = load_trading_dates(STOCK_DATA_CSV)
    try:
        window = make_evaluation_window(trading_dates, signal_date, horizon=horizon)
        signal = window.signal_date
        train_label_end = window.train_label_end
        buy_date = window.buy_date
        sell_date = window.sell_date
        test_date = signal
    except ValueError:
        if not allow_unrevealed:
            raise
        signal = previous_or_same_trade_date(trading_dates, signal_date)
        train_label_end = shift_trade_date(trading_dates, signal, -horizon)
        buy_date = None
        sell_date = None
        test_date = train_label_end
    val_end = train_label_end
    val_start = shift_trade_date(trading_dates, val_end, -(val_days - 1))
    train_end = shift_trade_date(trading_dates, val_start, -1)
    return {
        "train_start": train_start or trading_dates[0],
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_date,
        "test_end": test_date,
        "signal_date": signal,
        "train_label_end": train_label_end,
        "test_proxy_date": test_date if test_date != signal else None,
        "buy_date": buy_date,
        "sell_date": sell_date,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--output-root", default=str(OUTPUT_DIR / "walk_forward_models"))
    parser.add_argument("--model-root", default=str(MODEL_DIR / "walk_forward_models"))
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--val-days", type=int, default=21)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--macro-epochs", type=int, default=100)
    parser.add_argument("--macro-patience", type=int, default=20)
    parser.add_argument("--macro-infer-batch-size", type=int, default=128)
    parser.add_argument("--linear-epochs", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--final-refit",
        action="store_true",
        help="Select epochs on train/val, then refit every branch on train+val.",
    )
    args = parser.parse_args()

    split = build_weekly_split(
        args.signal_date,
        args.train_start,
        args.val_days,
        args.horizon,
        allow_unrevealed=args.final_refit,
    )
    fold_name = split["signal_date"].replace("-", "")
    output_dir = Path(args.output_root) / fold_name
    model_dir = Path(args.model_root) / fold_name
    ensemble_scores = output_dir / "ensemble" / "final_oof_scores.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "split": split,
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "smoke": args.smoke,
        "seeds": args.seeds,
        "final_refit": args.final_refit,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(f"[Weekly] signal={split['signal_date']}, train_label_end={split['train_label_end']}", flush=True)
    print(f"[Weekly] train={split['train_start']}~{split['train_end']}, val={split['val_start']}~{split['val_end']}", flush=True)
    print(f"[Weekly] eval buy/sell={split['buy_date']}->{split['sell_date']}", flush=True)

    if args.skip_existing and ensemble_scores.exists():
        print(f"[Weekly] skip existing fold: {ensemble_scores}", flush=True)
        return

    common_split = _split_args({k: split[k] for k in ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]})
    seed_args = [str(s) for s in args.seeds]
    smoke = ["--smoke"] if args.smoke else []
    final_refit = ["--final-refit"] if args.final_refit else []
    macro_epochs = 1 if args.smoke else args.macro_epochs
    macro_patience = 1 if args.smoke else args.macro_patience
    linear_epochs = 1 if args.smoke else args.linear_epochs

    _stream_run([
        sys.executable, str(SRC_DIR / "train_transformer_branch.py"),
        "--model_dir", _child_path(model_dir / "macro_transformer"),
        "--output_dir", _child_path(output_dir / "macro_transformer"),
        "--epochs", str(macro_epochs),
        "--patience", str(macro_patience),
        "--infer-batch-size", str(args.macro_infer_batch_size),
        "--seeds", *seed_args,
        *smoke,
        *final_refit,
        *common_split,
    ], PROJECT_ROOT)
    _stream_run([
        sys.executable, str(SRC_DIR / "tree_branch.py"),
        "--model_dir", _child_path(model_dir / "tree"),
        "--output_dir", _child_path(output_dir / "tree"),
        "--val_meta_path", _child_path(output_dir / "macro_transformer" / "val_meta.json"),
        *smoke,
        *final_refit,
        *common_split,
    ], PROJECT_ROOT)
    _stream_run([
        sys.executable, str(SRC_DIR / "mdsrp_regression.py"),
        "--model_dir", _child_path(model_dir / "linear_regression"),
        "--output_dir", _child_path(output_dir / "linear_regression"),
        "--val_meta_path", _child_path(output_dir / "macro_transformer" / "val_meta.json"),
        "--epochs", str(linear_epochs),
        "--seeds", *seed_args,
        *smoke,
        *final_refit,
        *common_split,
    ], PROJECT_ROOT)
    _stream_run([
        sys.executable, str(SRC_DIR / "integrated_model.py"),
        "--base_dir", _child_path(output_dir),
        "--model_dir", _child_path(model_dir / "ensemble"),
        "--output_dir", _child_path(output_dir / "ensemble"),
        *final_refit,
        *common_split,
    ], PROJECT_ROOT)


if __name__ == "__main__":
    main()
