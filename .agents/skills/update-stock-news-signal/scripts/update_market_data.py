#!/usr/bin/env python3
"""Run the project's stock and HS300 index data updaters."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def run(command: list[str], cwd: Path) -> None:
    print(f"[Run] {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--index-end-date", required=True)
    parser.add_argument("--index-start-date", default="2019-01-02")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    root = args.project_root.resolve()
    stock_script = root / "get_stock_data.py"
    index_script = root / "update_index_data.py"
    for script in (stock_script, index_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    end_date = date.fromisoformat(args.index_end_date)
    if end_date > datetime.now(ZoneInfo("Asia/Shanghai")).date():
        raise ValueError("index-end-date cannot be in the future")

    run([args.python, str(stock_script)], root)
    run(
        [
            args.python,
            str(index_script),
            "--start-date",
            args.index_start_date,
            "--end-date",
            args.index_end_date,
            "--output",
            str(root / "data" / "index_data.csv"),
        ],
        root,
    )


if __name__ == "__main__":
    main()
