"""CLI helpers for overriding train/val/test date splits."""
from __future__ import annotations


SPLIT_KEYS = ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")


def add_split_arguments(parser) -> None:
    """Add optional split override arguments to an argparse parser."""
    for key in SPLIT_KEYS:
        parser.add_argument(f"--{key}", type=str, default=None)


def apply_split_overrides(args, cfg: dict) -> dict:
    """Apply non-empty CLI split values to the shared config dict."""
    for key in SPLIT_KEYS:
        value = getattr(args, key, None)
        if value:
            cfg[key] = value
    return {key: cfg[key] for key in SPLIT_KEYS}
