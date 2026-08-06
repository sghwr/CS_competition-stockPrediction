#!/usr/bin/env python3
"""Validate and freeze a time-bounded research manifest and manual policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


REQUIRED_ITEM_FIELDS = {
    "title",
    "url",
    "source",
    "published_at",
    "retrieved_at",
    "query",
    "applies_to",
    "evidence",
}
ASIA_SHANGHAI = ZoneInfo("Asia/Shanghai")


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Expected YYYY-MM-DD, got {value!r}") from exc


def parse_cutoff(value: str) -> datetime:
    if len(value) == 10:
        cutoff_date = parse_date(value)
        now = datetime.now(ASIA_SHANGHAI)
        if cutoff_date == now.date():
            return now
        return datetime.combine(cutoff_date, time.max, tzinfo=ASIA_SHANGHAI)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("news_cutoff datetime must include a timezone")
    return parsed


def published_not_later(value: str, cutoff: datetime) -> bool:
    if len(value) == 10:
        return parse_date(value) <= cutoff.date()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"published_at must include a timezone: {value!r}")
    return parsed.astimezone(timezone.utc) <= cutoff.astimezone(timezone.utc)


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_manifest(manifest: object, cutoff: datetime) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["Manifest root must be an object"]
    queries = manifest.get("queries")
    items = manifest.get("items")
    if not isinstance(queries, list) or not all(isinstance(q, str) and q.strip() for q in queries):
        errors.append("queries must be a non-empty-string array (an empty array is allowed)")
    if not isinstance(items, list):
        return errors + ["items must be an array"]

    seen_urls: set[str] = set()
    for index, item in enumerate(items):
        prefix = f"items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = REQUIRED_ITEM_FIELDS - set(item)
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
            continue
        url = str(item["url"])
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            errors.append(f"{prefix}.url is not an HTTP(S) URL")
        if url in seen_urls:
            errors.append(f"{prefix}.url is duplicated")
        seen_urls.add(url)
        for field in ("title", "source", "query", "evidence"):
            if not isinstance(item[field], str) or not item[field].strip():
                errors.append(f"{prefix}.{field} must be non-empty")
        if isinstance(queries, list) and item["query"] not in queries:
            errors.append(f"{prefix}.query is not present in top-level queries")
        if not isinstance(item["applies_to"], list):
            errors.append(f"{prefix}.applies_to must be an array")
        try:
            if not published_not_later(str(item["published_at"]), cutoff):
                errors.append(f"{prefix}.published_at exceeds news_cutoff")
        except ValueError as exc:
            errors.append(f"{prefix}.{exc}")
        try:
            retrieved = datetime.fromisoformat(str(item["retrieved_at"]).replace("Z", "+00:00"))
            if retrieved.tzinfo is None:
                errors.append(f"{prefix}.retrieved_at must include a timezone")
        except ValueError:
            errors.append(f"{prefix}.retrieved_at is not ISO-8601")
    return errors


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["test", "predict"])
    parser.add_argument("--news-cutoff", required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--buy-date", required=True)
    parser.add_argument("--sell-date", required=True)
    parser.add_argument("--news-manifest", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    cutoff = parse_cutoff(args.news_cutoff)
    signal_date = parse_date(args.signal_date)
    buy_date = parse_date(args.buy_date)
    sell_date = parse_date(args.sell_date)
    now = datetime.now(ASIA_SHANGHAI)
    today = now.date()

    errors: list[str] = []
    if cutoff > now:
        errors.append("news_cutoff cannot be later than the current time")
    if cutoff.date() >= buy_date:
        errors.append("news_cutoff date must be earlier than buy_date")
    if not signal_date < buy_date <= sell_date:
        errors.append("Expected signal_date < buy_date <= sell_date")
    if args.mode == "test" and sell_date > today:
        errors.append("test mode requires a fully ended holding window")
    if args.mode == "predict" and buy_date <= today:
        errors.append("predict mode requires buy_date later than today")

    manifest = load_json(args.news_manifest)
    policy = load_json(args.policy)
    errors.extend(validate_manifest(manifest, cutoff))
    if not isinstance(policy, dict):
        errors.append("Policy root must be an object")
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        args.output_dir / "research_manifest.json",
        args.output_dir / "manual_policy.json",
        args.output_dir / "freeze.json",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite frozen artifacts: {existing}")

    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "mode": args.mode,
        "news_cutoff": args.news_cutoff,
        "resolved_news_cutoff": cutoff.isoformat(),
        "signal_date": args.signal_date,
        "buy_date": args.buy_date,
        "sell_date": args.sell_date,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": digest(manifest),
        "policy_sha256": digest(policy),
    }
    write_new(targets[0], manifest)
    write_new(targets[1], policy)
    write_new(targets[2], freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
