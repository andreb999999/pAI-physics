#!/usr/bin/env python3
"""
Build a human-readable local token usage report from private per-call ledger data.

This reads .local/private_token_usage/api_token_calls.jsonl by default and writes
.local/private_token_usage/token_usage_report.txt.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_ledger_path() -> str:
    env = os.getenv("FREEPHDLABOR_PRIVATE_TOKEN_LEDGER")
    if env:
        return os.path.abspath(env)
    return os.path.join(project_root(), ".local", "private_token_usage", "api_token_calls.jsonl")


def default_output_path() -> str:
    env = os.getenv("FREEPHDLABOR_PRIVATE_TOKEN_TEXT")
    if env:
        # Keep report separate from raw append log.
        base, _ = os.path.splitext(os.path.abspath(env))
        return base + "_report.txt"
    return os.path.join(project_root(), ".local", "private_token_usage", "token_usage_report.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export private token usage report")
    p.add_argument("--ledger", default=default_ledger_path(), help="Path to api_token_calls.jsonl")
    p.add_argument("--out", default=default_output_path(), help="Path to output .txt report")
    p.add_argument("--since-days", type=int, default=None, help="Only include calls from the last N days")
    p.add_argument("--max-calls", type=int, default=200, help="Number of recent calls to include in detail section")
    return p.parse_args()


def parse_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def build_report(rows: List[Dict[str, Any]], since_days: int | None, max_calls: int) -> str:
    now = datetime.now(timezone.utc)
    cutoff = None
    if since_days is not None and since_days >= 0:
        cutoff = now - timedelta(days=since_days)

    filtered: List[Dict[str, Any]] = []
    for r in rows:
        ts = parse_timestamp(str(r.get("timestamp", "")))
        if cutoff is not None and ts is not None and ts < cutoff:
            continue
        filtered.append(r)

    total_calls = len(filtered)
    total_in = sum(safe_int(r.get("input_tokens")) for r in filtered)
    total_out = sum(safe_int(r.get("output_tokens")) for r in filtered)
    total_all = total_in + total_out

    by_model: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "total": 0})
    by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "total": 0})
    by_day: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "total": 0})

    for r in filtered:
        model = str(r.get("model_id") or "unknown")
        source = str(r.get("source") or "unknown")
        ts = parse_timestamp(str(r.get("timestamp", "")))
        day = ts.date().isoformat() if ts else "unknown"
        i = safe_int(r.get("input_tokens"))
        o = safe_int(r.get("output_tokens"))
        t = i + o

        for bucket, key in ((by_model, model), (by_source, source), (by_day, day)):
            bucket[key]["calls"] += 1
            bucket[key]["input"] += i
            bucket[key]["output"] += o
            bucket[key]["total"] += t

    def format_bucket(title: str, bucket: Dict[str, Dict[str, int]], sort_key: str = "total") -> List[str]:
        lines = [title]
        lines.append("-" * len(title))
        for key, vals in sorted(bucket.items(), key=lambda kv: kv[1].get(sort_key, 0), reverse=True):
            lines.append(
                f"{key}: calls={vals['calls']}, input={vals['input']}, output={vals['output']}, total={vals['total']}"
            )
        if len(bucket) == 0:
            lines.append("(none)")
        lines.append("")
        return lines

    lines: List[str] = []
    lines.append("freephdlabor Private Token Usage Report")
    lines.append("====================================")
    lines.append(f"Generated (UTC): {now.isoformat()}")
    lines.append(f"Window: last {since_days} day(s)" if since_days is not None else "Window: all recorded calls")
    lines.append("")
    lines.append(f"Total calls: {total_calls}")
    lines.append(f"Input tokens: {total_in}")
    lines.append(f"Output tokens: {total_out}")
    lines.append(f"Total tokens: {total_all}")
    lines.append("")

    lines.extend(format_bucket("By model", by_model))
    lines.extend(format_bucket("By source", by_source))
    lines.extend(format_bucket("By day", by_day, sort_key="calls"))

    lines.append("Recent calls")
    lines.append("------------")
    recent = sorted(
        filtered,
        key=lambda r: parse_timestamp(str(r.get("timestamp", ""))) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )[: max(0, max_calls)]

    if not recent:
        lines.append("(none)")
    else:
        for r in recent:
            lines.append(
                f"{r.get('timestamp', '')} | run={r.get('run_id', '')} | source={r.get('source', '')} | "
                f"model={r.get('model_id', '') or 'unknown'} | input={safe_int(r.get('input_tokens'))} | "
                f"output={safe_int(r.get('output_tokens'))} | total={safe_int(r.get('total_tokens'))}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    ledger = os.path.abspath(args.ledger)
    out = os.path.abspath(args.out)

    rows = load_rows(ledger)
    report = build_report(rows=rows, since_days=args.since_days, max_calls=args.max_calls)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Ledger: {ledger}")
    print(f"Rows: {len(rows)}")
    print(f"Report written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
