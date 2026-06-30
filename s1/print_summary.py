#!/usr/bin/env python3
"""Pretty-print an S1 summary CSV in the terminal."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_COLUMNS = [
    ("model_id", "model"),
    ("quant", "quant"),
    ("runs", "runs"),
    ("verdict", "verdict"),
    ("peak_rss_mb", "rss_mb"),
    ("mean_prefill_tps", "prefill"),
    ("mean_decode_tps", "decode"),
    ("mean_ttft_ms", "ttft_ms"),
    ("max_pgswapin_delta", "swap_in"),
    ("max_pgswapout_delta", "swap_out"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretty-print S1 summary CSV")
    parser.add_argument("summary_csv", help="Path to logs/s1/<run>/s1_summary.csv")
    parser.add_argument("--all-columns", action="store_true")
    args = parser.parse_args()

    path = Path(args.summary_csv)
    rows = read_csv(path)
    if not rows:
        print(f"No rows in {path}")
        return 0

    if args.all_columns:
        columns = [(name, name) for name in rows[0].keys()]
    else:
        columns = [(key, label) for key, label in DEFAULT_COLUMNS if key in rows[0]]

    print_table(rows, columns)
    return 0


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def print_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    widths = {}
    for key, label in columns:
        widths[key] = len(label)
        for row in rows:
            widths[key] = max(widths[key], len(format_cell(row.get(key, ""))))

    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    rule = "  ".join("-" * widths[key] for key, _label in columns)
    print(header)
    print(rule)
    for row in rows:
        print("  ".join(format_cell(row.get(key, "")).ljust(widths[key]) for key, _label in columns))


def format_cell(value: object) -> str:
    text = "" if value is None else str(value)
    if text == "":
        return "-"
    try:
        number = float(text)
    except ValueError:
        return text

    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())

