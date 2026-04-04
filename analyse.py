#!/usr/bin/env python3
"""
Quick summary analysis of benchmark results CSV.

Usage:
    python3 analyse.py /path/to/C1_results.csv
    python3 analyse.py /path/to/C1_results.csv /path/to/C2_results.csv
"""

import csv
import sys
from collections import defaultdict


def load_csv(filepath: str) -> list:
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def summarise(rows: list):
    """Print a summary table grouped by model × prompt."""

    # Filter out warm-up runs
    data = [r for r in rows if r.get("is_warmup", "").lower() not in ("true", "1", "yes")]

    if not data:
        print("  No non-warmup data found.")
        return

    config_id = data[0].get("config_id", "?")
    framework = data[0].get("framework", "?")

    print(f"\n{'=' * 80}")
    print(f"  Config: {config_id} | Framework: {framework}")
    print(f"  Non-warmup rows: {len(data)}")
    print(f"{'=' * 80}")

    # Group by model
    by_model = defaultdict(list)
    for r in data:
        by_model[r.get("model_id", "unknown")].append(r)

    for model_id, model_rows in sorted(by_model.items()):
        tier = model_rows[0].get("model_tier", "?")
        desc = model_rows[0].get("model_description", model_id)

        print(f"\n  ── {desc} (Tier {tier}) ──")
        print(f"  {'Prompt':<8} {'Eval tok/s':>10} {'Prompt tok/s':>12} "
              f"{'TTFT ms':>9} {'E2E ms':>9} {'CPU%':>6} {'Mem MB':>8}")
        print(f"  {'─' * 8} {'─' * 10} {'─' * 12} {'─' * 9} {'─' * 9} {'─' * 6} {'─' * 8}")

        # Group by prompt within this model
        by_prompt = defaultdict(list)
        for r in model_rows:
            by_prompt[r.get("prompt_id", "?")].append(r)

        model_eval_rates = []
        model_prompt_rates = []
        model_ttfts = []
        model_e2es = []
        model_cpus = []
        model_mems = []

        for prompt_id in sorted(by_prompt.keys()):
            prompt_rows = by_prompt[prompt_id]

            eval_rates = [safe_float(r["eval_rate_tps"]) for r in prompt_rows]
            prompt_rates = [safe_float(r["prompt_rate_tps"]) for r in prompt_rows]
            ttfts = [safe_float(r["ttft_ms"]) for r in prompt_rows]
            e2es = [safe_float(r["end_to_end_ms"]) for r in prompt_rows]
            cpus = [safe_float(r["avg_cpu_pct"]) for r in prompt_rows]
            mems = [safe_float(r["peak_mem_mb"]) for r in prompt_rows]

            avg = lambda lst: sum(lst) / len(lst) if lst else 0

            print(f"  {prompt_id:<8} {avg(eval_rates):>10.2f} {avg(prompt_rates):>12.2f} "
                  f"{avg(ttfts):>9.1f} {avg(e2es):>9.0f} {avg(cpus):>6.1f} {avg(mems):>8.0f}")

            model_eval_rates.extend(eval_rates)
            model_prompt_rates.extend(prompt_rates)
            model_ttfts.extend(ttfts)
            model_e2es.extend(e2es)
            model_cpus.extend(cpus)
            model_mems.extend(mems)

        avg = lambda lst: sum(lst) / len(lst) if lst else 0
        print(f"  {'AVERAGE':<8} {avg(model_eval_rates):>10.2f} {avg(model_prompt_rates):>12.2f} "
              f"{avg(model_ttfts):>9.1f} {avg(model_e2es):>9.0f} {avg(model_cpus):>6.1f} {avg(model_mems):>8.0f}")


def compare_configs(rows_list: list):
    """If multiple CSVs provided, print a side-by-side comparison."""
    configs = {}
    for rows in rows_list:
        data = [r for r in rows if r.get("is_warmup", "").lower() not in ("true", "1", "yes")]
        if data:
            cid = data[0].get("config_id", "?")
            configs[cid] = data

    if len(configs) < 2:
        return

    print(f"\n\n{'=' * 80}")
    print(f"  CROSS-CONFIG COMPARISON")
    print(f"{'=' * 80}")

    # Compare average eval rate per model across configs
    for model_id in sorted(set(r.get("model_id") for rows in configs.values() for r in rows)):
        print(f"\n  Model: {model_id}")
        for cid, data in sorted(configs.items()):
            model_rows = [r for r in data if r.get("model_id") == model_id]
            if model_rows:
                avg = lambda lst: sum(lst) / len(lst) if lst else 0
                eval_rates = [safe_float(r["eval_rate_tps"]) for r in model_rows]
                prompt_rates = [safe_float(r["prompt_rate_tps"]) for r in model_rows]
                ttfts = [safe_float(r["ttft_ms"]) for r in model_rows]
                e2es = [safe_float(r["end_to_end_ms"]) for r in model_rows]
                cpus = [safe_float(r["avg_cpu_pct"]) for r in model_rows]
                mems = [safe_float(r["peak_mem_mb"]) for r in model_rows]
                print(f"    {cid}: eval={avg(eval_rates):.2f} tok/s, prompt={avg(prompt_rates):.2f} tok/s, "
                      f"ttft={avg(ttfts):.1f} ms, e2e={avg(e2es):.0f} ms, "
                      f"cpu={avg(cpus):.1f}%, mem={avg(mems):.0f} MB")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyse.py <results.csv> [results2.csv ...]")
        sys.exit(1)

    all_rows = []
    for filepath in sys.argv[1:]:
        rows = load_csv(filepath)
        summarise(rows)
        all_rows.append(rows)

    if len(all_rows) > 1:
        compare_configs(all_rows)


if __name__ == "__main__":
    main()
