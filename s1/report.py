"""Reporting helpers for S1."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


VERDICT_RANK = {
    "FITS": 0,
    "FITS_TIGHT": 1,
    "SWAP": 2,
    "TIMEOUT": 3,
    "OOM": 4,
    "FAIL": 5,
    "MISSING_MODEL": 6,
}

VERDICT_COLOR = {
    "FITS": "#c7e9c0",
    "FITS_TIGHT": "#fee391",
    "SWAP": "#fdae6b",
    "TIMEOUT": "#bcbddc",
    "OOM": "#fb6a4a",
    "FAIL": "#9e9ac8",
    "MISSING_MODEL": "#d9d9d9",
}


def write_summary(rows: list[dict], output_dir: Path) -> Path:
    measured = [row for row in rows if row.get("phase") == "measured"]
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in measured:
        grouped[(row.get("runtime", ""), row["model_id"], row["quant"])].append(row)

    summary_rows = []
    for (runtime, model_id, quant), group in sorted(grouped.items()):
        verdict = worst_verdict(row["verdict"] for row in group)
        summary_rows.append(
            {
                "runtime": runtime,
                "runtime_threads": group[0].get("runtime_threads", ""),
                "model_id": model_id,
                "model": group[0].get("model", ""),
                "role": group[0].get("role", ""),
                "quant": quant,
                "runs": len(group),
                "verdict": verdict,
                "peak_rss_mb": max_float(group, "max_rss_mb"),
                "mean_prefill_tps": mean_float(group, "prompt_rate_tps"),
                "mean_decode_tps": mean_float(group, "eval_rate_tps"),
                "mean_ttft_ms": mean_float(group, "ttft_ms"),
                "mean_cpu_utilization": mean_float(group, "cpu_utilization"),
                "mean_cpu_utilization_percent": mean_float(group, "cpu_utilization_percent"),
                "mean_cpu_utilization_threads_percent": mean_float(
                    group, "cpu_utilization_threads_percent"
                ),
                "max_pgswapin_delta": max_float(group, "pgswapin_delta"),
                "max_pgswapout_delta": max_float(group, "pgswapout_delta"),
            }
        )

    path = output_dir / "s1_summary.csv"
    write_csv(path, summary_rows)
    write_heatmap_svg(summary_rows, output_dir / "s1_feasibility_heatmap.svg")
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def worst_verdict(verdicts: Iterable[str]) -> str:
    return max(verdicts, key=lambda verdict: VERDICT_RANK.get(verdict, 99))


def mean_float(rows: list[dict], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return f"{mean(values):.3f}" if values else ""


def max_float(rows: list[dict], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", None}]
    return f"{max(values):.3f}" if values else ""


def write_heatmap_svg(rows: list[dict], path: Path) -> None:
    models = sorted({row["model_id"] for row in rows})
    quants = sorted({row["quant"] for row in rows})
    cell_w = 118
    cell_h = 34
    left = 170
    top = 55
    width = left + cell_w * len(quants) + 20
    height = top + cell_h * len(models) + 55
    verdict_by_cell = {(row["model_id"], row["quant"]): row["verdict"] for row in rows}

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="16" y="28" font-family="Arial" font-size="18" font-weight="700">S1 feasibility heatmap</text>',
    ]

    for index, quant in enumerate(quants):
        x = left + index * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x}" y="47" text-anchor="middle" font-family="Arial" font-size="12">{quant}</text>'
        )

    for row_index, model_id in enumerate(models):
        y = top + row_index * cell_h
        lines.append(
            f'<text x="12" y="{y + 22}" font-family="Arial" font-size="12">{model_id}</text>'
        )
        for col_index, quant in enumerate(quants):
            x = left + col_index * cell_w
            verdict = verdict_by_cell.get((model_id, quant), "")
            color = VERDICT_COLOR.get(verdict, "#ffffff")
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color}" stroke="#ffffff"/>'
            )
            lines.append(
                f'<text x="{x + cell_w / 2}" y="{y + 21}" text-anchor="middle" font-family="Arial" font-size="11">{verdict}</text>'
            )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
