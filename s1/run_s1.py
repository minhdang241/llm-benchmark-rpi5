#!/usr/bin/env python3
"""Run S1 feasibility-map cells.

The runner is intentionally separate from the expired C1-C4 benchmark code.
It reuses only the llama.cpp timing parsing approach and makes OS-level peak
RSS from `/usr/bin/time` the source of truth for memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only outside the venv
    yaml = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s1.parsers import parse_dllama_output, parse_llamacpp_output, parse_time_output
from s1.prompts import get_prompt
from s1.report import write_summary
from s1.system import host_metadata, read_vmstat, swap_delta, total_memory_kb


@dataclass
class Cell:
    model_id: str
    model: str
    role: str
    quant: str
    model_path: Path
    tokenizer_path: Path | None = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run S1 feasibility-map benchmark")
    parser.add_argument("--manifest", default="s1/manifest.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--only", action="append", default=[], help="Run cells containing this text; repeatable")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of expanded cells to process")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--warmup-reps", type=int, default=None)
    parser.add_argument("--measured-reps", type=int, default=None)
    parser.add_argument("--generated-tokens", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    root = manifest_path.resolve().parents[1]
    experiment = manifest["experiment"]
    runtime = dict(manifest["runtime"])
    if args.generated_tokens is not None:
        runtime["generated_tokens"] = args.generated_tokens
    output_dir = Path(args.output_dir or experiment["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    cells = expand_cells(manifest, root)
    if args.only:
        cells = [
            cell
            for cell in cells
            if all(needle.lower() in cell_key(cell).lower() for needle in args.only)
        ]
    if args.limit:
        cells = cells[: args.limit]

    prompt = get_prompt(runtime.get("prompt", "s1_pp512"))
    measured_reps = (
        args.measured_reps
        if args.measured_reps is not None
        else int(experiment.get("measured_repetitions", 5))
    )
    warmup_reps = (
        args.warmup_reps
        if args.warmup_reps is not None
        else int(experiment.get("warmup_repetitions", 1))
    )
    timeout_seconds = int(experiment.get("timeout_seconds", 1800))
    total_reps = warmup_reps + measured_reps
    record_missing = bool(experiment.get("record_missing_models", True))

    print(f"S1 cells: {len(cells)}")
    print(f"Output: {run_dir}")

    rows: list[dict] = []
    for cell in cells:
        missing_path = first_missing_path(cell)
        if missing_path:
            message = f"MISSING {cell.model_id} {cell.quant}: {missing_path}"
            if args.skip_missing or not record_missing:
                print(f"skip {message}")
                continue
            print(message)
            for row in missing_rows(cell, runtime, experiment, total_reps):
                rows.append(row)
                append_csv(run_dir / "s1_results.csv", row)
                append_jsonl(run_dir / "s1_cells.jsonl", row)
            continue

        command = build_command(runtime, cell, prompt)
        if args.dry_run:
            print(f"{cell_key(cell)}")
            print("  " + shlex.join(command))
            continue

        for rep_index in range(total_reps):
            phase = "warmup" if rep_index < warmup_reps else "measured"
            print(f"{cell_key(cell)} rep={rep_index + 1}/{total_reps} phase={phase}")
            row = run_one(cell, runtime, experiment, command, rep_index, phase, timeout_seconds, run_dir)
            rows.append(row)
            append_csv(run_dir / "s1_results.csv", row)
            append_jsonl(run_dir / "s1_cells.jsonl", row)

    if args.dry_run:
        return 0

    if rows and not args.no_summary:
        summary_path = write_summary(rows, run_dir)
        print(f"Summary: {summary_path}")
        print(f"Heatmap: {run_dir / 's1_feasibility_heatmap.svg'}")

    return 0


def load_manifest(path: Path) -> dict:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read s1/manifest.yaml. "
            "Run with venv/bin/python or install requirements.txt."
        )
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expand_cells(manifest: dict, root: Path) -> list[Cell]:
    cells: list[Cell] = []
    for entry in manifest.get("cells", []):
        for quant in entry["quants"]:
            format_values = {
                "quant": quant,
                "quant_lower": quant.lower(),
                "model_id": entry["model_id"],
            }
            model_path = Path(entry["path_template"].format(**format_values))
            if not model_path.is_absolute():
                model_path = root / model_path
            tokenizer_path = None
            if entry.get("tokenizer_template"):
                tokenizer_path = Path(entry["tokenizer_template"].format(**format_values))
                if not tokenizer_path.is_absolute():
                    tokenizer_path = root / tokenizer_path
            cells.append(
                Cell(
                    model_id=entry["model_id"],
                    model=entry.get("model", entry["model_id"]),
                    role=entry.get("role", ""),
                    quant=quant,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                )
            )
    return cells


def build_command(runtime: dict, cell: Cell, prompt: str) -> list[str]:
    runtime_name = runtime.get("name", "llama.cpp")
    if runtime_name == "llama.cpp":
        return build_llamacpp_command(runtime, cell.model_path, prompt)
    if runtime_name == "distributed_llama":
        if cell.tokenizer_path is None:
            raise ValueError(f"dllama cell requires tokenizer_template: {cell_key(cell)}")
        return build_dllama_command(runtime, cell.model_path, cell.tokenizer_path, prompt)
    raise ValueError(f"Unknown runtime name: {runtime_name}")


def build_llamacpp_command(runtime: dict, model_path: Path, prompt: str) -> list[str]:
    command = [
        runtime["binary"],
        "-m",
        str(model_path),
        "-n",
        str(runtime.get("generated_tokens", 128)),
        "-t",
        str(runtime.get("threads", 4)),
        "--temp",
        str(runtime.get("temperature", 0.0)),
        "-c",
        str(runtime.get("context_size", 2048)),
    ]
    command.extend(str(arg) for arg in runtime.get("extra_args", []))
    command.extend(["-p", prompt])
    return command


def build_dllama_command(
    runtime: dict, model_path: Path, tokenizer_path: Path, prompt: str
) -> list[str]:
    command = [
        runtime["binary"],
        "inference",
        "--model",
        str(model_path),
        "--tokenizer",
        str(tokenizer_path),
        "--steps",
        str(runtime.get("generated_tokens", 128)),
        "--temperature",
        str(runtime.get("temperature", 0.0)),
        "--nthreads",
        str(runtime.get("threads", 4)),
    ]
    if runtime.get("buffer_float_type"):
        command.extend(["--buffer-float-type", str(runtime["buffer_float_type"])])
    command.extend(str(arg) for arg in runtime.get("extra_args", []))
    command.extend(["--prompt", prompt])
    return command


def run_one(
    cell: Cell,
    runtime: dict,
    experiment: dict,
    command: list[str],
    rep_index: int,
    phase: str,
    timeout_seconds: int,
    run_dir: Path,
) -> dict:
    cell_dir = run_dir / "raw" / safe_name(cell_key(cell)) / f"{rep_index:02d}_{phase}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = cell_dir / "stdout.txt"
    stderr_path = cell_dir / "stderr.txt"
    time_path = cell_dir / "time.txt"
    command_path = cell_dir / "command.json"
    command_path.write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")

    before_swap = read_vmstat()
    start = time.monotonic()
    timed_out = False
    proc_returncode: Optional[int] = None
    stdout_text = ""
    stderr_text = ""
    time_text = ""

    wrapped = wrap_with_time(command, time_path)
    try:
        completed = subprocess.run(
            wrapped.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        proc_returncode = completed.returncode
        stdout_text = completed.stdout
        stderr_text = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc_returncode = -1
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""

    wall_time_ms = round((time.monotonic() - start) * 1000, 3)
    after_swap = read_vmstat()
    pgswapin_delta, pgswapout_delta = swap_delta(before_swap, after_swap)

    if wrapped.time_in_stderr:
        time_text = stderr_text
    elif time_path.exists():
        time_text = time_path.read_text(encoding="utf-8", errors="replace")

    stdout_path.write_text(stdout_text, encoding="utf-8", errors="replace") # type: ignore
    stderr_path.write_text(stderr_text, encoding="utf-8", errors="replace") # type: ignore
    if wrapped.time_in_stderr:
        time_path.write_text(time_text, encoding="utf-8", errors="replace") # type: ignore

    metrics = parse_runtime_output(runtime, stderr_text, stdout_text) # type: ignore
    time_metrics = parse_time_output(time_text, sys.platform) # type: ignore
    mem_total_kb = total_memory_kb()
    verdict = classify(
        returncode=proc_returncode,
        timed_out=timed_out,
        stderr_text=stderr_text + "\n" + time_text, # type: ignore
        max_rss_kb=time_metrics.max_rss_kb,
        mem_total_kb=mem_total_kb,
        tight_fraction=float(experiment.get("tight_rss_fraction", 0.90)),
        pgswapin_delta=pgswapin_delta,
        pgswapout_delta=pgswapout_delta,
    )

    row = base_row(cell, runtime, experiment, rep_index, phase)
    row.update(host_metadata())
    row.update(
        {
            "command": shlex.join(command),
            "returncode": proc_returncode,
            "verdict": verdict,
            "timed_out": timed_out,
            "wall_time_ms": wall_time_ms,
            "max_rss_kb": time_metrics.max_rss_kb,
            "max_rss_mb": round(time_metrics.max_rss_kb / 1024, 3)
            if time_metrics.max_rss_kb is not None
            else "",
            "mem_total_kb": mem_total_kb or "",
            "rss_fraction": round(time_metrics.max_rss_kb / mem_total_kb, 5)
            if time_metrics.max_rss_kb and mem_total_kb
            else "",
            "time_rss_unit": time_metrics.raw_unit,
            "pgswapin_delta": pgswapin_delta if pgswapin_delta is not None else "",
            "pgswapout_delta": pgswapout_delta if pgswapout_delta is not None else "",
            "load_time_ms": value_or_blank(metrics.load_time_ms),
            "prompt_eval_time_ms": value_or_blank(metrics.prompt_eval_time_ms),
            "prompt_tokens": value_or_blank(metrics.prompt_tokens),
            "prompt_rate_tps": value_or_blank(metrics.prompt_rate_tps),
            "eval_time_ms": value_or_blank(metrics.eval_time_ms),
            "eval_tokens": value_or_blank(metrics.eval_tokens),
            "eval_rate_tps": value_or_blank(metrics.eval_rate_tps),
            "total_time_ms": value_or_blank(metrics.total_time_ms),
            "total_tokens": value_or_blank(metrics.total_tokens),
            "ttft_ms": value_or_blank(metrics.ttft_ms),
            "time_elapsed_seconds": value_or_blank(time_metrics.elapsed_seconds),
            "time_exit_status": value_or_blank(time_metrics.exit_status),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "time_path": str(time_path),
            "parse_errors": "; ".join(metrics.parse_errors + time_metrics.parse_errors),
        }
    )
    return row


def parse_runtime_output(runtime: dict, stderr_text: str, stdout_text: str):
    runtime_name = runtime.get("name", "llama.cpp")
    if runtime_name == "llama.cpp":
        return parse_llamacpp_output(stderr_text, stdout_text)
    if runtime_name == "distributed_llama":
        return parse_dllama_output(stderr_text, stdout_text)
    raise ValueError(f"Unknown runtime name: {runtime_name}")


@dataclass
class TimeWrapper:
    command: list[str]
    time_in_stderr: bool


def wrap_with_time(command: list[str], time_path: Path) -> TimeWrapper:
    if sys.platform == "darwin":
        return TimeWrapper(["/usr/bin/time", "-l", *command], time_in_stderr=True)
    return TimeWrapper(["/usr/bin/time", "-v", "-o", str(time_path), *command], time_in_stderr=False)


def classify(
    returncode: Optional[int],
    timed_out: bool,
    stderr_text: str,
    max_rss_kb: Optional[int],
    mem_total_kb: Optional[int],
    tight_fraction: float,
    pgswapin_delta: Optional[int],
    pgswapout_delta: Optional[int],
) -> str:
    lowered = stderr_text.lower()
    if timed_out:
        return "TIMEOUT"
    if returncode not in {0, None} and (
        returncode in {-9, 137}
        or any(
            marker in lowered
            for marker in [
                "out of memory",
                "cannot allocate memory",
                "failed to allocate",
                "allocation failed",
                "std::bad_alloc",
                "killed",
                "oom",
            ]
        )
    ):
        return "OOM"
    if returncode != 0:
        return "FAIL"
    if (pgswapin_delta and pgswapin_delta > 0) or (pgswapout_delta and pgswapout_delta > 0):
        return "SWAP"
    if max_rss_kb and mem_total_kb and max_rss_kb >= tight_fraction * mem_total_kb:
        return "FITS_TIGHT"
    return "FITS"


def missing_rows(cell: Cell, runtime: dict, experiment: dict, total_reps: int) -> list[dict]:
    rows = []
    for rep_index in range(total_reps):
        phase = "warmup" if rep_index < int(experiment.get("warmup_repetitions", 1)) else "measured"
        row = base_row(cell, runtime, experiment, rep_index, phase)
        row.update(host_metadata())
        row.update(
            {
                "command": "",
                "returncode": "",
                "verdict": "MISSING_MODEL",
                "timed_out": False,
                "wall_time_ms": "",
                "max_rss_kb": "",
                "max_rss_mb": "",
                "mem_total_kb": total_memory_kb() or "",
                "rss_fraction": "",
                "time_rss_unit": "",
                "pgswapin_delta": "",
                "pgswapout_delta": "",
                "load_time_ms": "",
                "prompt_eval_time_ms": "",
                "prompt_tokens": "",
                "prompt_rate_tps": "",
                "eval_time_ms": "",
                "eval_tokens": "",
                "eval_rate_tps": "",
                "total_time_ms": "",
                "total_tokens": "",
                "ttft_ms": "",
                "time_elapsed_seconds": "",
                "time_exit_status": "",
                "stdout_path": "",
                "stderr_path": "",
                "time_path": "",
                "parse_errors": f"model file not found: {cell.model_path}",
            }
        )
        rows.append(row)
    return rows


def base_row(cell: Cell, runtime: dict, experiment: dict, rep_index: int, phase: str) -> dict:
    return {
        "experiment_id": experiment.get("id", "S1"),
        "runtime": runtime.get("name", "llama.cpp"),
        "model_id": cell.model_id,
        "model": cell.model,
        "role": cell.role,
        "quant": cell.quant,
        "model_path": str(cell.model_path),
        "tokenizer_path": str(cell.tokenizer_path) if cell.tokenizer_path else "",
        "rep_index": rep_index,
        "phase": phase,
    }


def first_missing_path(cell: Cell) -> Path | None:
    if not cell.model_path.exists():
        return cell.model_path
    if cell.tokenizer_path is not None and not cell.tokenizer_path.exists():
        return cell.tokenizer_path
    return None


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def value_or_blank(value):
    return "" if value is None else value


def cell_key(cell: Cell) -> str:
    return f"{cell.model_id}:{cell.quant}"


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
