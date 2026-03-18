#!/usr/bin/env python3
"""
Benchmark runner for C1 (llama.cpp single-device) and C2 (Distributed Llama single-node).

Usage:
    python3 benchmark.py --config C1                 # run all models for C1
    python3 benchmark.py --config C2                 # run all models for C2
    python3 benchmark.py --config C1 --model qwen3-1.7b-q4_0   # single model
    python3 benchmark.py --config C1 --dry-run       # show commands without executing

Results are saved to CSV in the output directory.
Generated text is saved separately for the consistency check.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from config import (
    CONFIGURATIONS,
    MODELS,
    MODEL_DIR,
    OUTPUT_DIR,
    NUM_RUNS,
    WARMUP_RUNS,
    TEMPERATURE,
    NUM_THREADS,
    MONITOR_INTERVAL,
)
from monitor import ResourceMonitor
from parsers import parse_llama_cpp_output, parse_dllama_output
from prompts import PROMPTS


# ============================================================
# Command builders
# ============================================================


def build_llama_cpp_cmd(
    binary: str, model_path: str, prompt: str, n_predict: int
) -> list:
    """Build the llama-cli command."""
    return [
        binary,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "--temp",
        str(TEMPERATURE),
        "-t",
        str(NUM_THREADS),
        "--simple-io",
        "--single-turn",
        "--no-display-prompt",
        "-no-cnv",
    ]


def build_dllama_cmd(binary: str, model_path: str, prompt: str, n_predict: int) -> list:
    """
    Build the dllama inference command for single-node mode.

    NOTE: Adjust flags to match your Distributed Llama version.
    Common flags:
        dllama inference --model <path> --prompt <text> --steps <n> --temperature 0
    Some versions use --nWorkers 0 for single-node.
    """
    return [
        binary,
        "inference",
        "--model",
        model_path,
        "--prompt",
        prompt,
        "--steps",
        str(n_predict),
        "--temperature",
        str(TEMPERATURE),
        "--nthreads",
        str(NUM_THREADS),
    ]


# ============================================================
# Single prompt execution
# ============================================================


def run_single_prompt(
    config_id: str, binary: str, framework: str, model_path: str, prompt_data: dict
) -> tuple:
    """
    Execute one prompt and collect all metrics.
    Returns a dict with timing + resource metrics.
    """
    prompt_text = prompt_data["text"]
    n_predict = prompt_data["n_predict"]

    # Build command
    if framework == "llama.cpp":
        cmd = build_llama_cpp_cmd(binary, model_path, prompt_text, n_predict)
        parser_fn = parse_llama_cpp_output
    elif framework == "distributed_llama":
        cmd = build_dllama_cmd(binary, model_path, prompt_text, n_predict)
        parser_fn = parse_dllama_output
    else:
        raise ValueError(f"Unknown framework: {framework}")

    # Start resource monitor
    monitor = ResourceMonitor(interval=MONITOR_INTERVAL)

    # Run inference
    wall_start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Start monitoring with the process PID
        monitor.start(pid=proc.pid)

        stdout, stderr = proc.communicate(timeout=600)  # 10 min timeout
        wall_end = time.monotonic()
        wall_time_ms = (wall_end - wall_start) * 1000.0
        returncode = proc.returncode

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        wall_time_ms = 600_000
        stdout, stderr = "", "TIMEOUT after 600s"
        returncode = -1

    except Exception as e:
        wall_time_ms = 0
        stdout, stderr = "", str(e)
        returncode = -1

    # Stop monitor and get resource metrics
    resource_metrics = monitor.stop()

    # Parse framework output
    if returncode != 0:
        parsed = {
            "load_time_ms": 0,
            "prompt_eval_time_ms": 0,
            "prompt_tokens": 0,
            "prompt_rate_tps": 0,
            "eval_time_ms": 0,
            "eval_tokens": 0,
            "eval_rate_tps": 0,
            "total_time_ms": wall_time_ms,
            "total_tokens": 0,
            "ttft_ms": 0,
            "generated_text": "",
            "parse_errors": [f"exit code {returncode}: {stderr[:500]}"],
        }
    else:
        parsed = parser_fn(stderr, stdout, wall_time_ms)

    # Combine into final result row
    result = {
        "config_id": config_id,
        "framework": framework,
        "prompt_id": prompt_data["id"],
        "prompt_name": prompt_data["name"],
        "input_category": prompt_data["input_category"],
        "output_category": prompt_data["output_category"],
        "n_predict": n_predict,
        # Timing metrics
        "ttft_ms": parsed["ttft_ms"],  # time to first token
        "eval_rate_tps": parsed["eval_rate_tps"],  # token generation rate
        "prompt_rate_tps": parsed["prompt_rate_tps"],  # prompt processing rate
        "end_to_end_ms": parsed["total_time_ms"],  # total time
        "load_time_ms": parsed["load_time_ms"],  # model load time
        "prompt_eval_time_ms": parsed["prompt_eval_time_ms"],
        "eval_time_ms": parsed["eval_time_ms"],
        "prompt_tokens": parsed["prompt_tokens"],
        "eval_tokens": parsed["eval_tokens"],
        "total_tokens": parsed["total_tokens"],
        # Resource metrics
        "avg_cpu_pct": resource_metrics["avg_cpu_pct"],  # average CPU utilization
        "max_cpu_pct": resource_metrics["max_cpu_pct"],
        "peak_mem_mb": resource_metrics["peak_mem_mb"],  # peak memory usage
        "avg_mem_mb": resource_metrics["avg_mem_mb"],  # average memory usage
        "net_rx_mb": resource_metrics["net_rx_mb"],
        "net_tx_mb": resource_metrics["net_tx_mb"],
        # Wall time from Python (backup)
        # "wall_time_ms": round(wall_time_ms, 2),
        # Metadata
        "returncode": returncode,
        "parse_errors": "; ".join(parsed.get("parse_errors", [])),
    }

    return result, parsed.get("generated_text", "")


# ============================================================
# Model load time measurement
# ============================================================


def measure_model_load_time(binary: str, framework: str, model_path: str) -> float:
    """
    Measure model load time by running a minimal prompt and extracting
    the load_time field from the output.
    """
    dummy_prompt = "Hi"
    n_predict = 1

    if framework == "llama.cpp":
        cmd = build_llama_cpp_cmd(binary, model_path, dummy_prompt, n_predict)
        parser_fn = parse_llama_cpp_output
    elif framework == "distributed_llama":
        cmd = build_dllama_cmd(binary, model_path, dummy_prompt, n_predict)
        parser_fn = parse_dllama_output
    else:
        return 0.0

    wall_start = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        wall_time_ms = (time.monotonic() - wall_start) * 1000.0
        parsed = parser_fn(proc.stderr, proc.stdout, wall_time_ms)
        return parsed["load_time_ms"] if parsed["load_time_ms"] > 0 else wall_time_ms
    except Exception as e:
        print(f"  [WARN] Failed to measure load time: {e}")
        return 0.0


# ============================================================
# CSV writer
# ============================================================

CSV_FIELDS = [
    "config_id",
    "framework",
    "model_id",
    "model_description",
    "model_tier",
    "prompt_id",
    "prompt_name",
    "input_category",
    "output_category",
    "run_number",
    "is_warmup",
    "ttft_ms",
    "eval_rate_tps",
    "prompt_rate_tps",
    "end_to_end_ms",
    "load_time_ms",
    "prompt_eval_time_ms",
    "eval_time_ms",
    "prompt_tokens",
    "eval_tokens",
    "total_tokens",
    "avg_cpu_pct",
    "max_cpu_pct",
    "peak_mem_mb",
    "avg_mem_mb",
    "net_rx_mb",
    "net_tx_mb",
    "wall_time_ms",
    "n_predict",
    "returncode",
    "parse_errors",
]


def init_csv(filepath: str):
    """Create CSV with headers."""
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv(filepath: str, row: dict):
    """Append a single row to the CSV."""
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# ============================================================
# Main benchmark loop
# ============================================================


def run_benchmark(config_id: str, model_filter: str = None, dry_run: bool = False):
    """Run the full benchmark for a given configuration."""

    config = CONFIGURATIONS[config_id]
    framework = config["framework"]
    binary = config["binary"]

    # Validate binary exists
    if not dry_run and not os.path.isfile(binary):
        print(f"ERROR: Binary not found: {binary}")
        print(f"  Edit config.py to set the correct path.")
        sys.exit(1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"{config_id}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "outputs"), exist_ok=True)

    csv_path = os.path.join(run_dir, f"{config_id}_results.csv")
    init_csv(csv_path)

    # Determine which models to run
    models_to_run = {}
    for model_id, model_info in MODELS.items():
        if model_filter and model_id != model_filter:
            continue
        models_to_run[model_id] = model_info

    if not models_to_run:
        print(f"ERROR: No matching models found. Available: {list(MODELS.keys())}")
        sys.exit(1)

    print("=" * 70)
    print(f"  BENCHMARK: {config_id} — {config['name']}")
    print(f"  Framework: {framework}")
    print(f"  Models:    {list(models_to_run.keys())}")
    print(f"  Prompts:   {len(PROMPTS)} prompts × {NUM_RUNS} runs each")
    print(f"  Output:    {run_dir}")
    print("=" * 70)

    if dry_run:
        print("\n[DRY RUN] Showing commands that would be executed:\n")

    # Iterate models
    for model_id, model_info in models_to_run.items():
        model_path = os.path.join(MODEL_DIR, model_info["filename"])

        if not dry_run and not os.path.isfile(model_path):
            print(f"\n  [SKIP] Model file not found: {model_path}")
            continue

        print(f"\n{'─' * 60}")
        print(f"  Model: {model_info['description']} (Tier {model_info['tier']})")
        print(f"  File:  {model_path}")
        print(f"{'─' * 60}")

        # Measure model load time
        if not dry_run:
            print(f"  Measuring model load time...")
            load_time = measure_model_load_time(binary, framework, model_path)
            print(f"  Model load time: {load_time:.1f} ms")

            # Save load time metadata
            with open(os.path.join(run_dir, f"{model_id}_load_time.json"), "w") as f:
                json.dump(
                    {"model_id": model_id, "load_time_ms": load_time}, f, indent=2
                )

        # Iterate prompts
        for prompt_data in PROMPTS:
            print(
                f"\n  [{prompt_data['id']}] {prompt_data['name']} "
                f"(in={prompt_data['input_category']}, out={prompt_data['output_category']})"
            )

            if dry_run:
                if framework == "llama.cpp":
                    cmd = build_llama_cpp_cmd(
                        binary,
                        model_path,
                        prompt_data["text"][:60] + "...",
                        prompt_data["n_predict"],
                    )
                else:
                    cmd = build_dllama_cmd(
                        binary,
                        model_path,
                        prompt_data["text"][:60] + "...",
                        prompt_data["n_predict"],
                    )
                print(f"    CMD: {' '.join(cmd[:8])} ...")
                continue

            # Run NUM_RUNS times
            for run_num in range(1, NUM_RUNS + 1):
                is_warmup = run_num <= WARMUP_RUNS
                label = (
                    "WARM-UP"
                    if is_warmup
                    else f"RUN {run_num - WARMUP_RUNS}/{NUM_RUNS - WARMUP_RUNS}"
                )

                print(f"    {label}...", end=" ", flush=True)

                result, generated_text = run_single_prompt(
                    config_id, binary, framework, model_path, prompt_data
                )

                # Add model and run metadata
                result["model_id"] = model_id
                result["model_description"] = model_info["description"]
                result["model_tier"] = model_info["tier"]
                result["run_number"] = run_num
                result["is_warmup"] = is_warmup

                # Print summary
                if result["returncode"] == 0:
                    print(
                        f"eval={result['eval_rate_tps']:.2f} tok/s | "
                        f"prompt={result['prompt_rate_tps']:.2f} tok/s | "
                        f"e2e={result['end_to_end_ms']:.0f} ms | "
                        f"CPU={result['avg_cpu_pct']:.0f}% | "
                        f"mem={result['peak_mem_mb']:.0f} MB"
                    )
                else:
                    print(f"FAILED (exit {result['returncode']})")

                # Save to CSV
                append_csv(csv_path, result)

                # Save generated text for consistency check (non-warmup only)
                if not is_warmup and generated_text:
                    text_dir = os.path.join(run_dir, "outputs", model_id)
                    os.makedirs(text_dir, exist_ok=True)
                    text_file = os.path.join(
                        text_dir, f"{prompt_data['id']}_run{run_num}.txt"
                    )
                    with open(text_file, "w") as f:
                        f.write(generated_text)

                # Brief pause between runs to let system settle
                if run_num < NUM_RUNS:
                    time.sleep(2)

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE. Results saved to: {csv_path}")
    print(f"  Generated outputs saved to: {os.path.join(run_dir, 'outputs')}")
    print(f"{'=' * 70}")


# ============================================================
# Entry point
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark C1/C2 configurations for thesis experiments"
    )
    parser.add_argument(
        "--config",
        required=True,
        choices=["C1", "C2"],
        help="Configuration to run (C1=llama.cpp, C2=Distributed Llama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Run only this model. Choices: {list(MODELS.keys())}",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing"
    )
    args = parser.parse_args()

    # Pre-flight checks
    print("\n[Pre-flight checks]")
    print(f"  Config:     {args.config}")
    print(f"  Model dir:  {MODEL_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Runs:       {NUM_RUNS} ({WARMUP_RUNS} warm-up)")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Threads:    {NUM_THREADS}")

    if not args.dry_run:
        print("\n  ⚠  REMINDER: Reboot the Pi before each configuration run")
        print("     to clear residual state (as per Section 3.7.4).")
        # resp = input("\n  Press Enter to start (or 'q' to quit): ")
        # if resp.strip().lower() == "q":
        # sys.exit(0)

    run_benchmark(args.config, model_filter=args.model, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
