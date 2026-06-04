#!/usr/bin/env python3
"""
Output consistency check (Section 3.6.2).

Compares generated outputs from C2/C3/C4/C5 against C1 reference outputs.
Reports exact match, character-level diff stats, and minor/major classification.

Usage:
    python3 consistency_check.py \
        --reference /path/to/C1_results/outputs \
        --compare /path/to/C2_results/outputs \
        --output consistency_report.csv
"""

import argparse
import csv
import difflib
import sys
from pathlib import Path
from typing import cast
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import transformers
transformers.logging.set_verbosity_error()
from bert_score import score as bs_score


def _bert_score(candidates: list[str], references: list[str]) -> list[float]:
    """
    Compute BERTScore F1 for each candidate/reference pair.
    """
    _, _, F1 = bs_score(candidates, references, lang="en", verbose=False)
    return [round(float(f), 4) for f in F1]


def load_outputs(output_dir: str) -> dict:
    """
    Load all generated text files from a results/outputs directory.
    Returns: { "model_id/prompt_id": text, ... }
    Averages across runs by taking the first non-warmup run found.
    """
    outputs = {}
    base = Path(output_dir)
    if not base.exists():
        print(f"ERROR: Directory not found: {output_dir}")
        sys.exit(1)

    # Auto-detect outputs/ subdirectory if the user passed the run log directory
    if (base / "outputs").is_dir():
        base = base / "outputs"

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        for txt_file in sorted(model_dir.glob("*.txt")):
            # e.g., P01_run2.txt
            prompt_id = txt_file.stem.split("_run")[0]
            key = f"{model_id}/{prompt_id}"
            # Take the first (lowest run number) file found
            if key not in outputs:
                outputs[key] = txt_file.read_text(encoding="utf-8").strip()

    return outputs


def evaluate_consistency(reference_output: str, distributed_output: str) -> dict:
    """
    Evaluates the consistency between a baseline LLM output and a distributed LLM output
    based on the methodology defined in Section 3.5.2 of the thesis.

    Args:
        reference_output (str): The output from the single-device baseline (C1).
        distributed_output (str): The output from the distributed configuration (C3 or C4).

    Returns:
        dict: A dictionary containing the exact match boolean and the BLEU score.
    """

    # Clean whitespace for an accurate baseline comparison
    ref_clean = reference_output.strip()
    dist_clean = distributed_output.strip()

    # 1. Exact Token Match
    # Evaluates whether the distributed output is identical to the C1 reference.
    exact_match = ref_clean == dist_clean

    # 2. BLEU Score
    # Tokenize the outputs. Using basic whitespace split here, but you can swap
    # this with a specific tokenizer (like tiktoken or transformers) if you want
    # strict sub-word token level matching.
    ref_tokens = ref_clean.split()
    dist_tokens = dist_clean.split()

    # If the strings match exactly, BLEU is perfectly 1.0.
    # Otherwise, calculate the score.
    if exact_match:
        bleu_score = 1.0
    else:
        # We apply a smoothing function to prevent zero scores on short responses
        # that might lack higher-order n-gram overlaps.
        smoothing = SmoothingFunction().method1
        bleu_score = cast(
            float,
            sentence_bleu(
                [ref_tokens],  # sentence_bleu expects a list of reference token lists
                dist_tokens,
                smoothing_function=smoothing,
            ),
        )

    # 3. BERTScore (semantic similarity)
    if exact_match:
        bert_f1 = 1.0
    else:
        bert_f1 = _bert_score([dist_clean], [ref_clean])[0]

    return {
        "exact_match": exact_match,
        "bleu_score": round(bleu_score, 4),
        "bert_score": bert_f1,
    }


def run_consistency_check(reference_dir: str, compare_dir: str, output_csv: str):
    """Run the full consistency check and save results."""

    ref_outputs = load_outputs(reference_dir)
    cmp_outputs = load_outputs(compare_dir)

    print(f"Reference outputs: {len(ref_outputs)} entries")
    print(f"Compare outputs:   {len(cmp_outputs)} entries")

    results = []
    exact_matches = 0
    mismatches = 0
    missing = 0
    bleu_scores = []
    bert_scores = []

    for key in sorted(ref_outputs.keys()):
        model_id, prompt_id = key.split("/")

        if key not in cmp_outputs:
            results.append(
                {
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                    "exact_match": False,
                    "bleu_score": 0.0,
                    "bert_score": 0.0,
                    "status": "missing",
                }
            )
            missing += 1
            continue

        comparison = evaluate_consistency(ref_outputs[key], cmp_outputs[key])
        comparison["model_id"] = model_id
        comparison["prompt_id"] = prompt_id
        comparison["status"] = "match" if comparison["exact_match"] else "mismatch"
        results.append(comparison)

        if comparison["exact_match"]:
            exact_matches += 1
        else:
            mismatches += 1
        bleu_scores.append(comparison["bleu_score"])
        bert_scores.append(comparison["bert_score"])

    # Collect unique models and prompts (sorted)
    all_models = sorted({r["model_id"] for r in results})
    all_prompts = sorted({r["prompt_id"] for r in results})

    # Build lookup: (model_id, prompt_id) -> result row
    lookup = {(r["model_id"], r["prompt_id"]): r for r in results}

    # --- Flat CSV (detailed, one row per model+prompt) ---
    fields = [
        "model_id",
        "prompt_id",
        "exact_match",
        "bleu_score",
        "bert_score",
        "status",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    # --- Pivot CSV: rows=models, columns=prompts ---
    pivot_path = Path(output_csv).with_stem(Path(output_csv).stem + "_pivot")
    bleu_col = lambda p: f"{p}_bleu"
    bert_col = lambda p: f"{p}_bert"
    match_col = lambda p: f"{p}_exact"
    pivot_fields = (
        ["model_id"]
        + [col for p in all_prompts for col in (match_col(p), bleu_col(p), bert_col(p))]
        + ["avg_bleu", "avg_bert", "exact_pct"]
    )
    with open(pivot_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pivot_fields)
        writer.writeheader()
        for model in all_models:
            row: dict = {"model_id": model}
            model_bleu = []
            model_bert = []
            model_exact = 0
            model_total = 0
            for prompt in all_prompts:
                r = lookup.get((model, prompt))
                if r:
                    row[match_col(prompt)] = r["exact_match"]
                    row[bleu_col(prompt)] = r["bleu_score"]
                    row[bert_col(prompt)] = r.get("bert_score", "")
                    model_bleu.append(r["bleu_score"])
                    if "bert_score" in r:
                        model_bert.append(r["bert_score"])
                    model_exact += int(r["exact_match"])
                    model_total += 1
                else:
                    row[match_col(prompt)] = ""
                    row[bleu_col(prompt)] = ""
                    row[bert_col(prompt)] = ""
            row["avg_bleu"] = (
                round(sum(model_bleu) / len(model_bleu), 4) if model_bleu else ""
            )
            row["avg_bert"] = (
                round(sum(model_bert) / len(model_bert), 4) if model_bert else ""
            )
            row["exact_pct"] = (
                f"{model_exact/max(model_total,1)*100:.1f}%" if model_total else ""
            )
            writer.writerow(row)

    # Summary
    total = len(results)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_bert = sum(bert_scores) / len(bert_scores) if bert_scores else 0.0
    print(f"\n{'=' * 50}")
    print(f"  Consistency Check Summary")
    print(f"{'=' * 50}")
    print(f"  Total comparisons: {total}")
    print(
        f"  Exact matches:     {exact_matches} ({exact_matches/max(total,1)*100:.1f}%)"
    )
    print(f"  Mismatches:        {mismatches} ({mismatches/max(total,1)*100:.1f}%)")
    print(f"  Missing outputs:   {missing}")
    print(f"  Average BLEU:      {avg_bleu:.4f}")
    print(f"  Average BERTScore: {avg_bert:.4f}")

    # Per-model summary
    print(f"\n  Per-model breakdown:")
    print(f"  {'Model':<30} {'Exact%':>8} {'Avg BLEU':>10} {'Avg BERT':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
    for model in all_models:
        rows = [r for r in results if r["model_id"] == model]
        m_exact = sum(1 for r in rows if r["exact_match"])
        m_bleu = [r["bleu_score"] for r in rows if r["status"] != "missing"]
        m_bert = [
            r["bert_score"]
            for r in rows
            if r.get("bert_score") is not None and r["status"] != "missing"
        ]
        print(
            f"  {model:<30} {m_exact/max(len(rows),1)*100:>7.1f}%"
            f" {sum(m_bleu)/max(len(m_bleu),1):>10.4f}"
            f" {sum(m_bert)/max(len(m_bert),1):>10.4f}"
        )

    print(f"\n  Flat report:  {output_csv}")
    print(f"  Pivot report: {pivot_path}")


def compare_files(ref_file: str, cmp_file: str):
    """Directly compare two .txt output files and print results."""
    ref_path = Path(ref_file)
    cmp_path = Path(cmp_file)

    for p in (ref_path, cmp_path):
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    ref_text = ref_path.read_text(encoding="utf-8").strip()
    cmp_text = cmp_path.read_text(encoding="utf-8").strip()

    result = evaluate_consistency(ref_text, cmp_text)

    print(f"\n{'=' * 50}")
    print(f"  File Comparison")
    print(f"{'=' * 50}")
    print(f"  Reference : {ref_path}")
    print(f"  Compare   : {cmp_path}")
    print(f"{'=' * 50}")
    print(f"  Exact match : {result['exact_match']}")
    print(f"  BLEU score  : {result['bleu_score']:.4f}")
    print(f"  BERTScore   : {result['bert_score']:.4f}")

    if not result["exact_match"]:
        ref_lines = ref_text.splitlines()
        cmp_lines = cmp_text.splitlines()
        diff = list(
            difflib.unified_diff(
                ref_lines,
                cmp_lines,
                fromfile=ref_path.name,
                tofile=cmp_path.name,
                lineterm="",
            )
        )
        if diff:
            print(f"\n  Diff ({len(diff)} lines):")
            print("  " + "\n  ".join(diff[:60]))
            if len(diff) > 60:
                print(f"  ... ({len(diff) - 60} more lines)")


def main():
    parser = argparse.ArgumentParser(description="Output consistency check")
    subparsers = parser.add_subparsers(dest="command")

    # Default directory-based comparison
    dir_parser = subparsers.add_parser(
        "dirs", help="Compare two output directories (default mode)"
    )
    dir_parser.add_argument(
        "--reference", required=True, help="Path to C1 reference outputs directory"
    )
    dir_parser.add_argument(
        "--compare", required=True, help="Path to comparison outputs directory"
    )
    dir_parser.add_argument(
        "--output", default="consistency_report.csv", help="Output CSV path"
    )

    # Direct file comparison
    file_parser = subparsers.add_parser(
        "files", help="Compare two .txt output files directly"
    )
    file_parser.add_argument("reference", help="Path to reference .txt file")
    file_parser.add_argument("compare", help="Path to comparison .txt file")

    # Support legacy flat args (no subcommand) for backwards compatibility
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    parser.add_argument("--compare", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output", default="consistency_report.csv", help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    if args.command == "files":
        compare_files(args.reference, args.compare)
    elif args.command == "dirs":
        run_consistency_check(args.reference, args.compare, args.output)
    else:
        # Legacy: --reference / --compare flags without subcommand
        if args.reference and args.compare:
            run_consistency_check(args.reference, args.compare, args.output)
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
