#!/usr/bin/env python3
"""
build_workloads.py
==================
Generate the token-controlled workload prompts (W1-W4) for the distributed
edge-inference thesis. Implements the workload dimension defined in Methodology
section 3.6: content-neutral prompts whose only varying factor is token geometry.

DESIGN (per the methodology):
  - Prompts are NOT hand-written tasks. Each prompt is a fixed, original
    instruction wrapping exactly N tokens of neutral filler text sliced from
    WikiText-2 (the same corpus used for the perplexity quality screen).
    This decouples request COST (token counts) from request CONTENT, which is
    the entire reason the token-controlled suite replaced the old 2x2 semantic
    matrix.
  - The PROMPT length is swept (128 / 512 / 2048 / 8192 tokens).
  - The GENERATION length is a fixed control (256 tokens for W2-W4, 64 for W1),
    forced to an exact length via --n-predict + --ignore-eos so decode time is
    comparable across configurations regardless of when a model would naturally
    stop. (Generation length is a control, not a swept variable; W1's shorter
    output anchors the short-interactive case and exposes per-request fixed cost.)
  - Token counts are authoritative under the Qwen3 tokenizer (the primary family).
    The SAME prompt text is reused for every model; the actual token count under
    each model's own tokenizer is measured and recorded rather than assumed equal.

WHAT THIS SCRIPT DOES NOT DO:
  - It does not embed WikiText-2 text (that corpus has its own license). It
    slices the text at runtime from the dataset you provide, so nothing
    copyrighted is frozen into the thesis or this file.

USAGE:
  1. Get WikiText-2 raw text. Either:
       pip install datasets
       python build_workloads.py --download         # fetches via HF datasets
     or point at a local file:
       python build_workloads.py --source /path/to/wikitext-2-raw/wiki.test.raw
  2. Prompts are written to ./workloads/  as W{1..4}.txt plus a manifest.
  3. The manifest prints the exact llama.cpp / llama-bench commands per cell.

REQUIRES:  transformers   (for the tokenizer)
OPTIONAL:  datasets       (only if you use --download)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict


# ----------------------------------------------------------------------------
# Workload definitions (Methodology Table, section 3.6)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Workload:
    name: str
    prompt_tokens: int
    gen_tokens: int
    description: str


WORKLOADS = [
    Workload("W1", 128, 64, "Command / classification / short sensor query"),
    Workload(
        "W2", 512, 256, "Typical assistant request (representative interactive case)"
    ),
    Workload("W3", 2048, 256, "Moderate-context request (e.g. retrieval-augmented QA)"),
    Workload(
        "W4", 8192, 256, "Long-context request (long document / accumulated session)"
    ),
]

# ----------------------------------------------------------------------------
# Fixed instruction wrappers. These are ORIGINAL text (yours), not from any
# corpus. They are deliberately generic so they add a small, constant token
# cost and do not introduce task-specific decode behavior. The instruction is
# identical across W1-W4 so that the ONLY thing differing between workloads is
# the amount of filler body and the generation cap -- never the task itself.
#
# A single instruction is used for all four workloads to keep content constant.
# It asks for a fixed-length continuation/summary; because generation is forced
# to an exact length with --ignore-eos, the semantic adequacy of the instruction
# is irrelevant to the measurement -- it exists only to occupy the prompt and
# trigger generation.
# ----------------------------------------------------------------------------
INSTRUCTION = (
    "Read the passage below and write a continuous summary of its contents. "
    "Do not use bullet points; write in plain prose."
)

# Default authoritative tokenizer. Qwen3 models share a tokenizer; any Qwen3
# checkpoint's tokenizer is fine. Override with --tokenizer for a specific repo.
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"


def load_source_text(args) -> str:
    """Return raw WikiText-2 text, either downloaded or read from a local file."""
    if args.download:
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit(
                "`datasets` not installed. Run: pip install datasets  "
                "(or use --source PATH with a local wikitext file)"
            )
        # WikiText-2 raw, test split -- same split used for the perplexity screen.
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(row for row in ds["text"] if row.strip())
        return text
    if args.source:
        if not os.path.isfile(args.source):
            sys.exit(f"--source file not found: {args.source}")
        with open(args.source, "r", encoding="utf-8") as fh:
            return fh.read()
    sys.exit("Provide either --download or --source PATH")


def slice_body_to_exact_total(
    tokenizer, instruction: str, source_text: str, target_total: int
) -> tuple[str, int]:
    """
    Build `instruction + "\\n\\n" + body` whose TOTAL token count under
    `tokenizer` equals exactly `target_total`.

    BPE tokenizers can merge across the join boundary, shifting the total by a
    token or two versus instruction_tokens + body_tokens. We therefore slice an
    initial body estimate, measure the assembled prompt, and adjust the body
    length until the assembled total is exact.
    """
    src_ids = tokenizer.encode(source_text, add_special_tokens=False)
    instr_ids = tokenizer.encode(instruction, add_special_tokens=False)
    sep = "\n\n"

    # initial body budget
    body_budget = target_total - len(instr_ids)
    if body_budget < 1:
        raise ValueError(
            f"Instruction is {len(instr_ids)} tokens, which is >= target "
            f"{target_total}. Use a shorter instruction or a larger target."
        )
    if len(src_ids) < body_budget:
        raise ValueError(
            f"Source text has only {len(src_ids)} tokens; need at least "
            f"{body_budget} for a {target_total}-token prompt. Use a longer source."
        )

    # Adjust until the ASSEMBLED prompt hits target_total exactly.
    for _ in range(16):  # converges in 1-2 iterations in practice
        body = tokenizer.decode(src_ids[:body_budget])
        full = instruction + sep + body
        total = len(tokenizer.encode(full, add_special_tokens=False))
        if total == target_total:
            return full, total
        # nudge body length by the observed error
        body_budget -= total - target_total
        if body_budget < 1:
            raise ValueError("Could not converge; instruction too long for target.")
    # Best effort if exact convergence failed (should not happen for these sizes)
    body = tokenizer.decode(src_ids[:body_budget])
    full = instruction + sep + body
    return full, len(tokenizer.encode(full, add_special_tokens=False))


def main():
    ap = argparse.ArgumentParser(
        description="Build W1-W4 token-controlled workload prompts."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--download",
        action="store_true",
        help="Download WikiText-2 raw test split via HF datasets.",
    )
    src.add_argument(
        "--source", metavar="PATH", help="Path to a local WikiText-2 raw text file."
    )
    ap.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HF tokenizer repo (default: {DEFAULT_TOKENIZER}).",
    )
    ap.add_argument("--outdir", default="workloads", help="Output directory.")
    ap.add_argument(
        "--verify-models",
        nargs="*",
        default=[],
        help="Optional extra tokenizer repos (e.g. cross-family models) "
        "to report the actual token count of each prompt under "
        "their own tokenizer.",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading authoritative tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Loading source text (WikiText-2 raw, test split) ...")
    source_text = load_source_text(args)
    n_src = len(tok.encode(source_text, add_special_tokens=False))
    print(f"Source length: {n_src} tokens under {args.tokenizer}")
    if n_src < max(w.prompt_tokens for w in WORKLOADS):
        sys.exit(
            f"Source too short ({n_src} tok) for W4 (8192). Concatenate more text."
        )

    os.makedirs(args.outdir, exist_ok=True)

    # Optional cross-tokenizer verification
    verify_toks = {}
    for repo in args.verify_models:
        try:
            verify_toks[repo] = AutoTokenizer.from_pretrained(repo)
            print(f"  loaded verification tokenizer: {repo}")
        except Exception as e:
            print(f"  WARNING: could not load {repo}: {e}")

    manifest = {
        "authoritative_tokenizer": args.tokenizer,
        "instruction": INSTRUCTION,
        "source": "WikiText-2 raw, test split",
        "generation_policy": "forced exact length via --n-predict N --ignore-eos",
        "workloads": [],
    }

    print("\nBuilding prompts:")
    print(f"  {'W':<3} {'target':>7} {'actual(Qwen3)':>14} {'gen':>5}  file")
    for w in WORKLOADS:
        prompt, total = slice_body_to_exact_total(
            tok, INSTRUCTION, source_text, w.prompt_tokens
        )
        path = os.path.join(args.outdir, f"{w.name}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        entry = {
            **asdict(w),
            "actual_prompt_tokens_authoritative": total,
            "prompt_file": path,
            "cross_tokenizer_counts": {},
        }
        # record the SAME text's length under each other tokenizer (honest reporting
        # of the single-corpus tradeoff: counts are exact for Qwen3, approximate elsewhere)
        for repo, vtok in verify_toks.items():
            entry["cross_tokenizer_counts"][repo] = len(
                vtok.encode(prompt, add_special_tokens=False)
            )

        manifest["workloads"].append(entry)
        extra = ""
        if entry["cross_tokenizer_counts"]:
            extra = "  | other: " + ", ".join(
                f"{r.split('/')[-1]}={c}"
                for r, c in entry["cross_tokenizer_counts"].items()
            )
        print(
            f"  {w.name:<3} {w.prompt_tokens:>7} {total:>14} {w.gen_tokens:>5}  {path}{extra}"
        )

    man_path = os.path.join(args.outdir, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest written: {man_path}")

    # ------------------------------------------------------------------
    # Emit the exact run commands. Two forms:
    #   (a) llama-cli  -- for the per-prompt harness (P-e2e style)
    #   (b) llama-bench -- for clean prefill/decode rate numbers
    # Generation is forced to exact length with --ignore-eos so every run
    # decodes precisely gen_tokens tokens.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RUN COMMANDS (fill in MODEL, and for distributed runs the --rpc flags)")
    print("=" * 72)
    for w in WORKLOADS:
        npred = int(
            round(w.gen_tokens * 1.0)
        )  # exact; buffer not needed with --ignore-eos
        print(
            f"\n# {w.name}: prompt={w.prompt_tokens} tok, generate exactly {w.gen_tokens} tok"
        )
        print(f"#   {w.description}")
        # (a) llama-cli, single node (T0). For T1 add: --rpc HOST:PORT and -ngl as needed.
        print(f"llama-cli -m $MODEL \\")
        print(f"    -f {os.path.join(args.outdir, w.name + '.txt')} \\")
        print(f"    -n {w.gen_tokens} --ignore-eos \\")
        print(f"    -t 4 --temp 0 --top-k 1 --seed 0 \\")
        print(f"    --no-display-prompt")
        # (b) llama-bench equivalent for clean pp/tg rates
        print(f"#   llama-bench equivalent (clean prefill/decode rates):")
        print(
            f"#   llama-bench -m $MODEL -p {w.prompt_tokens} -n {w.gen_tokens} -t 4 -r 5"
        )

    print("\nNOTES:")
    print("  * --ignore-eos forces exactly -n tokens of generation (the fixed-length")
    print(
        "    control). Without it, models stop early and decode times are not comparable."
    )
    print(
        "  * --temp 0 --top-k 1 --seed 0 => greedy/deterministic decoding (protocol 3.8.5)."
    )
    print(
        "  * 5 reps + discard-first warm-up, fixed order, reboot between configs: handled"
    )
    print(
        "    by your run harness, not by this builder. llama-bench -r 5 covers reps for (b)."
    )
    print(
        "  * For T1 (llama.cpp RPC): start rpc-server on the worker, pass --rpc on the master."
    )
    print(
        "  * For T3 (Distributed Llama) and T2 (prima.cpp): use each runtime's own client;"
    )
    print(
        "    feed the SAME W*.txt prompt file so the workload is identical across topologies."
    )
    print(
        "  * W4 (8192) may OOM the KV cache on large models -- record the refusal as data"
    )
    print(
        "    (it marks the context ceiling for that config/topology), per section 3.6."
    )


if __name__ == "__main__":
    main()
