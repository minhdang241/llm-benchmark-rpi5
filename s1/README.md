# S1 Feasibility Harness

This is the fresh S1 runner for the feasibility map and harness validation
experiment described in `discussion.md`.

It is intentionally separate from the expired `benchmark.py` path. The runner
does reuse the llama.cpp timing parsing pattern, but memory comes from
`/usr/bin/time`, not from periodic `psutil` sampling.

## Dry Run

```bash
python3 -m s1.run_s1 --dry-run --skip-missing
```

Filter to one cell:

```bash
python3 -m s1.run_s1 --dry-run --only qwen3-1.7b --only Q4_0
```

## Run On Mac For Development

```bash
python3 -m s1.run_s1 --skip-missing --limit 1
```

macOS uses:

```bash
/usr/bin/time -l
```

The macOS RSS value is parsed as bytes and converted to KiB. Use Mac rows only
for harness/parser development, not as Pi S1 thesis data.

## Run On Raspberry Pi For S1 Data

```bash
python3 -m s1.run_s1 --skip-missing
```

Linux uses:

```bash
/usr/bin/time -v -o time.txt
```

The Linux RSS value is parsed from `Maximum resident set size (kbytes)`.
The runner also reads `/proc/vmstat` before and after each run to detect swap
activity.

## Download S1 Models On The Pi

Install the requirements and log in to Hugging Face if you need gated models:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/hf auth login
```

Check the model download map without downloading files:

```bash
venv/bin/python -m s1.download_models --dry-run
```

Download only one cell first:

```bash
venv/bin/python -m s1.download_models --only qwen3-1.7b --only Q4_0
```

Download the full S1 map:

```bash
venv/bin/python -m s1.download_models
```

The download manifest is `s1/download_manifest.yaml`. Hugging Face repo/file
names vary by uploader, so if `--dry-run` reports a missing file, update that
entry in the download manifest before starting the full download.

## Outputs

Each run directory contains:

- `s1_results.csv`: one row per warm-up/measured repetition
- `s1_cells.jsonl`: the same rows as JSONL
- `s1_summary.csv`: measured repetitions summarized by model and quant
- `s1_feasibility_heatmap.svg`: compact verdict heatmap
- `raw/...`: stdout, stderr, `/usr/bin/time` output, and command JSON

Verdicts are:

- `FITS`: command succeeded without detected swap
- `FITS_TIGHT`: succeeded but peak RSS reached the configured memory threshold
- `SWAP`: succeeded but `/proc/vmstat` swap counters changed
- `OOM`: killed or failed with an allocation/OOM signature
- `TIMEOUT`: exceeded the configured timeout
- `FAIL`: non-OOM runtime failure
- `MISSING_MODEL`: model file not present in the manifest path
