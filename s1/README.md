# S1 Feasibility Harness

This is the fresh S1 runner for the feasibility map and harness validation
experiment described in `discussion.md`.

It is intentionally separate from the expired `benchmark.py` path. The runner
does reuse the llama.cpp timing parsing pattern, but memory comes from
`/usr/bin/time`, not from periodic `psutil` sampling.

The S1 workload prompt is shared across llama.cpp and dllama during the
feasibility and throughput screen.
For Qwen3 cells, the runner prefixes the prompt with `/no_think` when
`disable_qwen3_thinking: true` is set in the manifest, keeping the S1 probe
focused on fixed-length throughput rather than reasoning-token variance.

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

CPU utilization is measured from explicit process CPU-time deltas, not
`cpu_percent()`. In this command-style S1 runner, the reported CPU fields use
the final `/usr/bin/time` user/sys accounting for the completed runtime command:
`cpu_utilization = (delta_user + delta_system) / elapsed`, so 1.0 means one
fully occupied core and 4.0 means four fully occupied cores.
`cpu_utilization_percent` is this core-equivalent value multiplied by 100 and
may exceed 100%. `cpu_utilization_threads_percent` normalizes that value by
the manifest's configured runtime thread count. Process CPU counters include
all threads in the runtime.

## Link Pi Models To An External SSD

The S1 manifests expect model files under `models/...` from the benchmark repo
root. On the Pi, keep the large files on the SSD and make `models` a symlink.

Find the SSD mount point:

```bash
lsblk -f
```

If the SSD is not mounted yet, mount the correct partition, for example:

```bash
sudo mkdir -p /mnt/ssd
sudo mount /dev/sda1 /mnt/ssd
```

Create or reuse a model root on the SSD. The preferred layout is:

```text
/mnt/ssd/models/llama/...
/mnt/ssd/models/dllama/...
```

From the benchmark repo, link `models` to that SSD model root:

```bash
cd ~/thesis/benchmark

if [ -L models ]; then
  readlink -f models
elif [ -e models ]; then
  mv models "models.local.$(date +%Y%m%d_%H%M%S)"
fi

ln -s /mnt/ssd/models models
ls -la models
```

If the dllama files are in a separate SSD folder such as
`/mnt/ssd/dllama-models`, link only the nested dllama path:

```bash
mkdir -p /mnt/ssd/models

if [ -L /mnt/ssd/models/dllama ]; then
  readlink -f /mnt/ssd/models/dllama
elif [ -e /mnt/ssd/models/dllama ]; then
  mv /mnt/ssd/models/dllama "/mnt/ssd/models/dllama.local.$(date +%Y%m%d_%H%M%S)"
fi

ln -s /mnt/ssd/dllama-models /mnt/ssd/models/dllama
ls -la /mnt/ssd/models/dllama
```

Verify path resolution before running data collection:

```bash
venv/bin/python -m s1.run_s1 \
  --manifest s1/manifest.yaml \
  --dry-run \
  --only qwen3-4b \
  --only Q4_K_M

venv/bin/python -m s1.run_s1 \
  --manifest s1/manifest_dllama.yaml \
  --dry-run \
  --only qwen3-1.7b \
  --only Q4_0
```

## Run Distributed Llama Q4_0/Q40 Cells

Use the dllama manifest:

```bash
venv/bin/python -m s1.run_s1 --manifest s1/manifest_dllama.yaml --skip-missing
```

Fast smoke test for one dllama cell:

```bash
venv/bin/python -m s1.run_s1 \
  --manifest s1/manifest_dllama.yaml \
  --only qwen3-1.7b \
  --only Q4_0 \
  --warmup-reps 0 \
  --measured-reps 1 \
  --generated-tokens 16
```

Download the Distributed Llama Q40 model/tokenizer files reproducibly:

```bash
venv/bin/python -m s1.download_dllama_models --print-commands
```

On Mac with the SanDisk SSD mounted:

```bash
venv/bin/python -m s1.download_dllama_models \
  --models-dir "/Volumes/Anh Linh/dllama-models"
```

Download only one dllama cell first:

```bash
venv/bin/python -m s1.download_dllama_models \
  --models-dir "/Volumes/Anh Linh/dllama-models" \
  --only qwen3-1.7b
```

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
- `s1_summary.csv`: measured repetitions summarized by model and quant,
  including mean CPU utilization
- `s1_feasibility_heatmap.svg`: compact verdict heatmap
- `raw/...`: stdout, stderr, `/usr/bin/time` output, and command JSON

Pretty-print a summary in the terminal:

```bash
venv/bin/python -m s1.print_summary logs/s1/<timestamp>/s1_summary.csv
```

Verdicts are:

- `FITS`: command succeeded without detected swap
- `FITS_TIGHT`: succeeded but peak RSS reached the configured memory threshold
- `SWAP`: succeeded but `/proc/vmstat` swap counters changed
- `OOM`: killed or failed with an allocation/OOM signature
- `TIMEOUT`: exceeded the configured timeout
- `FAIL`: non-OOM runtime failure
- `MISSING_MODEL`: model file not present in the manifest path
