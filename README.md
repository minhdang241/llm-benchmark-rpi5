# Pi setup
Raspberry Pi 5 (16 GB RAM) running Raspberry Pi OS (64-bit)

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y git build-essential cmake python3-pip

# Install zram for better swap
sudo apt install -y zram-tools
sudo reboot
```

After reboot

```bash
sudo systemctl disable bluetooth
sudo systemctl disable cups
sudo systemctl disable ModemManager
sudo systemctl disable avahi-daemon
```

## 1. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
# Binary: /home/pi/llama.cpp/build/bin/llama-cli
````

## 2. Build Distributed Llama

```bash
cd /home/pi
git clone https://github.com/b4rtaz/distributed-llama.git
cd distributed-llama
make dllama
# Binary: /home/pi/distributed-llama/dllama
```
Benchmarks single-device LLM inference on Raspberry Pi 5 using:

- **C1**: llama.cpp (baseline)
- **C2**: Distributed Llama (single-node mode)

## 3. Download Models

Move to the models folder from the root

**For llama.cpp**

```bash
mkdir -p models/llama
cd models/llama
```

```bash
hf download bartowski/Qwen_Qwen3-1.7B-GGUF/Qwen_Qwen3-1.7B-Q4_0.gguf  .
```

**For dllama.cpp**
**Important**: Update the `filename` values in `config.py` to match the exact
filenames you downloaded, as naming varies across repos.

## 4. Configure Paths

Edit `config.py` and update:

- `LLAMA_CPP_BIN` — path to your `llama-cli` binary
- `DLLAMA_BIN` — path to your `dllama` binary
- `MODEL_DIR` — path to your models directory
- `OUTPUT_DIR` — where results will be saved

## 5. Run Benchmarks

```bash
# Verify commands first (dry run)
python3 benchmark.py --config C1 --dry-run

# Run C1 (reboot Pi first!)
sudo reboot
# ... after reboot:
python3 benchmark.py --config C1

# Run C2 (reboot Pi first!)
sudo reboot
# ... after reboot:
python3 benchmark.py --config C2

# Run a single model only
python3 benchmark.py --config C1 --model qwen3-1.7b-q4_0
```

## 6. Analyse Results

```bash
# Summary for one config
python3 analyse.py /home/pi/benchmark_results/C1_*/C1_results.csv

# Compare C1 vs C2
python3 analyse.py \
    /home/pi/benchmark_results/C1_*/C1_results.csv \
    /home/pi/benchmark_results/C2_*/C2_results.csv
```

## 7. Consistency Check

After both C1 and C2 are complete:

```bash
python3 consistency_check.py \
    --reference /home/pi/benchmark_results/C1_*/outputs \
    --compare /home/pi/benchmark_results/C2_*/outputs \
    --output consistency_C1_vs_C2.csv
```

## Output Structure

```
benchmark_results/
├── C1_20260317_140000/
│   ├── C1_results.csv              # All metrics, all prompts, all runs
│   ├── qwen3-1.7b-q4_0_load_time.json
│   ├── llama-3.2-3b-instruct-q4_0_load_time.json
│   ├── ...
│   └── outputs/
│       ├── qwen3-1.7b-q4_0/
│       │   ├── P01_run2.txt        # Generated text (run 2)
│       │   ├── P01_run3.txt
│       │   └── ...
│       └── llama-3.2-3b-instruct-q4_0/
│           └── ...
└── C2_20260317_160000/
    └── ...
```

## Notes

- **Reboot between configurations** to clear residual state (per thesis Section 3.7.4)
- **Run 1 is warm-up** and excluded from analysis automatically
- Temperature is fixed at 0 for deterministic output
- If Distributed Llama's output format doesn't match the parser, check `parsers.py`
  and adjust the regex patterns for your version
- Energy measurement requires an external USB power meter — log readings manually
  and correlate with per-prompt wall times

## Adapting for C3/C4 (Distributed)

These scripts are designed to be extensible. To add C3 (llama.cpp RPC) and C4
(Distributed Llama 2-node):

1. Add new entries in `config.py` → `CONFIGURATIONS`
2. Add Tier C models in `config.py` → `MODELS`
3. Add command builders in `benchmark.py` for RPC worker setup
4. The CSV format, analysis, and consistency check scripts work unchanged
