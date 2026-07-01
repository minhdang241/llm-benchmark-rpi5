# Experiment Data-Collection Workbook

**Distributed LLM Inference on Resource-Constrained Edge Clusters**

**Experiments:** S1 · S3 · E1 · E2 · E3′  
_Real Term-1 data pre-filled · placeholders to collect_

## Cell Legend

| Formatting       | Meaning                                                                                         |
| :--------------- | :---------------------------------------------------------------------------------------------- |
| **Green fill**   | Real measured data carried from the Term 1 report — already collected.                          |
| **Orange fill**  | Load-bearing gap: a contribution depends on this cell. Collect these first.                     |
| **`[ ]` (grey)** | Placeholder — a value you must measure and fill in.                                             |
| **`[OOM?]`**     | Expected to OOM, but must be confirmed by an actual attempt (record the outcome).               |
| **—**            | Genuinely not applicable (e.g. no value because the run failed / single-node has no peer node). |
| **n/a**          | Not separately reported in Term 1; backfill from raw logs or mark permanently n/a.              |

---

## S1 — Feasibility Map & Harness Validation

Answers **RQ1** (loadability half) · Feeds **C1** (validated harness), **C3** (infeasible-on-one-node region).

- Single 16 GB Pi 5. Run order: S1 first — nothing downstream is trusted until the harness gate passes.

### Measurement steps

All rates are measured end-to-end through the completion endpoint and one common P-e2e client — the same method for both runtimes and the same method used in E2 and E3′, so every rate in the study is directly comparable. S1 probes each config once at a single fixed shape — **pp512 / tg128, 5 reps** (warm-up discarded) — plus one P-mem run; the full W1–W4 workload suite is an E2/E3′ concern, not S1. S1 disables Qwen thinking mode by prefixing the probe prompt with `/no_think`; this keeps the systems probe focused on fixed-length throughput rather than variable reasoning-token budgets. Feasibility (Verdict + Peak RSS) comes from P-mem (OS-level), independent of how inference is driven.

### How to measure — llama.cpp

All rates collected end-to-end via the completion endpoint + the common P-e2e client (`llama-completion`, same method as E2 and E3′, so numbers are comparable across all experiments). S1 uses one probe shape — **pp512 / tg128, 5 reps** — plus a P-mem run. Per config:

| Metric (probe)                    | How to collect it                                                                                                                                                                                                                                                                                                                                                                     |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Verdict + Peak RSS (P-mem)**    | Run the server process under `/usr/bin/time -v` and read 'Maximum resident set size'. If the process is killed before load completes → Verdict = OOM, RSS = — . If it loads but pages to disk → Verdict = SWAP. Comfortable load → FITS; near the 16 GB ceiling → FITS (tight).                                                                                                       |
| **Swap (P-mem)**                  | Read `/proc/vmstat` pgswapin / pgswapout before and after the run; the delta is swap activity. Cumulative counters, so a short burst can't be missed.                                                                                                                                                                                                                                 |
| **Decode + prefill rate (P-e2e)** | Start llama-server, hit `/completion` at the S1 probe shape **pp512 / tg128 (5 reps, warm-up discarded)** from a common Python streaming client. TTFT = request → first streamed token (gives prefill timing); inter-token gaps → decode tok/s; total = end-to-end latency. CRITICAL: force exact generation with `n_predict = 128` and `ignore_eos = true`, or the model stops early and the decode denominator is uncontrolled. |
| **Load time (P-load)**            | Wall-clock from process start to the model-ready log line (or first-request latency minus inference). Once per config, not per request.                                                                                                                                                                                                                                               |
| **Temp + throttle (P-thermal)**   | `vcgencmd measure_temp` and `vcgencmd get_throttled` before/after; 1 s sampling across the pp512/tg128 probe reps. Throttle flags are sticky bits.                                                                                                                                                                                                                                    |

### How to measure — dllama (Q4_0 cells only)

IDENTICAL method to llama.cpp — completion endpoint + the same P-e2e client. Because both runtimes are now measured the same way against the same stopwatch, their rates are directly comparable (no engine-internal-vs-stdout asterisk). dllama having no bench command is irrelevant: P-e2e measures from outside the engine.

| Metric (probe)                    | How to collect it                                                                                                                                                                                                                                                                  |
| :-------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Verdict + Peak RSS (P-mem)**    | Same as llama.cpp: wrap the dllama process in `/usr/bin/time -v`, read max RSS, observe load/OOM/swap. OS-level and runtime-agnostic — this is the column that carries the cross-runtime finding (the 8B that OOMs under llama.cpp but loads under dllama).                        |
| **Swap (P-mem)**                  | Same `/proc/vmstat` delta method as llama.cpp.                                                                                                                                                                                                                                     |
| **Decode + prefill rate (P-e2e)** | Start dllama-api, hit it with the SAME P-e2e client and the SAME probe shape used for llama.cpp (**pp512 / tg128, 5 reps**). TTFT, inter-token gaps, end-to-end — all defined identically across both runtimes. Force exact generation length the same way (dllama's equivalent of n_predict = 128 + ignore-eos). |
| **Load time (P-load)**            | Wall-clock from process start to dllama's model-ready output line.                                                                                                                                                                                                                 |
| **Temp + throttle (P-thermal)**   | Same `vcgencmd` method as llama.cpp.                                                                                                                                                                                                                                               |

_Note: prima.cpp is not run in S1: its memory management behaves like llama.cpp (buffer pre-allocation), and as a distributed runtime it first appears in E3′ as a two-node arm. There is no meaningful single-node prima.cpp feasibility row to collect here. When it does appear in E3′, it is measured through the same P-e2e client as the other two runtimes._

### Table S1-A — Single-node feasibility grid (scale ladder × quant ladder)

One run per cell on one 16 GB Pi 5. k-quant rows are llama.cpp-only (dllama cannot represent these formats). Q4_0 rows have BOTH runtimes — the cross-runtime cells where the OOM-vs-FITS finding lives. Verdict + Peak RSS from P-mem (OS-level). All rates from P-e2e (completion endpoint + common client) — same method for both runtimes, so they're directly comparable, and same method as E2/E3′. Qwen thinking mode is disabled with `/no_think`. Force exact generation length (n_predict + ignore_eos). ‡ = Term 1 data.

| Model      | Quant      | Runtime   | On-disk GB | Peak RSS GB | Verdict | Decode tok/s | Prefill tok/s | Load s | Swap MB |
| :--------- | :--------- | :-------- | :--------- | :---------- | :------ | :----------- | :------------ | :----- | :------ |
| Qwen3-0.6B | Q8_0       | llama.cpp | 0.75       | 1.48        | FITS    | 11.98        | 159.15        | [ ]    | 0.00    |
| Qwen3-0.6B | Q6_K       | llama.cpp | 0.58       | 1.21        | FITS    | 14.04        | 116.53        | [ ]    | 0.00    |
| Qwen3-0.6B | Q5_K_M     | llama.cpp | 0.51       | 1.10        | FITS    | 14.41        | 113.38        | [ ]    | 0.00    |
| Qwen3-0.6B | Q4_K_M     | llama.cpp | 0.45       | 0.99        | FITS    | 16.11        | 126.44        | [ ]    | 0.00    |
| Qwen3-0.6B | Q4_0       | llama.cpp | 0.44       | 0.96        | FITS    | 16.45        | 172.21        | [ ]    | 0.00    |
| Qwen3-0.6B | Q4_0       | dllama    | [ ]        | [ ]         | [ ]     | [ ]          | [ ]           | [ ]    | [ ]     |
| Qwen3-0.6B | Q3_K_M     | llama.cpp | 0.39       | 0.78        | FITS    | 16.96        | 73.57         | [ ]    | 0.00    |
| Qwen3-0.6B | Q2_K       | llama.cpp | 0.32       | 0.62        | FITS    | 18.02        | 62.53         | [ ]    | 0.00    |
| Qwen3-1.7B | Q8_0       | llama.cpp | 2.1        | 3.78        | FITS    | 5.33         | 41.80         | [ ]    | 0.00    |
| Qwen3-1.7B | Q6_K       | llama.cpp | 1.6        | 2.95        | FITS    | 6.53         | 46.22         | [ ]    | 0.00    |
| Qwen3-1.7B | Q5_K_M     | llama.cpp | 1.4        | 2.61        | FITS    | 7.21         | 45.27         | [ ]    | 0.00    |
| Qwen3-1.7B | Q4_K_M     | llama.cpp | 1.2        | 2.30        | FITS    | 8.03         | 52.44         | [ ]    | 0.00    |
| Qwen3-1.7B | Q4_0       | llama.cpp | 1.2        | 2.18        | FITS    | 8.20         | 78.49         | [ ]    | 0.00    |
| Qwen3-1.7B | Q4_0‡      | dllama    | 1.1        | 2.3         | FITS    | 10.14        | 12.04         | [ ]    | [ ]     |
| Qwen3-1.7B | Q3_K_M     | llama.cpp | 1.0        | 1.60        | FITS    | 9.02         | 26.83         | [ ]    | 0.00    |
| Qwen3-1.7B | Q2_K       | llama.cpp | 0.82       | 1.14        | FITS    | 10.10        | 22.64         | [ ]    | 0.00    |
| Qwen3-4B   | Q8_0       | llama.cpp | 4.0        | 8.36        | SWAP    | 2.24         | 12.46         | [ ]    | 0.27    |
| Qwen3-4B   | Q6_K       | llama.cpp | 3.1        | 6.55        | FITS    | 2.71         | 16.62         | [ ]    | 0.00    |
| Qwen3-4B   | Q5_K_M     | llama.cpp | 2.7        | 5.78        | SWAP    | 3.00         | 17.28         | [ ]    | 0.00    |
| Qwen3-4B   | Q4_K_M     | llama.cpp | 2.4        | 5.05        | SWAP    | 3.35         | 19.99         | [ ]    | 0.00    |
| Qwen3-4B   | Q4_0       | llama.cpp | 2.3        | 4.77        | SWAP    | 3.48         | 30.94         | [ ]    | 3.07    |
| Qwen3-4B   | Q4_0       | dllama    | [ ]        | [ ]         | [ ]     | [ ]          | [ ]           | [ ]    | [ ]     |
| Qwen3-4B   | Q3_K_M     | llama.cpp | 2.0        | 3.37        | SWAP    | 3.84         | 10.45         | [ ]    | 0.41    |
| Qwen3-4B   | Q2_K       | llama.cpp | 1.6        | 2.32        | SWAP    | 4.58         | 8.81          | [ ]    | 0.00    |
| Qwen3-8B   | Q8_0       | llama.cpp | 8.2        | 15.06       | SWAP    | 1.28         | 6.06          | [ ]    | 2.93    |
| Qwen3-8B   | Q6_K       | llama.cpp | 6.3        | 12.06       | SWAP    | 1.58         | 7.90          | [ ]    | 0.15    |
| Qwen3-8B   | Q5_K_M     | llama.cpp | 5.5        | 10.51       | SWAP    | 1.77         | 8.64          | [ ]    | 0.54    |
| Qwen3-8B   | Q4_K_M     | llama.cpp | 4.7        | 9.09        | SWAP    | 2.02         | 10.26         | [ ]    | 0.26    |
| Qwen3-8B   | Q4_0       | llama.cpp | 4.5        | 8.52        | SWAP    | 2.11         | 17.86         | [ ]    | 0.02    |
| Qwen3-8B   | Q4_0‡      | dllama    | 4.7        | 12.4        | FITS    | 2.12         | 2.45          | [ ]    | [ ]     |
| Qwen3-8B   | Q3_K_M     | llama.cpp | 3.9        | 5.73        | SWAP    | 2.44         | 5.55          | [ ]    | 0.01    |
| Qwen3-8B   | Q2_K       | llama.cpp | 3.1        | 3.67        | SWAP    | 2.81         | 4.85          | [ ]    | 0.02    |
| Qwen3-14B  | Q4_K_M     | llama.cpp | 8.4        | 15.22       | SWAP    | 1.14         | 5.06          | [ ]    | 62.58   |
| Qwen3-14B  | Q3_K_M     | llama.cpp | 6.9        | 10.02       | SWAP    | 1.39         | 2.89          | [ ]    | 0.00    |
| Qwen3-14B  | Q4_0 (M-b) | llama.cpp | 8.0        | 14.69       | SWAP    | 1.23         | 9.32          | [ ]    | 31.12   |
| Qwen3-14B  | Q4_0 (M-b) | dllama    | 10.9       | [ ]         | [ ]     | [ ]          | [ ]           | [ ]    | [ ]     |

### Table S1-B — Capacity tenants (infeasible-on-one-node cases)

Each is ONE attempt. If it OOMs, the OOM is the result. Record the failure mode precisely (OOM-at-alloc / swap-thrash / OOM-killed-mid-load).

| Model               | Quant  | On-disk GB | Peak RSS at failure GB | Verdict | Failure mode | Feeds                    |
| :------------------ | :----- | :--------- | :--------------------- | :------ | :----------- | :----------------------- |
| Qwen3-32B (dense)   | Q4_0   | ~18–20     | [ ]                    | [OOM?]  | [ ]          | RQ3 / C3                 |
| Qwen3-32B (dense)   | Q4_K_M | ~20        | [ ]                    | [OOM?]  | [ ]          | C3                       |
| Qwen3-30B-A3B (MoE) | Q4_0   | ~18        | [ ]                    | [OOM?]  | [ ]          | RQ5 / C4                 |
| Qwen3-30B-A3B (MoE) | Q4_K_M | ~18        | [ ]                    | [ ]     | [ ]          | confirm MoE loads at all |

### Table S1-C — Cross-family probes and structural outlier

Cross-family generalisation of the sweet-spot finding. BitNet appears in S1 only.

| Model                 | Quant   | On-disk GB | Peak RSS GB | Verdict | Decode tok/s | Prefill tok/s | Notes                |
| :-------------------- | :------ | :--------- | :---------- | :------ | :----------- | :------------ | :------------------- |
| Gemma-3-4B-it         | Q4_K_M  | 2.4        | 4.98        | SWAP    | 3.98         | 23.35         | sweet-spot probe     |
| Llama-3.2-3B-Instruct | Q4_K_M  | 1.9        | 4.09        | SWAP    | 4.84         | 27.02         | dllama-native family |

### Table S1-D — Carried-forward Q4_0 cross-runtime reference (Term 1)

The cross-runtime feasibility finding, already collected in Term 1. Verdict + RSS are P-mem (OS-level, runtime-agnostic) — fully valid. CAVEAT on the rate columns: these Term 1 rates predate the P-e2e method — llama.cpp's came from its internal reporting, dllama's from its stdout, which define prefill differently (hence the 127 vs 16.56 prefill gap, a chunk-size artifact). New S1 runs use P-e2e for both runtimes, making rates comparable. If you want these legacy rows comparable too, re-measure them under P-e2e; otherwise treat their rate columns as Term-1-method and the verdict/RSS as the carried-forward finding.

| Model        | Quant | Runtime   | Peak RSS GB | Verdict      | Decode tok/s | Prefill tok/s | Source  |
| :----------- | :---- | :-------- | :---------- | :----------- | :----------- | :------------ | :------ |
| Llama-3.2-1B | Q4_0  | llama.cpp | 5.6         | FITS         | 13.87        | 127.36        | T1 T4.1 |
| Llama-3.2-1B | Q4_0  | dllama    | 3.1         | FITS         | 14.31        | 16.56         | T1 T4.1 |
| Qwen3-1.7B   | Q4_0  | llama.cpp | 6.5         | FITS         | 9.98         | 83.92         | T1 T4.2 |
| Qwen3-1.7B   | Q4_0  | dllama    | 2.3         | FITS         | 10.14        | 12.04         | T1 T4.2 |
| Llama-3.1-8B | Q4_0  | llama.cpp | —           | OOM          | —            | —             | T1 T4.3 |
| Llama-3.1-8B | Q4_0  | dllama    | 13.8        | FITS         | 2.20         | 2.51          | T1 T4.3 |
| Qwen3-8B     | Q4_0  | llama.cpp | 14.1        | FITS (tight) | 2.57         | 17.41         | T1 T4.4 |
| Qwen3-8B     | Q4_0  | dllama    | 12.4        | FITS         | 2.12         | 2.45          | T1 T4.4 |

### Table S1-E — Harness-validation gate

Measured decode vs published envelope. A large deviation = harness bug, not a finding. Must PASS before E2/E3' rates are trusted.

| Config           | Measured decode tok/s | Published envelope | Source          | Within range?                                        |
| :--------------- | :-------------------- | :----------------- | :-------------- | :--------------------------------------------------- |
| ~1B Q4_0         | 13.9                  | 5–15 tok/s         | Nguyen (Pi 5)   | ✓                                                    |
| ~2B Q4_0         | 10.0                  | 5–15 tok/s         | Nguyen          | ✓                                                    |
| 8B Q4_0          | 2.1–2.6               | 1–3 tok/s          | Berglund (Pi 5) | ✓                                                    |
| 4B Q4_K_M        | 3.35                  | 8–11 tok/s         | community       | ✗ (swap-contaminated — re-run needed)                |
| **GATE VERDICT** |                       |                    |                 | **[BLOCKED — 4B run swapped; re-run on clean node]** |

---

## S3 — Cross-Host Quality Equivalence

Gate for **RQ2** · Feeds **C1**. Licenses running the quality track on the reference host (m1) instead of the Pi. Not in the S1→E1→E2→E3′ chain — a one-time methodological check.

### Table S3 — Cross-host quality-equivalence check (Pi vs reference host)

20 prompts (5 factual QA, 5 GSM8K, 5 instruction, 5 JSON), greedy, identical GGUF+commit. Acceptance: ≥90% exact match OR all divergences semantically neutral → quality track runs on m1. Pick a small but representative model (e.g. Qwen3-1.7B Q4_K_M).

| Prompt set      | pi-a (4 thr) exact% | m1 (4 thr) exact% | m1 (8 thr) exact% | First-divergence idx | Semantic-equiv on non-exact | Verdict                      |
| :-------------- | :------------------ | :---------------- | :---------------- | :------------------- | :-------------------------- | :--------------------------- |
| Factual QA (5)  | [ ]                 | [ ]               | [ ]               | [ ]                  | [ ]                         | [ ]                          |
| GSM8K (5)       | [ ]                 | [ ]               | [ ]               | [ ]                  | [ ]                         | [ ]                          |
| Instruction (5) | [ ]                 | [ ]               | [ ]               | [ ]                  | [ ]                         | [ ]                          |
| JSON output (5) | [ ]                 | [ ]               | [ ]               | [ ]                  | [ ]                         | [ ]                          |
| **OVERALL**     | [ ]                 | [ ]               | [ ]               | [ ]                  | [ ]                         | **[accept → m1 / fallback]** |

---

## E1 — Quality–Compression Frontier

Answers **RQ2** · Feeds **C4**. Two stages: cheap perplexity screen over the full grid, then expensive downstream tasks on the ≤6 survivors. Produces the frontier set E2 and E3′ both consume — lock it before running them.

### Table E1-Stage1 — Perplexity over the full grid (WikiText-2 raw test, ctx 2048)

llama-perplexity on the reference host. ~34–38 runs. Record measured bits-per-weight (from on-disk size ÷ params), NOT just the format name. PPL drives frontier selection.

| Model         | Quant  | Measured bits/weight | On-disk GB | Perplexity (WikiText-2) | Pareto-optimal? | → Frontier? |
| :------------ | :----- | :------------------- | :--------- | :---------------------- | :-------------- | :---------- |
| Qwen3-0.6B    | Q8_0   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-0.6B    | Q6_K   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-0.6B    | Q5_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-0.6B    | Q4_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-0.6B    | Q3_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-0.6B    | Q2_K   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-1.7B    | Q8_0   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-1.7B    | Q6_K   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-1.7B    | Q5_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-1.7B    | Q4_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-1.7B    | Q3_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-1.7B    | Q2_K   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-4B      | Q8_0   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-4B      | Q6_K   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-4B      | Q5_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-4B      | Q4_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-4B      | Q3_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-4B      | Q2_K   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-8B      | Q8_0   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-8B      | Q6_K   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-8B      | Q5_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-8B      | Q4_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-8B      | Q3_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-8B      | Q2_K   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-14B     | Q4_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Qwen3-14B     | Q3_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Gemma-3-4B-it | Q8_0   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Gemma-3-4B-it | Q4_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Llama-3.2-3B  | Q8_0   | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |
| Llama-3.2-3B  | Q4_K_M | [ ]                  | [ ]        | [ ]                     | [ ]             | [ ]         |

### Table E1-EqMem — Equal-memory comparison (tests H2)

Group configs at matched RAM footprint; compare smaller-higher-bit vs larger-lower-bit. Fill the cells with the PPL of whichever config hits that footprint band. This is the core RQ2 evidence.

| Memory band | Smaller model / higher bit | PPL | Larger model / lower bit | PPL | Winner |
| :---------- | :------------------------- | :-- | :----------------------- | :-- | :----- |
| ~1.5–2 GB   | 1.7B-Q8_0 (cfg)            | [ ] | 4B-Q3 (cfg)              | [ ] | [ ]    |
| ~2.5 GB     | 4B-Q6_K                    | [ ] | 8B-Q2/Q3                 | [ ] | [ ]    |
| ~5 GB       | 8B-Q4_K_M                  | [ ] | 14B-Q2/Q3                | [ ] | [ ]    |
| [add band]  | [ ]                        | [ ] | [ ]                      | [ ] | [ ]    |

### Table E1-Sel — Frontier configuration selection (the ≤6 set E2 & E3' consume)

Fix this BEFORE running E2/E3'. Selection rule: Pareto-optimal in (PPL, RAM) AND feasible per S1, biased to the coverage targets below.

| Slot (coverage target)                    | Selected config | PPL | RAM GB | Feasible per S1? | Locked? |
| :---------------------------------------- | :-------------- | :-- | :----- | :--------------- | :------ |
| Best sub-2 GB                             | [ ]             | [ ] | [ ]    | [ ]              | [ ]     |
| Best ~2.5 GB (expect 4B-Q4_K_M)           | [ ]             | [ ] | [ ]    | [ ]              | [ ]     |
| Best single-node ~5 GB (expect 8B-Q4_K_M) | [ ]             | [ ] | [ ]    | [ ]              | [ ]     |
| Quality-floor probe (Q3 tier)             | [ ]             | [ ] | [ ]    | [ ]              | [ ]     |
| Cross-family (Gemma or Llama)             | [ ]             | [ ] | [ ]    | [ ]              | [ ]     |
| [optional 6th]                            | [ ]             | [ ] | [ ]    | [ ]              | [ ]     |

### Table E1-Stage2 — Downstream tasks on frontier configs only (≤6 configs)

Run ONLY on the ≤6 frontier configs from Stage 1. All automated, /no_think, greedy. Scores as deltas vs the FP16/Q8 reference. Confirm Qwen3 thinking-mode policy before running.

| Frontier config   | MMLU acc% (500) | GSM8K acc% (200) | IFEval pass% (150) | JSON valid% (100) | JSON field-acc% | Δ vs Q8 ref  |
| :---------------- | :-------------- | :--------------- | :----------------- | :---------------- | :-------------- | :----------- |
| Config 1 = [fill] | [ ]             | [ ]              | [ ]                | [ ]               | [ ]             | [ ]          |
| Config 2 = [fill] | [ ]             | [ ]              | [ ]                | [ ]               | [ ]             | [ ]          |
| Config 3 = [fill] | [ ]             | [ ]              | [ ]                | [ ]               | [ ]             | [ ]          |
| Config 4 = [fill] | [ ]             | [ ]              | [ ]                | [ ]               | [ ]             | [ ]          |
| Config 5 = [fill] | [ ]             | [ ]              | [ ]                | [ ]               | [ ]             | [ ]          |
| Config 6 = [fill] | [ ]             | [ ]              | [ ]                | [ ]               | [ ]             | [ ]          |
| Q8 reference      | [ ]             | [ ]              | [ ]                | [ ]               | [ ]             | — (baseline) |

---

## E2 — Single-Node Systems Characterisation

Answers **RQ1** (usability half), **RQ2** (latency axis) · Feeds **C1**, **C4**. Produces the n=1 baselines E3′ compares against. All metrics via P-e2e (the same common client used in S1 and E3′); MBU derived from the P-e2e decode rate.

### Table E2 — Single-node systems characterisation (frontier configs × workloads)

Each of the ≤6 frontier configs × W1–W4. All metrics via P-e2e (completion + common client) — TTFT, decode, prefill timing, end-to-end. MBU derived from the P-e2e decode rate × per-token bytes ÷ achievable bandwidth (S1-F); decode is the bandwidth-bound phase, so decode-rate MBU is the meaningful one. 5 reps + warm-up discard. Force exact generation (n_predict + ignore_eos). W4 may OOM the KV cache → record refusal as data.

| Config   | Workload (P/G tok) | TTFT ms | Decode tok/s | Prefill tok/s | Peak RSS GB | CPU % | MBU % | Throttled? |
| :------- | :----------------- | :------ | :----------- | :------------ | :---------- | :---- | :---- | :--------- |
| Config 1 | W1 (128/64)        | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W2 (512/256)       | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W3 (2048/256)      | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W4 (8192/256†)     | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
| Config 2 | W1 (128/64)        | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W2 (512/256)       | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W3 (2048/256)      | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W4 (8192/256†)     | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
| Config 3 | W1 (128/64)        | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W2 (512/256)       | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W3 (2048/256)      | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W4 (8192/256†)     | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
| Config 4 | W1 (128/64)        | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W2 (512/256)       | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W3 (2048/256)      | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W4 (8192/256†)     | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
| Config 5 | W1 (128/64)        | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W2 (512/256)       | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W3 (2048/256)      | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W4 (8192/256†)     | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
| Config 6 | W1 (128/64)        | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W2 (512/256)       | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W3 (2048/256)      | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |
|          | W4 (8192/256†)     | [ ]     | [ ]          | [ ]           | [ ]         | [ ]   | [ ]   | [ ]        |

### Table E2-Usab — Usability map (RQ1 usability half, at W2)

The RQ1 usability verdict. Threshold: decode ≥5 tok/s AND TTFT ≤3 s, at W2. Run the sensitivity check (4–6 tok/s band) and confirm the classification is stable.

| Frontier config   | Decode tok/s @W2 | ≥5 tok/s? | TTFT s @W2 | ≤3 s? | Usable? | Stable across 4–6? |
| :---------------- | :--------------- | :-------- | :--------- | :---- | :------ | :----------------- |
| Config 1 = [fill] | [ ]              | [ ]       | [ ]        | [ ]   | [ ]     | [ ]                |
| Config 2 = [fill] | [ ]              | [ ]       | [ ]        | [ ]   | [ ]     | [ ]                |
| Config 3 = [fill] | [ ]              | [ ]       | [ ]        | [ ]   | [ ]     | [ ]                |
| Config 4 = [fill] | [ ]              | [ ]       | [ ]        | [ ]   | [ ]     | [ ]                |
| Config 5 = [fill] | [ ]              | [ ]       | [ ]        | [ ]   | [ ]     | [ ]                |
| Config 6 = [fill] | [ ]              | [ ]       | [ ]        | [ ]   | [ ]     | [ ]                |

---

## E3′ — Two-Node Decision Study

Answers **RQ3**, **RQ4**, **RQ5** · Feeds **C2**, **C3**, **C4**. The study the thesis is built toward. 4 model cells × 4 arms (T0/T1/T2/T3) × 2 prompt lengths. All at Q4_0 (only shared format). Runs last — depends on S1, E1, and E2.

### Table E3'-Cells — Model cells (span the single-node feasibility boundary)

All at Q4_0 (the only format all three two-node strategies share). M-a control, M-b pivot, M-c dense capacity, M-d MoE capacity.

| Cell | Config             | Single-node status (16 GB)          | What it tests                        | RQ  |
| :--- | :----------------- | :---------------------------------- | :----------------------------------- | :-- |
| M-a  | Qwen3-4B Q4_0      | Fits comfortably (~8–11 tok/s)      | Control: distribution should NOT pay | RQ3 |
| M-b  | Qwen3-14B Q4_0     | Fits tightly (slow) — confirm in S1 | Does n=2 lift slow→usable?           | RQ3 |
| M-c  | Qwen3-32B Q4_0     | OOM — confirm in S1-B               | Dense capacity: n=2 only option      | RQ3 |
| M-d  | Qwen3-30B-A3B Q4_0 | OOM — confirm in S1-B               | MoE: same bind, ~3B active           | RQ5 |

### Table E3'-Main — Per cell × arm × workload (the core two-node data)

4 cells × 4 arms (T0/T1/T2/T3) × 2 prompt lengths (512, 2048), G=256, batch 1, 5 reps + warm-up. All metrics via P-e2e (cross-engine comparable). P-net + P-mem on BOTH nodes. T3 unavailable for unsupported architectures (record as arch-unsupported, not blank). M-c/M-d T0 = attempted-once OOM.

| Cell | Arm (strategy)           | P len | TTFT ms | Decode tok/s | E2E ms | Bytes/token | RSS pi-a GB | RSS pi-b GB |
| :--- | :----------------------- | :---- | :------ | :----------- | :----- | :---------- | :---------- | :---------- |
| M-a  | T0 (llama.cpp 1-node)    | 512   | [ ]     | [ ]          | [ ]    | n/a         | [ ]         | n/a         |
| M-a  | T0 (llama.cpp 1-node)    | 2048  | [ ]     | [ ]          | [ ]    | n/a         | [ ]         | n/a         |
| M-a  | T1 (llama.cpp RPC split) | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-a  | T1 (llama.cpp RPC split) | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-a  | T2 (prima.cpp ring)      | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-a  | T2 (prima.cpp ring)      | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-a  | T3 (dllama tensor-par)   | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-a  | T3 (dllama tensor-par)   | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-b  | T0 (llama.cpp 1-node)    | 512   | [ ]     | [ ]          | [ ]    | n/a         | [ ]         | n/a         |
| M-b  | T0 (llama.cpp 1-node)    | 2048  | [ ]     | [ ]          | [ ]    | n/a         | [ ]         | n/a         |
| M-b  | T1 (llama.cpp RPC split) | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-b  | T1 (llama.cpp RPC split) | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-b  | T2 (prima.cpp ring)      | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-b  | T2 (prima.cpp ring)      | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-b  | T3 (dllama tensor-par)   | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-b  | T3 (dllama tensor-par)   | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-c  | T0 (llama.cpp 1-node)    | 512   | OOM     | OOM          | OOM    | n/a         | OOM         | n/a         |
| M-c  | T0 (llama.cpp 1-node)    | 2048  | OOM     | OOM          | OOM    | n/a         | OOM         | n/a         |
| M-c  | T1 (llama.cpp RPC split) | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-c  | T1 (llama.cpp RPC split) | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-c  | T2 (prima.cpp ring)      | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-c  | T2 (prima.cpp ring)      | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-c  | T3 (dllama tensor-par)   | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-c  | T3 (dllama tensor-par)   | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-d  | T0 (llama.cpp 1-node)    | 512   | OOM     | OOM          | OOM    | n/a         | OOM         | n/a         |
| M-d  | T0 (llama.cpp 1-node)    | 2048  | OOM     | OOM          | OOM    | n/a         | OOM         | n/a         |
| M-d  | T1 (llama.cpp RPC split) | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-d  | T1 (llama.cpp RPC split) | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-d  | T2 (prima.cpp ring)      | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-d  | T2 (prima.cpp ring)      | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-d  | T3 (dllama tensor-par)   | 512   | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |
| M-d  | T3 (dllama tensor-par)   | 2048  | [ ]     | [ ]          | [ ]    | [ ]         | [ ]         | [ ]         |

### Table E3'-Rank — Strategy ranking per cell (RQ4 → C2), at P=512

The head-to-head at n=2. Lower TTFT/E2E/bytes better; higher decode better. T3 cell blank where architecture unsupported.

| Cell | Best strategy | T1 decode | T2 decode | T3 decode | T1 TTFT | T2 TTFT | T3 TTFT |
| :--- | :------------ | :-------- | :-------- | :-------- | :------ | :------ | :------ |
| M-a  | [ ]           | [ ]       | [ ]       | [ ]       | [ ]     | [ ]     | [ ]     |
| M-b  | [ ]           | [ ]       | [ ]       | [ ]       | [ ]     | [ ]     | [ ]     |
| M-c  | [ ]           | [ ]       | [ ]       | [ ]       | [ ]     | [ ]     | [ ]     |
| M-d  | [ ]           | [ ]       | [ ]       | [ ]       | [ ]     | [ ]     | [ ]     |

### Table E3'-Speedup — S(2) = decode(n=2) ÷ decode(n=1), fitting models only (RQ3/H3a)

Only meaningful for M-a and M-b (models that have an n=1 baseline). >1 means distribution helped; <1 means it cost. M-c/M-d have no n=1 baseline (OOM).

| Cell      | n=1 decode (from E2) | T1 S(2) | T2 S(2) | T3 S(2) | Did any strategy pay? |
| :-------- | :------------------- | :------ | :------ | :------ | :-------------------- |
| M-a (4B)  | [ ]                  | [ ]     | [ ]     | [ ]     | [ ]                   |
| M-b (14B) | [ ]                  | [ ]     | [ ]     | [ ]     | [ ]                   |

### Table E3'-Comm — Communication model vs measurement (RQ4 → C2)

Analytic per-token bytes (computed from architecture) vs measured (P-net). Close agreement validates the model and licenses n>2 extrapolation. TP all-reduce ≈ f(hidden,layers,nodes); layer-split ≈ one hidden-state transfer per cut.

| Cell                  | Strategy         | Analytic bytes/token | Measured bytes/token (P-net) | Ratio (meas/analytic) | Link util % |
| :-------------------- | :--------------- | :------------------- | :--------------------------- | :-------------------- | :---------- |
| M-b                   | T1 (layer-split) | [ ]                  | [ ]                          | [ ]                   | [ ]         |
| M-b                   | T3 (tensor-par)  | [ ]                  | [ ]                          | [ ]                   | [ ]         |
| M-c                   | T1 (layer-split) | [ ]                  | [ ]                          | [ ]                   | [ ]         |
| M-c                   | T3 (tensor-par)  | [ ]                  | [ ]                          | [ ]                   | [ ]         |
| [add cells as needed] |                  | [ ]                  | [ ]                          | [ ]                   | [ ]         |

### Table E3'-MoE — MoE vs dense quality-per-second (RQ5 → C4), matched two-node footprint

The exploratory RQ5 result. M-d (MoE) vs M-c (dense) at matched two-node footprint. Quality from E1 downstream (if frontier) or a spot eval. The bet: MoE gives better decode×quality.

| Metric                           | M-c Qwen3-32B (dense) | M-d Qwen3-30B-A3B (MoE) | MoE better?   |
| :------------------------------- | :-------------------- | :---------------------- | :------------ |
| Decode tok/s (best 2-node arm)   | [ ]                   | [ ]                     | [ ]           |
| Active params / token            | ~32.8B                | ~3.3B                   | — (by design) |
| Total params                     | ~32.8B                | ~30.5B                  | —             |
| Downstream accuracy (mean)       | [ ]                   | [ ]                     | [ ]           |
| Quality-per-second (acc × tok/s) | [ ]                   | [ ]                     | [ ]           |

---

## Collection Priority

If time is limited, collect in this order — these resolve the gaps that block contributions:

1. **S1-B capacity tenants (32B + 30B-A3B OOM attempts)** — converts the C3/RQ3/RQ5 capacity case from extrapolation to measured data. ~4 overnight runs.
2. **Qwen3-14B Q4_0 single-node (S1-A flagged row)** — supplies the M-b T0 baseline that E3′ speedup needs and confirms the 'fits tightly' claim. 1 run.
3. **S1-F per-node bandwidth (STREAM-triad)** — unblocks every MBU number across S1/E2. 2 one-time runs.
4. **E1 Stage-1 perplexity grid** — selects the ≤6 frontier set that gates all of E2 and E3′. ~34–38 reference-host runs.
5. **Verify dllama supports the 30B-A3B MoE and Gemma at the pinned commit** — determines whether M-d gets a T3 arm or is arch-unsupported.
