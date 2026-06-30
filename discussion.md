## Probes and measurement harness

| Probe                                                                                                                                               | Tool                                                                                                                                                                                                                                                                          | Measures                                        | Used in        |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | -------------- |
| P-e2e                                                                                                                                               | `llama-completion` / `dllama` / `prima`                                                                                                                                                                                                                                       | TTFT, prefill rate, generation rate, load times | S1, S2, E2, E3 |
| P-net                                                                                                                                               | Read `/sys/class/net/<eth>/statistics/{rx,tx}_bytes` on both nodes before/after each run                                                                                                                                                                                      | Bytes on the wire per generated token           | E3             |
| P-mem                                                                                                                                               | Run the server process under /usr/bin/time -v and read 'Maximum resident set size'. If the process is killed before load completes → Verdict = OOM, RSS = — . If it loads but pages to disk → Verdict = SWAP. Comfortable load → FITS; near the 16 GB ceiling → FITS (tight). |
| Read /proc/vmstat pgswapin / pgswapout before and after the run; the delta is swap activity. Cumulative counters, so a short burst can't be missed. | Peak RSS per node; swap activity via `vmstat` sampling                                                                                                                                                                                                                        | All                                             |

## Evaluation Framework

### Configuration space

A systems configuration is defined as

`c_s = (m, q, r, d, w)`

where:

- \(m\): model architecture and size;
- \(q\): weight quantization and bits per weight;
- \(r\): runtime configuration: llama.cpp/prima.cpp/dllama.cpp
- \(d\): deployment topology and inference engine: single vs cluster;
- \(w\): workload shape;

Since `quality` and `performance` factor different sub-tuples, we measure them separately

### Quality Track

`Quality depends on (m, q, task, decoding params)`

The quality track evaluates model and quantization choices on the Mac M1 reference host using the same GGUF artifacts.

### System Track

`Performance, Memory depends on (m, q, r, d, token counts)`

The systems track uses fixed token shapes and controlled prompts on the Raspberry Pis, measuring performance, memory.

## 2. Dimension

### Model dimension (M)

The preferred anchor family is Qwen3 because it provides a clean dense size ladder and an MoE model.

| Role                  | Preferred candidates              | Purpose                                                               |
| --------------------- | --------------------------------- | --------------------------------------------------------------------- |
| Scale ladder          | Qwen3 0.6B, 1.7B, 4B, 8B, 14B     | Isolate the effect of model scale within one family.                  |
| Cross-family probe    | One of Gemma 3 4B or Llama 3.2 3B | Test whether the 3–4B “sweet spot” is family-specific.                |
| Dense capacity tenant | Qwen3 32B                         | Create the one-node-low-bit versus two-node-higher-bit comparison.    |
| MoE tenant            | Qwen3-30B-A3B                     | Test whether active-parameter sparsity changes the two-node frontier. |

### Compression dimension (Q)

| Compression variable        | Core levels                                                                               | Use                                                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Weight quantization         | Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K where available                                  | Full ladder for the quality frontier. Analyze using measured bits per weight rather than only format names. |
| Capacity-model quantization | At least one single-node-fitting low-bit variant and one two-node-only higher-bit variant | Directly tests “compress versus distribute.” Exact levels are selected from the feasibility screen.         |

### Topology dimension (D)

| Topology | Definition                                                            |
| -------- | --------------------------------------------------------------------- |
| T0       | One Raspberry Pi 5 running llama.cpp                                  |
| T1       | Two Raspberry Pi 5 devices using llama.cpp RPC layer splitting        |
| T2       | Two Raspberry Pi 5 devices using prima.cpp pipelined-ring parallelism |
| T3       | Two Raspberry Pi 5 devices using dllama tensor parallelism            |

### Workload dimension (W)

The core workload suite uses token-controlled archetypes rather than four semantically different prompts:

| Workload | Prompt tokens | Generated tokens | Representative use                             |
| -------- | ------------: | ---------------: | ---------------------------------------------- |
| W1       |           128 |               64 | Command, classification, or short sensor query |
| W2       |           512 |              256 | Typical assistant request                      |
| W3       |         2,048 |              256 | Retrieval-augmented question answering         |
| W4       |         8,192 |              256 | Long document or accumulated session context   |

### Quality tasks (T)

| Capability            | Instrument                                           | Planned subset |
| --------------------- | ---------------------------------------------------- | -------------: |
| General knowledge     | MMLU or an equivalent stratified multiple-choice set |     ~500 items |
| Multi-step reasoning  | GSM8K                                                |     ~200 items |
| Instruction following | IFEval                                               |   ~150 prompts |
| Structured output     | JSON-schema tasks                                    |   ~100 prompts |

**Two-stage quality measurement:**

- Cheap pass on everything. **Perplexity** (how well a model predicts standard WikiText-2 text — lower = better) is a fast, continuous quality score. Run it on all ~40 model × quant combinations to spot the contenders.

- Expensive pass on the winners only. The real tests (MMLU, GSM8K, SQuAD, etc.) cost much more, so run them only on the ~7 best configs from stage 1, not all 40.

## 3. Metrics

## Performance Metrics

| Group       | Metrics                                                                                                          |
| ----------- | ---------------------------------------------------------------------------------------------------------------- |
| Performance | TTFT; prefill rate (tok/s); decode rate (tok/s) / TPOT (ms); end-to-end request latency;                         |
| Resource    | Peak RSS; CPU utilization                                                                                        |
| Quality     | Perplexity (WikiText-2); task-subset accuracy deltas vs. an FP16/Q8 reference; JSON/function-call validity rates |

## 4. Research questions and hypotheses

### RQ1 - Feasibility vs Usability (Can it load vs. Can you use it?):

**Question:** Which model/quantization configurations are loadable, and which are fast enough for interactive or deferred use on one Raspberry Pi 5?

**Hypothesis H1:** More memory changes what loads, but not what is fast enough to use. With 16 GB, massive 14-billion (14B) and 32-billion (32B) parameter models will successfully load without crashing. However, because generating text is bottlenecked by the memory pipe, a 14B model will only type at a painfully slow 1 to 1.5 words per second. Therefore, real-time chatting on a single machine is still restricted to smaller 3B to 4B models.

**We can set a threshold here for TTFT and Generation rate to define the usability of the model.** For example: generate rate ≥5 tok/s + TTFT ≤ 3 s;

### RQ2 — What is the quality–compression–latency frontier?

**Question.** At a fixed memory budget, should the user choose a smaller model at higher precision or a larger model at lower precision?

**Hypothesis H2.** Moderate four- to five-bit quantization will dominate most configurations. Quality will degrade sharply under extreme two- or three-bit compression, although larger models may tolerate compression better than smaller models.

### RQ3 — When does the second node pay?

**Question.** For a fixed model family and workload, when is higher-precision two-node execution better than aggressive one-node quantization?

**Hypothesis H3.** For models that already fit comfortably on one node, adding the second node will not improve batch-1 latency. The second node becomes valuable when it moves a configuration across a feasibility or quality boundary. The key comparison will be the largest model for which a low-bit artifact fits one node while a higher-bit artifact requires two nodes.

### RQ4 — Does prima.cpp outperform basic layer splitting on a homogeneous pair?

**Question.** Does pipelined-ring parallelism provide a measurable benefit over llama.cpp RPC when the two nodes are identical?

**Hypothesis H4.** For in-RAM batch-1 decoding, T1 and T2 may be close because both must execute all model layers sequentially. prima.cpp's advantage is expected to appear under long-prompt, paging, or memory-pressure conditions where prefetch and overlap are useful. A result showing no meaningful advantage in the simple case is still practically valuable.

### RQ5 — Are MoE models natural tenants for a two-node SBC pair? (exploratory)

**Question.** Can a large-total-parameter MoE model provide a better quality-per-second point than dense models when its weights fit only across two nodes?

**Hypothesis H5.** An MoE model (like Qwen3-30B) only activates a tiny fraction of its "brain" for any given word, meaning less data has to move through the memory pipe. If split across two machines, it should run at the speed of a tiny model (3-6 tokens per second) but deliver the high-quality intelligence of a massive 30B model.

## Expected contributions

**C1 — Benchmark suite framework.**
A benchmarking pipeline that automatically deploys models across different hardware setups, measures their text quality, and tracks edge-system hardware slowdowns independently.

**C2 — Controlled topology comparison.**
Benchmark tables and timeline graphs comparing llama.cpp RPC layer splitting against prima.cpp pipelined-ring execution (measuring metrics like tokens-per-second, latency, and network overhead).

**C3 — Quantization-versus-topology rule.**
A chart showing the crossover point where "Quantized Model on 1 Node" becomes worse/better than "Full-Precision Model split across 2 Nodes".

**C4 — Quality-aware edge frontier.**
A scatter plot graphing Quality (Accuracy/Perplexity) vs. Speed (Latency) vs. Memory (RAM). It will explicitly highlight the winning model configurations (including the MoE test) so users can instantly see which setup gives them the best bang for their buck.

## Setup

### Hardware

| Node   | Role                                                 | Spec                                                                    | Notes                                                                                                                                                       |
| ------ | ---------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pi-a` | Single-node testbed; master/root in distributed runs | Raspberry Pi 5, 16GB                                                    | Hosts model files for RPC runs                                                                                                                              |
| `pi-b` | Worker in distributed runs                           | Identical to `pi-a`                                                     | Identical OS image — clone, don't reinstall                                                                                                                 |
| `m1`   | Reference host (quality track E1, S3)                | Mac M1                                                                  | Runs the **same llama.cpp commit, CPU-only build** for the quality track (avoid the Metal backend to minimize numerical divergence from the Pis' NEON path) |
| Link   | Interconnect                                         | Direct Cat-6 cable between `pi-a`↔`pi-b`, static IPs (e.g., 10.0.0.1/2) | Direct cable removes the switch as a confound. The home LAN stays on Wi-Fi or a second interface for control/SSH only.                                      |

### Artifacts: Models x Quantization

| ID  | Model                 | Quants                                        | Approx. size (Q4_K_M) | Role                                                                                                                      |
| --- | --------------------- | --------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| L1  | Qwen3-0.6B            | Q8_0 · Q6_K · Q5_K_M · Q4_K_M · Q3_K_M · Q2_K | ~0.5 GB               | Size ladder                                                                                                               |
| L2  | Qwen3-1.7B            | same six                                      | ~1.1 GB               | Size ladder                                                                                                               |
| L3  | Qwen3-4B              | same six                                      | ~2.5 GB               | Size ladder; expected single-node frontier                                                                                |
| L4  | Qwen3-8B              | same six                                      | ~5.0 GB               | Size ladder; "fits-tightly" boundary on 8 GB                                                                              |
| L5  | Qwen3-14B             | Q4_K_M · Q3_K_M (+ Q4_0 for E3)               | ~9.0 GB               | **Capacity case**: exceeds one 8 GB node, fits across two ▲16GB → fits one node; replace capacity case per assumption box |
| F1  | Gemma-3-4B-it         | Q4_K_M (+ Q8_0)                               | ~2.5 GB               | Cross-family probe at the sweet spot                                                                                      |
| F2  | Llama-3.2-3B-Instruct | Q4_K_M (+ Q8_0)                               | ~2.0 GB               | Cross-family probe; also dllama-native fallback family                                                                    |

### Experiment Design

### S1 — Feasibility map and harness validation

Answers RQ1 (loadability half) · Feeds C1 (validated harness), C3 (infeasible-on-one-node region). Single 16 GB Pi 5. Run order: S1 first — nothing downstream is trusted until the harness gate passes.

All rates are measured end-to-end through the completion endpoint and one common P-e2e — the same method for both runtimes and the same method used in E2 and E3, so every rate in the study is directly comparable. Feasibility (Verdict + Peak RSS) comes from P-mem (OS-level), independent of how inference is driven.

Single node. Configs: L1–L5 x {Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K} + F1, F2 at Q4_K_M (2). Probe P-e2e at pp512/tg128, 5 reps, plus P-mem.

Note: dllama only supports:

- Qwen 3 0.6B Q40
- Qwen 3 1.7B Q40
- Qwen 3 8B Q40
- Qwen 3 14B Q40

Thus we also support Q4_0 for these models when running llama.cpp

**Outputs:** feasibility heatmap (fits / swaps / OOM), prefill & decode tok/s, peak RSS per config; sanity check against published community numbers

### S3 — Cross-host quality-equivalence check

20 prompts (5 factual QA, 5 GSM8K, 5 instruction-following, 5 JSON-output), greedy, identical GGUF + commit: `pi-a` (4 threads) vs `m1` CPU-only (4 and 8 threads, to bound thread-count nondeterminism). Metrics: exact-match rate, index of first diverging token, semantic-equivalence judgment for non-exact cases.
**Acceptance:** ≥ 90% exact match or all divergences semantically neutral → quality track runs entirely on `m1`. **Fallback:** if violated, frontier configs (≤ 6) get on-device spot evaluation of one instrument (GSM8K-50) on `pi-a` — slow but bounded.

### E1 — Quality–compression frontier

**Stage 1 — perplexity over the full grid.** `llama-perplexity` on WikiText-2 (raw, test split), ctx 2048, for L1–L5 × the full quant ladder {Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K} where artifacts exist, plus F1/F2 at {Q8_0, Q4_K_M} — ≈ 34–38 runs.
**Outputs:** PPL vs. bits-per-weight per size tier; the **equal-memory comparison** (e.g., 8B-Q3 vs. 4B-Q6 vs. 1.7B-Q8 at matched footprint) that tests H1.

**Stage 2 — downstream subsets on frontier candidates only.** Select ≤ 6 configs. Instruments, all automated, all with `/no_think`, greedy:

| Instrument             | Size         | Scoring                             |
| ---------------------- | ------------ | ----------------------------------- |
| MMLU stratified subset | 500 items    | accuracy (log-prob or letter-match) |
| GSM8K subset           | 200 items    | exact final-answer match            |
| IFEval subset          | ~150 prompts | rule-based pass rate                |
| JSON-schema task       | 100 prompts  | validity + field-accuracy rate      |

**Frontier selection rule:** from Stage 1, keep the configs that are Pareto-optimal in (PPL, RAM footprint) _and_ feasible per S1, biased to include: best sub-2 GB config, best ~2.5 GB config (expected L3-Q4_K_M), best single-node ~5 GB config (L4-Q4_K_M), one quality-floor probe (Q3 tier), one cross-family (F1 or F2). These ≤ 6 configs are the only ones evaluated downstream — this is the decoupling dividend.

### E2 — Single-node systems characterization

**Factors:** the ≤ 6 frontier configs × prompt length p ∈ {128, 512, 2048, 8192†} × generation g ∈ {64, 256}, batch 1, via P-e2e; one P-e2e run per config at p = 512 to quantify server overhead. († 8192 only where KV fits in RAM at f16 — record refusals as data.)
≈ 6 × 7 cells × 5 reps ≈ 210 short runs + cool-downs.
**Outputs:** TTFT and prefill tok/s vs. p (compute-bound curve); decode tok/s vs. p (KV-pressure curve); peak RSS vs. p; the **n = 1 baselines** that E3 compares against. Figure: usability map — which configs clear ≥ 5 tok/s decode and TTFT ≤ 3 s at p = 512.

### E3 — Two-node decision study

**Model cells (chosen to span the feasibility boundary, all at matched Q4_0 per §2 policy):**

| Cell | Config         | Single-node status (8 GB)      | What it tests                                                                         |
| ---- | -------------- | ------------------------------ | ------------------------------------------------------------------------------------- |
| M-a  | Qwen3-4B Q4_0  | Fits comfortably (~8–11 tok/s) | Control: distribution should _not_ pay (H3a)                                          |
| M-b  | Qwen3-8B Q4_0  | Fits tightly (~2 tok/s)        | Does n = 2 lift a slow-but-feasible model into usability? (H3b)                       |
| M-c  | Qwen3-14B Q4_0 | OOM — infeasible               | Capacity case: n = 2 is the only option; n = 1 arm attempted once and recorded as OOM |

**Arms per model cell:**

- T0: n = 1, llama.cpp local — baseline
- T1: n = 2, llama.cpp RPC layer-split (`rpc-server` on `pi-b`, master on `pi-a`; split flags per the pinned commit's `tools/rpc` README — verify, they evolve);
- T2: n = 2, prima.cpp pipelined-ring.
- T3: n = 2, dllama tensor-parallel (worker on `pi-b`, root + API on `pi-a`);

**Workload:** p ∈ {512, 2048}, g = 256, batch 1,5 reps + warm-up
Cell count: 3 models × 4 arms × 2 prompt lengths × 5 reps = 120 runs (minus infeasible n = 1 arms)

**Analysis & outputs (the paper's core figures):**

1. **Decision map:** (model size × quant) grid colored {single node / two nodes + best strategy / infeasible}, annotated with decode tok/s and TTFT
2. **Strategy ranking table** at n = 2 per model cell: TTFT, decode tok/s, bytes/token, per-node peak RSS.
3. **Speedup statement** for fitting models: S(2) = decode(n=2)/decode(n=1) per strategy — the honest replacement for a scaling curve.
4. **Communication model vs. measurement:** analytic per-token volume (TP all-reduce ≈ f(hidden size, layers, nodes); layer-split ≈ one hidden-state transfer per cut) plotted against measurements — turns raw numbers into an explanatory result and supports generalization beyond n = 2.
