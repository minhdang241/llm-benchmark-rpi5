"""
Parsers for extracting timing metrics from framework output.

llama.cpp prints timing info to stderr in a known format.
Distributed Llama has a different output format.
Both parsers return a standardised dict.
"""

import re


def parse_llama_cpp_output(
    stderr_text: str, stdout_text: str, wall_time_ms: float
) -> dict:
    """
    Parse llama.cpp stderr timing lines.

    Expected format (may vary slightly across versions):
        llama_perf_context_print:        load time =    1234.56 ms
        llama_perf_sampler_print:    sampling time =      12.34 ms /    50 runs   (...)
        llama_perf_context_print: prompt eval time =     567.89 ms /    25 tokens (   22.72 ms per token,    44.02 tokens per second)
        llama_perf_context_print:        eval time =    5678.90 ms /    49 runs   (  115.90 ms per token,     8.63 tokens per second)
        llama_perf_context_print:       total time =    6789.01 ms /    74 tokens

    Also handles older format with llama_print_timings prefix.
    """

    result = {
        "load_time_ms": 0.0,
        "prompt_eval_time_ms": 0.0,
        "prompt_tokens": 0,
        "prompt_rate_tps": 0.0,
        "eval_time_ms": 0.0,
        "eval_tokens": 0,
        "eval_rate_tps": 0.0,
        "total_time_ms": wall_time_ms,
        "total_tokens": 0,
        "ttft_ms": 0.0,  # estimated as prompt_eval_time + first token eval
        "generated_text": stdout_text.strip(),
        "parse_errors": [],
    }

    text = stderr_text

    # --- load time ---
    m = re.search(r"load time\s*=\s*([\d.]+)\s*ms", text)
    if m:
        result["load_time_ms"] = float(m.group(1))
    else:
        result["parse_errors"].append("load_time not found")

    # --- prompt eval ---
    m = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens?\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        text,
    )
    if m:
        result["prompt_eval_time_ms"] = float(m.group(1))
        result["prompt_tokens"] = int(m.group(2))
        result["prompt_rate_tps"] = float(m.group(4))
    else:
        result["parse_errors"].append("prompt_eval not found")

    # --- eval (generation) ---
    m = re.search(
        r"(?<!prompt\s)eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs?\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        text,
    )
    if m:
        result["eval_time_ms"] = float(m.group(1))
        result["eval_tokens"] = int(m.group(2))
        result["eval_rate_tps"] = float(m.group(4))
    else:
        result["parse_errors"].append("eval not found")

    # --- total tokens ---
    # total_time_ms stays as wall_time_ms (set at init) for consistency with dllama.
    m = re.search(r"total time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens?", text)
    if m:
        result["total_tokens"] = int(m.group(1))
    
    # TTFT estimate: prompt processing + one token generation step
    if result["prompt_eval_time_ms"] > 0 and result["eval_tokens"] > 0:
        per_token_ms = result["eval_time_ms"] / result["eval_tokens"]
        result["ttft_ms"] = round(result["prompt_eval_time_ms"] + per_token_ms, 2)

    return result


def parse_dllama_output(
    stderr_text: str, stdout_text: str, wall_time_ms: float
) -> dict:
    """
    Parse Distributed Llama output.

    Expected format:
        🔷️ Eval  111 ms Sync    0 ms | Sent     0 kB Recv     0 kB | (10 tokens)
        🔶 Pred   18 ms Sync    0 ms | Sent     0 kB Recv     0 kB |  Edge
        ...
        Evaluation
           nBatches: 32
            nTokens: 10
           tokens/s: 89.71 (11.15 ms/tok)
        Prediction
            nTokens: 40
           tokens/s: 62.05 (16.12 ms/tok)
    """
    combined = stderr_text + "\n" + stdout_text
    result = {
        "load_time_ms": 0.0,
        "prompt_eval_time_ms": 0.0,
        "prompt_tokens": 0,
        "prompt_rate_tps": 0.0,
        "eval_time_ms": 0.0,
        "eval_tokens": 0,
        "eval_rate_tps": 0.0,
        "total_time_ms": wall_time_ms,
        "total_tokens": 0,
        "ttft_ms": 0.0,
        "generated_text": "",
        "parse_errors": [],
    }

    # --- Prompt eval time from the 🔷 Eval line ---
    m = re.search(r"Eval\s+(\d+)\s+ms", combined)
    if m:
        result["prompt_eval_time_ms"] = float(m.group(1))

    # --- Evaluation (prompt) summary block ---
    m = re.search(
        r"Evaluation\s+nBatches:\s+\d+\s+nTokens:\s+(\d+)\s+tokens/s:\s+([\d.]+)\s+\(([\d.]+)\s+ms/tok\)",
        combined,
    )
    if m:
        result["prompt_tokens"] = int(m.group(1))
        result["prompt_rate_tps"] = float(m.group(2))
        # Use ms/tok * nTokens if Eval line wasn't found
        if result["prompt_eval_time_ms"] == 0.0:
            result["prompt_eval_time_ms"] = round(int(m.group(1)) * float(m.group(3)), 2)
    else:
        result["parse_errors"].append("Evaluation section not found")

    # --- Prediction (generation) summary block ---
    m = re.search(
        r"Prediction\s+nTokens:\s+(\d+)\s+tokens/s:\s+([\d.]+)\s+\(([\d.]+)\s+ms/tok\)",
        combined,
    )
    if m:
        result["eval_tokens"] = int(m.group(1))
        result["eval_rate_tps"] = float(m.group(2))
        result["eval_time_ms"] = round(int(m.group(1)) * float(m.group(3)), 2)
    else:
        result["parse_errors"].append("Prediction section not found")

    # --- TTFT: prompt eval time + first prediction step time ---
    pred_times = re.findall(r"Pred\s+(\d+)\s+ms", combined)
    if pred_times and result["prompt_eval_time_ms"] > 0:
        result["ttft_ms"] = result["prompt_eval_time_ms"] + float(pred_times[0])

    # --- Generated text: tokens from Pred lines (after the last | ) ---
    token_matches = re.findall(r"Pred.*\|\s+(.+)$", combined, re.MULTILINE)
    if token_matches:
        result["generated_text"] = "".join(token_matches).strip()
    else:
        result["parse_errors"].append("generated text not found in Pred lines")

    result["total_tokens"] = result["prompt_tokens"] + result["eval_tokens"]

    return result
