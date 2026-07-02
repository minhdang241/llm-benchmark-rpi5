"""Parsers for S1 command output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LlamaMetrics:
    load_time_ms: Optional[float] = None
    prompt_eval_time_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    prompt_rate_tps: Optional[float] = None
    eval_time_ms: Optional[float] = None
    eval_tokens: Optional[int] = None
    eval_rate_tps: Optional[float] = None
    total_time_ms: Optional[float] = None
    total_tokens: Optional[int] = None
    ttft_ms: Optional[float] = None
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class TimeMetrics:
    max_rss_kb: Optional[int] = None
    user_time_seconds: Optional[float] = None
    system_time_seconds: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    exit_status: Optional[int] = None
    raw_unit: str = ""
    parse_errors: list[str] = field(default_factory=list)


def _float(value: str) -> float:
    return float(value.replace(",", ""))


def _int(value: str) -> int:
    return int(value.replace(",", ""))


def parse_llamacpp_output(stderr_text: str, stdout_text: str = "") -> LlamaMetrics:
    """Extract llama.cpp timing metrics.

    This is a hardened version of the old benchmark parser. It accepts both
    newer `llama_perf_context_print` and older `llama_print_timings` prefixes.
    """

    text = stderr_text + "\n" + stdout_text
    result = LlamaMetrics()

    match = re.search(r"load time\s*=\s*([\d.,]+)\s*ms", text, re.IGNORECASE)
    if match:
        result.load_time_ms = _float(match.group(1))
    else:
        result.parse_errors.append("load_time not found")

    match = re.search(
        r"prompt eval time\s*=\s*([\d.,]+)\s*ms\s*/\s*(\d+)\s*tokens?"
        r"\s*\(\s*([\d.,]+)\s*ms per token,\s*([\d.,]+)\s*tokens per second\)",
        text,
        re.IGNORECASE,
    )
    if match:
        result.prompt_eval_time_ms = _float(match.group(1))
        result.prompt_tokens = _int(match.group(2))
        result.prompt_rate_tps = _float(match.group(4))
    else:
        result.parse_errors.append("prompt_eval not found")

    match = re.search(
        r"(?<!prompt\s)eval time\s*=\s*([\d.,]+)\s*ms\s*/\s*(\d+)\s*(?:runs?|tokens?)"
        r"\s*\(\s*([\d.,]+)\s*ms per token,\s*([\d.,]+)\s*tokens per second\)",
        text,
        re.IGNORECASE,
    )
    if match:
        result.eval_time_ms = _float(match.group(1))
        result.eval_tokens = _int(match.group(2))
        result.eval_rate_tps = _float(match.group(4))
    else:
        result.parse_errors.append("eval not found")

    match = re.search(
        r"total time\s*=\s*([\d.,]+)\s*ms(?:\s*/\s*(\d+)\s*tokens?)?",
        text,
        re.IGNORECASE,
    )
    if match:
        result.total_time_ms = _float(match.group(1))
        if match.group(2):
            result.total_tokens = _int(match.group(2))

    if result.prompt_eval_time_ms and result.eval_time_ms and result.eval_tokens:
        result.ttft_ms = round(
            result.prompt_eval_time_ms + (result.eval_time_ms / result.eval_tokens),
            2,
        )

    return result


def parse_dllama_output(stderr_text: str, stdout_text: str = "") -> LlamaMetrics:
    """Extract Distributed Llama timing metrics."""

    text = (stderr_text + "\n" + stdout_text).replace("\r\n", "\n").replace("\r", "\n")
    result = LlamaMetrics(load_time_ms=0.0)

    match = re.search(r"Eval\s+(\d+)\s+ms", text)
    if match:
        result.prompt_eval_time_ms = float(match.group(1))

    match = re.search(
        r"Evaluation\s+nBatches:\s+\d+\s+nTokens:\s+(\d+)\s+tokens/s:\s+([\d.]+)\s+\(([\d.]+)\s+ms/tok\)",
        text,
    )
    if match:
        result.prompt_tokens = int(match.group(1))
        result.prompt_rate_tps = float(match.group(2))
        if result.prompt_eval_time_ms is None:
            result.prompt_eval_time_ms = round(int(match.group(1)) * float(match.group(3)), 2)
    else:
        result.parse_errors.append("Evaluation section not found")

    match = re.search(
        r"Prediction\s+nTokens:\s+(\d+)\s+tokens/s:\s+([\d.]+)\s+\(([\d.]+)\s+ms/tok\)",
        text,
    )
    if match:
        result.eval_tokens = int(match.group(1))
        result.eval_rate_tps = float(match.group(2))
        result.eval_time_ms = round(int(match.group(1)) * float(match.group(3)), 2)
    else:
        result.parse_errors.append("Prediction section not found")

    pred_times = re.findall(r"Pred\s+(\d+)\s+ms", text)
    if pred_times and result.prompt_eval_time_ms is not None:
        result.ttft_ms = result.prompt_eval_time_ms + float(pred_times[0])

    if result.prompt_tokens is not None and result.eval_tokens is not None:
        result.total_tokens = result.prompt_tokens + result.eval_tokens

    return result


def parse_time_output(text: str, platform_name: str) -> TimeMetrics:
    """Parse GNU `/usr/bin/time -v` or macOS `/usr/bin/time -l` output."""

    result = TimeMetrics()

    linux_rss = re.search(
        r"Maximum resident set size \(kbytes\):\s*(\d+)", text, re.IGNORECASE
    )
    mac_rss = re.search(r"^\s*(\d+)\s+maximum resident set size", text, re.MULTILINE)

    if linux_rss:
        result.max_rss_kb = _int(linux_rss.group(1))
        result.raw_unit = "kbytes"
    elif mac_rss:
        result.max_rss_kb = round(_int(mac_rss.group(1)) / 1024)
        result.raw_unit = "bytes"
    else:
        result.parse_errors.append("max_rss not found")

    elapsed = re.search(
        r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text, re.IGNORECASE
    )
    if elapsed:
        result.elapsed_seconds = parse_gnu_elapsed(elapsed.group(1))
    else:
        mac_elapsed = re.search(
            r"^\s*([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys\b",
            text,
            re.MULTILINE,
        )
        if mac_elapsed:
            result.elapsed_seconds = _float(mac_elapsed.group(1))
            result.user_time_seconds = _float(mac_elapsed.group(2))
            result.system_time_seconds = _float(mac_elapsed.group(3))

    user_time = re.search(r"User time \(seconds\):\s*([\d.]+)", text, re.IGNORECASE)
    if user_time:
        result.user_time_seconds = _float(user_time.group(1))

    system_time = re.search(r"System time \(seconds\):\s*([\d.]+)", text, re.IGNORECASE)
    if system_time:
        result.system_time_seconds = _float(system_time.group(1))

    status = re.search(r"Exit status:\s*(-?\d+)", text, re.IGNORECASE)
    if status:
        result.exit_status = int(status.group(1))

    return result


def parse_gnu_elapsed(value: str) -> Optional[float]:
    parts = value.strip().split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        return float(parts[0])
    except ValueError:
        return None
