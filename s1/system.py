"""Host and OS counters used by the S1 harness."""

from __future__ import annotations

import os
import platform
import socket
from pathlib import Path
from typing import Optional


def host_metadata() -> dict:
    return {
        "host_id": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system().lower(),
        "python": platform.python_version(),
    }


def total_memory_kb() -> Optional[int]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round((page_size * pages) / 1024)
    except (AttributeError, OSError, ValueError):
        return None


def read_vmstat() -> dict[str, int]:
    path = Path("/proc/vmstat")
    if not path.exists():
        return {}

    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in {"pswpin", "pswpout", "pgswapin", "pgswapout"}:
            values[parts[0]] = int(parts[1])
    return values


def swap_delta(before: dict[str, int], after: dict[str, int]) -> tuple[Optional[int], Optional[int]]:
    if not before or not after:
        return None, None

    in_key = "pgswapin" if "pgswapin" in before or "pgswapin" in after else "pswpin"
    out_key = "pgswapout" if "pgswapout" in before or "pgswapout" in after else "pswpout"
    return (
        after.get(in_key, 0) - before.get(in_key, 0),
        after.get(out_key, 0) - before.get(out_key, 0),
    )

