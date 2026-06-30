#!/usr/bin/env python3
"""Download S1 model artifacts to the paths expected by the harness."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only outside the venv
    yaml = None


@dataclass
class Download:
    model_id: str
    model: str
    quant: str
    repo: str
    source_file: str
    dest: Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download S1 GGUF models from Hugging Face")
    parser.add_argument("--manifest", default="s1/download_manifest.yaml")
    parser.add_argument("--hf-bin", default="hf")
    parser.add_argument("--only", action="append", default=[], help="Keep downloads whose key contains this text; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Ask Hugging Face what would be downloaded")
    parser.add_argument("--print-commands", action="store_true", help="Print commands without executing them")
    parser.add_argument("--force", action="store_true", help="Re-download even if destination exists")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = manifest_path.resolve().parents[1]
    downloads = expand_downloads(load_manifest(manifest_path), root)

    if args.only:
        downloads = [
            item
            for item in downloads
            if all(needle.lower() in key(item).lower() for needle in args.only)
        ]

    print(f"S1 downloads: {len(downloads)}")
    for item in downloads:
        if item.dest.exists() and not args.force:
            print(f"exists {key(item)} -> {item.dest}")
            continue

        item.dest.parent.mkdir(parents=True, exist_ok=True)
        command = [
            args.hf_bin,
            "download",
            item.repo,
            item.source_file,
            "--local-dir",
            str(item.dest.parent),
            "--max-workers",
            str(args.max_workers),
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.force:
            command.append("--force-download")

        if args.print_commands:
            print(" ".join(shell_quote(part) for part in command))
            continue

        print(f"download {key(item)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"FAILED {key(item)} from {item.repo}/{item.source_file}", file=sys.stderr)
            return completed.returncode

        if not args.dry_run:
            downloaded_path = item.dest.parent / item.source_file
            if downloaded_path != item.dest:
                if not downloaded_path.exists():
                    print(f"FAILED expected downloaded file missing: {downloaded_path}", file=sys.stderr)
                    return 1
                if item.dest.exists():
                    item.dest.unlink()
                shutil.move(str(downloaded_path), str(item.dest))
            print(f"saved {item.dest}")

    return 0


def load_manifest(path: Path) -> dict:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read s1/download_manifest.yaml. "
            "Run with venv/bin/python or install requirements.txt."
        )
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expand_downloads(manifest: dict, root: Path) -> list[Download]:
    downloads: list[Download] = []
    for group in manifest["groups"]:
        for quant in group["quants"]:
            values = {"quant": quant, "quant_lower": quant.lower()}
            dest = Path(group["dest_template"].format(**values))
            if not dest.is_absolute():
                dest = root / dest
            downloads.append(
                Download(
                    model_id=group["model_id"],
                    model=group["model"],
                    quant=quant,
                    repo=group["repo"],
                    source_file=group["source_file_template"].format(**values),
                    dest=dest,
                )
            )
    return downloads


def key(item: Download) -> str:
    return f"{item.model_id}:{item.quant}"


def shell_quote(value: str) -> str:
    if not value or any(char.isspace() or char in "'\"\\$`" for char in value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


if __name__ == "__main__":
    raise SystemExit(main())

