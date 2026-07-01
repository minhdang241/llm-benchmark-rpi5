#!/usr/bin/env python3
"""Download Distributed Llama Q40 models for S1 via launch.py."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only outside the venv
    yaml = None


@dataclass
class DllamaModel:
    model_id: str
    model: str
    launch_name: str
    expected_files: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Distributed Llama Q40 model/tokenizer files for S1"
    )
    parser.add_argument("--manifest", default="s1/download_dllama_manifest.yaml")
    parser.add_argument("--distributed-llama-dir", default="../distributed-llama")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Optional external models directory, e.g. '/Volumes/Anh Linh/dllama-models'.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Keep models whose id/name contains this text; repeatable",
    )
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run launch.py even if files exist")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = manifest_path.resolve().parents[1]
    dllama_dir = Path(args.distributed_llama_dir)
    if not dllama_dir.is_absolute():
        dllama_dir = root / dllama_dir
    dllama_dir = dllama_dir.resolve()

    if not (dllama_dir / "launch.py").exists():
        raise SystemExit(f"Could not find distributed-llama launch.py: {dllama_dir / 'launch.py'}")

    if args.models_dir:
        configure_models_dir(dllama_dir, Path(args.models_dir))

    models = load_models(manifest_path)
    if args.only:
        models = [
            model
            for model in models
            if all(needle.lower() in key(model).lower() for needle in args.only)
        ]

    print(f"dllama downloads: {len(models)}")
    for model in models:
        model_dir = dllama_dir / "models" / model.launch_name
        if expected_files_exist(model_dir, model) and not args.force:
            print(f"exists {key(model)} -> {model_dir}")
            continue

        command = [
            args.python_bin,
            "launch.py",
            model.launch_name,
            "-skip-run",
            "-y",
        ]
        if args.print_commands:
            print(f"cd {shell_quote(str(dllama_dir))}")
            print(" ".join(shell_quote(part) for part in command))
            continue

        print(f"download {key(model)}")
        completed = subprocess.run(command, cwd=dllama_dir, check=False)
        if completed.returncode != 0:
            print(f"FAILED {key(model)} via launch.py", file=sys.stderr)
            return completed.returncode

        missing = missing_expected_files(model_dir, model)
        if missing:
            print(
                f"FAILED {key(model)} missing expected files after download: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1

        print(f"saved {model_dir}")

    return 0


def load_models(path: Path) -> list[DllamaModel]:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read s1/download_dllama_manifest.yaml. "
            "Run with venv/bin/python or install requirements.txt."
        )
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    return [
        DllamaModel(
            model_id=item["model_id"],
            model=item["model"],
            launch_name=item["launch_name"],
            expected_files=list(item["expected_files"]),
        )
        for item in manifest["models"]
    ]


def configure_models_dir(dllama_dir: Path, models_dir: Path) -> None:
    models_dir = models_dir.expanduser()
    models_dir.mkdir(parents=True, exist_ok=True)

    link = dllama_dir / "models"
    if link.is_symlink():
        current = Path(os.readlink(link))
        if not current.is_absolute():
            current = (link.parent / current).resolve()
        if current.resolve() == models_dir.resolve():
            return
        raise SystemExit(f"{link} is already a symlink to {current}; update it manually.")

    if link.exists():
        raise SystemExit(
            f"{link} already exists and is not a symlink. Move it aside first, then rerun."
        )

    link.symlink_to(models_dir)
    print(f"linked {link} -> {models_dir}")


def expected_files_exist(model_dir: Path, model: DllamaModel) -> bool:
    return not missing_expected_files(model_dir, model)


def missing_expected_files(model_dir: Path, model: DllamaModel) -> list[str]:
    return [name for name in model.expected_files if not (model_dir / name).exists()]


def key(model: DllamaModel) -> str:
    return f"{model.model_id}:Q4_0"


def shell_quote(value: str) -> str:
    if not value or any(char.isspace() or char in "'\"\\$`" for char in value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


if __name__ == "__main__":
    raise SystemExit(main())

