#!/usr/bin/env python3
"""Run the official LIBERO-Para evaluator against an openpi policy server.

This intentionally delegates to the upstream evaluator instead of maintaining a
fork of its 4,092-instruction protocol in openpi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _load_task_manifest(manifest_path: Path) -> tuple[dict, list[str]]:
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in task manifest: {manifest_path}")
    filenames = manifest.get("bddl_files")
    if not isinstance(filenames, list) or not filenames:
        raise ValueError(f"Task manifest must contain a non-empty bddl_files list: {manifest_path}")
    if not all(isinstance(filename, str) for filename in filenames):
        raise ValueError(f"All bddl_files entries must be strings: {manifest_path}")
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Task manifest contains duplicate filenames: {manifest_path}")
    return manifest, filenames


def _manifest_digest(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-para-root", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="JSON manifest containing a fixed bddl_files subset. Intended for checkpoint curves.",
    )
    parser.add_argument("--model-name", default="pi0-libero-goal-consistency")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.libero_para_root.resolve()
    evaluator = root / "eval_scripts" / "examples" / "eval_pi05.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"Official evaluator not found: {evaluator}")

    output_dir = args.output_dir.resolve()
    command = [
        sys.executable,
        str(evaluator),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--max_tasks",
        str(args.max_tasks),
        "--mode",
        "para",
        "--model_name",
        args.model_name,
        "--output_dir",
        str(output_dir),
    ]
    manifest = None
    manifest_path = args.task_manifest.resolve() if args.task_manifest else None
    if manifest_path is None:
        subprocess.run(command, cwd=root, check=True)
    else:
        manifest, filenames = _load_task_manifest(manifest_path)
        source_dir = root / "libero" / "libero" / "bddl_files" / "libero_para"
        with tempfile.TemporaryDirectory(prefix="libero-para-subset-") as subset_dir_str:
            subset_dir = Path(subset_dir_str)
            for filename in filenames:
                if Path(filename).name != filename or not filename.endswith(".bddl"):
                    raise ValueError(f"Unsafe BDDL filename in manifest: {filename!r}")
                source = source_dir / filename
                if not source.is_file():
                    raise FileNotFoundError(f"Manifest BDDL file not found: {source}")
                shutil.copy2(source, subset_dir / filename)
            command.extend(["--bddl_dir", str(subset_dir)])
            subprocess.run(command, cwd=root, check=True)

    # The upstream filename predates π0 support and hard-codes this metadata
    # label. The WebSocket protocol and evaluator are otherwise model-agnostic.
    meta_path = output_dir / "meta.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["model_family"] = "pi0"
        metadata["evaluator"] = "LIBERO-Para/eval_scripts/examples/eval_pi05.py"
        if manifest is not None:
            metadata["task_manifest"] = str(manifest_path)
            metadata["task_manifest_sha256"] = _manifest_digest(manifest)
            metadata["task_manifest_size"] = len(manifest["bddl_files"])
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    if manifest is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "task_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
