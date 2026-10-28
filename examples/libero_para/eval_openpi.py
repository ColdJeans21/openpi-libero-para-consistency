#!/usr/bin/env python3
"""Run the official LIBERO-Para evaluator against an openpi policy server.

This intentionally delegates to the upstream evaluator instead of maintaining a
fork of its 4,092-instruction protocol in openpi.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-para-root", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tasks", type=int, default=-1)
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
    subprocess.run(command, cwd=root, check=True)

    # The upstream filename predates π0 support and hard-codes this metadata
    # label. The WebSocket protocol and evaluator are otherwise model-agnostic.
    meta_path = output_dir / "meta.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["model_family"] = "pi0"
        metadata["evaluator"] = "LIBERO-Para/eval_scripts/examples/eval_pi05.py"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
