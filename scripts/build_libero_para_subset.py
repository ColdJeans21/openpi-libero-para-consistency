#!/usr/bin/env python3
"""Build a deterministic, task-balanced LIBERO-Para validation subset."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

FILENAME_RE = re.compile(r"^(act|obj|comp)_.+_eval(\d+)_ver\d+\.bddl$")
DEFAULT_QUOTAS = {"act": 2, "obj": 1, "comp": 7}


def _stable_score(filename: str, seed: int) -> str:
    return hashlib.sha256(f"libero-para-validation-v1:{seed}:{filename}".encode()).hexdigest()


def _upstream_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-para-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = args.libero_para_root.resolve()
    bddl_dir = root / "libero" / "libero" / "bddl_files" / "libero_para"
    groups: dict[tuple[int, str], list[str]] = {}
    for path in bddl_dir.glob("*.bddl"):
        match = FILENAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected LIBERO-Para BDDL filename: {path.name}")
        paraphrase_type, eval_id = match.group(1), int(match.group(2))
        groups.setdefault((eval_id, paraphrase_type), []).append(path.name)

    selected = []
    for eval_id in range(10):
        for paraphrase_type, quota in DEFAULT_QUOTAS.items():
            candidates = groups.get((eval_id, paraphrase_type), [])
            if len(candidates) < quota:
                raise ValueError(
                    f"eval{eval_id}/{paraphrase_type} has {len(candidates)} candidates; need {quota}"
                )
            candidates.sort(key=lambda filename: (_stable_score(filename, args.seed), filename))
            selected.extend(candidates[:quota])

    selected.sort()
    counts = Counter()
    for filename in selected:
        match = FILENAME_RE.fullmatch(filename)
        assert match is not None
        counts[f"eval{match.group(2)}"] += 1
        counts[match.group(1)] += 1

    manifest = {
        "schema_version": 1,
        "name": "libero_para_validation_subset_v1",
        "description": (
            "Checkpoint-curve subset: 10 paraphrases per original task, with "
            "7 comp, 2 act, and 1 obj paraphrases per task. Selection uses a stable SHA-256 score."
        ),
        "upstream_repository": "https://github.com/cau-hai-lab/LIBERO-Para.git",
        "upstream_commit": _upstream_commit(root),
        "selection_seed": args.seed,
        "selection_method": "lowest_sha256(libero-para-validation-v1:seed:filename)",
        "quota_per_eval": DEFAULT_QUOTAS,
        "counts": dict(sorted(counts.items())),
        "bddl_files": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "size": len(selected), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
