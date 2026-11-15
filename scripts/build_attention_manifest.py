"""Build a fixed development/test manifest using BDDL goals, not model outcomes."""

import argparse
import json
from pathlib import Path

import torch

from openpi.evaluation.attention_protocol import PAIRS
from openpi.evaluation.attention_protocol import bddl_sections
from openpi.evaluation.attention_protocol import digest
from openpi.evaluation.attention_protocol import file_digest
from openpi.evaluation.attention_protocol import physical_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, default=Path("third_party/libero/libero/libero"))
    parser.add_argument("--para-root", type=Path, default=Path("/home/gml/xz/LIBERO-Para"))
    parser.add_argument("--output", type=Path, default=Path("examples/libero_attention/manifest_v1.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {args.output}")
    subset = Path("examples/libero_para/validation_subset_v1_seed7.json")
    training = json.loads(Path("examples/libero_para/train_paraphrases_v1.json").read_text())["paraphrases"]
    training_texts = {text.strip().casefold() for values in training.values() for text in values}
    filenames = json.loads(subset.read_text())["bddl_files"]
    candidates = []
    for name in filenames:
        if name.startswith("act_"):
            path = args.para_root / "libero/libero/bddl_files/libero_para" / name
            sections = bddl_sections(path.read_text())
            candidates.append((name, sections, file_digest(path)))
    pairs = []
    for pair_id, task_a, task_b in PAIRS:
        targets = []
        for name in (task_a, task_b):
            path = args.libero_root / "bddl_files/libero_goal" / f"{name}.bddl"
            sections = bddl_sections(path.read_text())
            matches = [
                (n, s, sha)
                for n, s, sha in candidates
                if s[":goal"] == sections[":goal"] and physical_signature(s) == physical_signature(sections)
            ]
            if not matches:
                raise ValueError(f"No fixed held-out same-goal paraphrase for {name}")
            para_name, para_sections, para_sha = min(matches, key=lambda x: digest([7, x[0]]))
            if " ".join(para_sections[":language"]).strip().casefold() in training_texts:
                raise ValueError("Selected held-out instruction occurs in the training paraphrases")
            targets.append(
                {
                    "task": name,
                    "canonical": name.replace("_", " "),
                    "paraphrase": " ".join(para_sections[":language"]),
                    "paraphrase_file": para_name,
                    "paraphrase_sha256": para_sha,
                    "bddl_sha256": file_digest(path),
                    "goal": sections[":goal"],
                    "physical_sha256": physical_signature(sections),
                }
            )
        if targets[0]["physical_sha256"] != targets[1]["physical_sha256"]:
            raise ValueError(f"Pair {pair_id} changes physical scene")
        initial = args.libero_root / "init_files/libero_goal" / f"{task_a}.pruned_init"
        states = torch.load(initial, map_location="cpu", weights_only=False)
        if len(states) < 30:
            raise ValueError("Need at least30 disjoint fixed initial states")
        order = sorted(range(len(states)), key=lambda i: digest(["attention-states-v1", 7, pair_id, i]))
        pairs.append(
            {
                "pair_id": pair_id,
                "targets": targets,
                "initial_state_source": task_a,
                "initial_states_sha256": file_digest(initial),
                "dev_state_indices": order[:10],
                "test_state_indices": order[10:30],
            }
        )
    manifest = {
        "schema_version": 1,
        "seed": 7,
        "name": "instruction_attention_counterfactual_v1",
        "selection": "physics/goal match and stable SHA256 only; no success labels read",
        "paraphrase_manifest_sha256": file_digest(subset),
        "pairs": pairs,
        "dev_episodes": 80,
        "test_episodes": 160,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "sha256": file_digest(args.output), "dev_episodes": 80, "test_episodes": 160}
        )
    )


if __name__ == "__main__":
    main()
