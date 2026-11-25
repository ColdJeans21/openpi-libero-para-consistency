"""Mix complete Goal demonstration pairs with verified counterfactual pairs.

Transform each label with the SAME existing action pipeline (including the
legacy delta transform and normalization); never subtract raw actions from
normalized velocities. Seven demonstration pairs per counterfactual pair.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


class MixedVerifiedPairs:
    def __init__(self, base, root, data_config, *, skip_norm_stats=False):
        from openpi.training.data_loader import transform_dataset

        self.base = base
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema_version") != 2 or not manifest.get("completed"):
            raise ValueError("Verified pair collection must be complete")
        if manifest.get("label_type") != "both_successful_executed_prefix":
            raise ValueError("Unverified alternate labels are not allowed")
        accepted = [r for r in manifest["records"] if r["accepted"]]
        if len(accepted) < 20 or len({r["pair_id"] for r in accepted}) != 2:
            raise ValueError("Need at least 20 paired states covering both pilot families")
        if {r["shard"] for r in accepted} != set(manifest["shards"]):
            raise ValueError("Verification records do not match shards")
        source = Path(__file__).resolve().parents[3] / "examples/libero_attention/manifest_v1.json"
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_manifest_sha256"]:
            raise ValueError("Frozen split changed")
        pairs = {p["pair_id"]: p for p in json.loads(source.read_text())["pairs"]}
        for record in accepted:
            pair = pairs[record["pair_id"]]
            held_out = set(pair["dev_state_indices"]) | set(pair["test_state_indices"])
            if record["state_index"] in held_out or not all(record["branch_success"]) or any(record["wrong_goal"]):
                raise ValueError("Held-out or failed counterfactual pair")
        self.pairs = []
        for relative in manifest["shards"]:
            path = root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["shard_sha256"][relative]:
                raise ValueError("Pair file changed after verification")
            with np.load(path, allow_pickle=False) as item:
                data = {k: item[k] for k in item.files}
            if data["actions"].shape != (10, 7) or data["actions_alt"].shape != (10, 7):
                raise ValueError("Expected two length-10 seven-dimensional chunks")
            if not np.isfinite(data["actions"]).all() or not np.isfinite(data["actions_alt"]).all():
                raise ValueError("Nonfinite actions")
            for reverse in (False, True):
                prompt, alt_prompt = str(data["prompt"].item()), str(data["prompt_alt"].item())
                action, alt = data["actions"], data["actions_alt"]
                if reverse:
                    prompt, alt_prompt, action, alt = alt_prompt, prompt, alt, action
                raw = {k: data[k].copy() for k in ("image", "wrist_image", "state")}
                raw.update(prompt=prompt, prompt_alt=alt_prompt, actions=action.copy())
                second = {**raw, "actions": alt.copy()}
                transformed = transform_dataset([raw, second], data_config, skip_norm_stats=skip_norm_stats)
                first, second = transformed[0], transformed[1]
                first["paired_actions"] = second["actions"]
                first["pair_is_negative"] = np.asarray(1, dtype=np.bool_)
                self.pairs.append(first)

    def __len__(self):
        return ((len(self.base) + 6) // 7) * 8

    def __getitem__(self, index):
        index = int(index)
        block, slot = divmod(index, 8)
        if slot == 7:
            data = self.pairs[block % len(self.pairs)]
        else:
            data = dict(self.base[(block * 7 + slot) % len(self.base)])
            data["paired_actions"] = data["actions"]
            data["pair_is_negative"] = np.asarray(0, dtype=np.bool_)
        # Observations.from_dict normalizes images in place; keep cached samples immutable.
        import copy

        return copy.deepcopy(data)
