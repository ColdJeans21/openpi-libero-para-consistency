"""Validate the real normalized mixed dataset before allocating training GPUs."""

import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.training import config
from openpi.training import data_loader


def main():
    c = config.get_config("pi0_libero_goal_verified_r0")
    dc = c.data.create(c.assets_dirs, c.model)
    base = data_loader.create_torch_dataset(dc, c.model.action_horizon, c.model)
    base = data_loader.transform_dataset(base, dc)
    from openpi.training.verified_pairs import MixedVerifiedPairs

    mixed = MixedVerifiedPairs(base, dc.verified_pair_dataset_path, dc)
    counts = []
    for i in range(16):
        sample = mixed[i]
        assert sample["actions"].shape == sample["paired_actions"].shape == (10, 32)
        assert np.isfinite(sample["actions"]).all()
        assert np.isfinite(sample["paired_actions"]).all()
        negative = bool(sample["pair_is_negative"])
        counts.append(negative)
        if not negative:
            np.testing.assert_array_equal(sample["actions"], sample["paired_actions"])
        assert sample["tokenized_prompt_alt"].shape == sample["tokenized_prompt"].shape
    assert sum(counts) == 2
    # Every run must have literally equal data configuration; losses are the only switches.
    for r in (1, 2):
        other = config.get_config(f"pi0_libero_goal_verified_r{r}")
        assert c.data == other.data
        assert c.batch_size == other.batch_size
        assert c.seed == other.seed
    p = Path(dc.verified_pair_dataset_path) / "manifest.json"
    print(
        json.dumps(
            {
                "base_samples": len(base),
                "mixed_pairs": len(mixed),
                "counterfactual_orientations": len(mixed.pairs),
                "manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "first16_negative": counts,
                "data_equivalent_across_runs": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
