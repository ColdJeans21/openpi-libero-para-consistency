"""Reload a real semantic-interface checkpoint and exercise cached and unseen text."""

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.shared.semantic_text import file_sha256
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    rng = np.random.default_rng(42)
    observation = libero_policy.make_libero_example()
    observation["observation/image"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    observation["observation/wrist_image"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    observation["observation/state"] = rng.random(8)
    observation["prompt"] = "put the bowl on the plate"
    noise = rng.normal(size=(cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
    first = policy.infer(observation, noise=noise)["actions"]
    repeat = policy.infer(observation, noise=noise)["actions"]
    np.testing.assert_array_equal(first, repeat)
    asset = Path(args.checkpoint) / "assets" / "semantic_features.npz"
    before = file_sha256(asset) if cfg.model.semantic_rank else None
    # This is deliberately absent from the training-only text cache. No action
    # labels or benchmark instructions are used in this deployment smoke test.
    observation["prompt"] = "Please move the bowl onto the plate now."
    unseen = policy.infer(observation, noise=noise)["actions"]
    assert unseen.shape == (10, 7)
    assert np.isfinite(unseen).all()
    if before is not None:
        assert file_sha256(asset) == before
    observation["prompt"] = "turn on the stove"
    different = policy.infer(observation, noise=noise)["actions"]
    difference = float(np.max(np.abs(first - different)))
    if difference <= 1e-8:
        raise AssertionError("Trained policy is insensitive to different text")
    switch_delta = None
    if cfg.model.semantic_preserve_prompt:
        # module_jit captures module state, so construct a fresh callable after
        # changing the static scale; changing the model alone would test stale JIT state.
        policy._model.semantic_residual_scale = 0.0  # noqa: SLF001
        policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)  # noqa: SLF001
        disabled = policy.infer(observation, noise=noise)["actions"]
        switch_delta = float(np.max(np.abs(different - disabled)))
        assert np.isfinite(disabled).all()
        assert switch_delta > 1e-8
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "finite_actions": True,
                "action_shape": list(first.shape),
                "fixed_noise_repeat_exact": True,
                "unseen_instruction_cpu_encoding": bool(cfg.model.semantic_rank),
                "training_cache_unchanged": before is not None,
                "different_task_max_action_delta": difference,
                "semantic_cache_sha256": before,
                "preserves_original_language_prefix": cfg.model.semantic_preserve_prompt or not cfg.model.semantic_rank,
                "residual_switch_max_action_delta": switch_delta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
