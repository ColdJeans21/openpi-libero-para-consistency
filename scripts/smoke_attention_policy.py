"""GPU preflight deferred until the existing training/evaluation queue finishes."""

import argparse
import json
import math

from flax import nnx
import numpy as np

from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    policy = policy_config.create_trained_policy(config.get_config("pi0_libero_goal_lora_canonical"), args.checkpoint)
    rng = np.random.default_rng(7)
    obs = libero_policy.make_libero_example()
    obs["observation/image"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    obs["observation/wrist_image"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    obs["prompt"] = "put the bowl on the plate"
    noise = rng.standard_normal((10, 32)).astype(np.float32)
    model = policy._model  # noqa: SLF001
    before = {key: value.value for key, value in nnx.state(model, nnx.Param).flat_state().items()}
    first = policy.infer(obs, noise=noise)["actions"]
    model.instruction_attention_bias = math.log(2)
    policy._sample_actions = nnx_utils.module_jit(model.sample_actions)  # noqa: SLF001
    intervened = policy.infer(obs, noise=noise)["actions"]
    model.instruction_attention_bias = 0.0
    policy._sample_actions = nnx_utils.module_jit(model.sample_actions)  # noqa: SLF001
    restored = policy.infer(obs, noise=noise)["actions"]
    np.testing.assert_array_equal(first, restored)
    assert np.isfinite(intervened).all()
    delta = float(np.max(np.abs(first - intervened)))
    assert delta > 1e-8
    assert all(before[key] is value.value for key, value in nnx.state(model, nnx.Param).flat_state().items())
    print(
        json.dumps(
            {
                "zero_bias_restores_exact_actions": True,
                "parameters_unchanged": True,
                "finite_actions": True,
                "intervention_action_delta": delta,
            }
        )
    )


if __name__ == "__main__":
    main()
