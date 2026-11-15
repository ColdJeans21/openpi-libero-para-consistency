# ruff: noqa: SLF001
import dataclasses
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.semantic_interface_test import TinyVision


def test_action_and_instruction_boundaries_match_full_and_cached_attention():
    ids = jnp.array([[2, 10, 108, 0, 1, 20], [2, 10, 108, 0, 1, 20]])
    mask = jnp.array([[1, 1, 1, 0, 1, 1], [1, 0, 1, 0, 1, 1]], dtype=bool)
    full = pi0.make_instruction_attention_bias(
        ids, mask, prefix_length=9, query_length=13, key_length=13, action_horizon=3, beta=math.log(2)
    )
    assert full.shape == (2, 13, 13)
    expected = np.zeros((2, 13, 13))
    expected[0, -3:, [4, 8]] = math.log(2)
    expected[1, -3:, 8] = math.log(2)
    np.testing.assert_allclose(full, expected)
    cached = pi0.make_instruction_attention_bias(
        ids, mask, prefix_length=9, query_length=4, key_length=13, action_horizon=3, beta=math.log(2)
    )
    np.testing.assert_array_equal(cached, full[:, -4:])


@pytest.mark.parametrize("beta", [0.0, math.log(2), math.log(4)])
def test_euler_cached_sampling_matches_full_velocity_without_parameter_changes(monkeypatch, beta):
    monkeypatch.setattr(pi0._siglip, "Module", lambda **kwargs: TinyVision())
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=7,
        action_horizon=3,
        instruction_attention_bias=beta,
    )
    model = config.create(jax.random.key(0))
    original = dataclasses.replace(config, instruction_attention_bias=0).create(jax.random.key(0))
    params, old = nnx.state(model, nnx.Param).flat_state(), nnx.state(original, nnx.Param).flat_state()
    assert params.keys() == old.keys()
    for key in params:
        np.testing.assert_array_equal(params[key].value, old[key].value)
    obs = dataclasses.replace(
        config.fake_obs(),
        tokenized_prompt=jnp.full((1, config.max_token_len), 10, jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, config.max_token_len), bool),
    )
    noise = config.fake_act()
    expected = noise - model._predict_velocity(obs, noise, jnp.ones(1))
    sampled = model.sample_actions(jax.random.key(1), obs, noise=noise, num_steps=1)
    np.testing.assert_allclose(sampled, expected, rtol=1e-5, atol=2e-5)
    baseline = original.sample_actions(jax.random.key(1), obs, noise=noise, num_steps=1)
    if beta == 0:
        assert model._instruction_bias(obs, 51, 4, 55) is None
        np.testing.assert_array_equal(sampled, baseline)
    else:
        assert not np.allclose(sampled, baseline)


def test_attention_bias_invalid_values_are_rejected():
    for beta in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="instruction_attention_bias"):
            pi0_config.Pi0Config(instruction_attention_bias=beta)
