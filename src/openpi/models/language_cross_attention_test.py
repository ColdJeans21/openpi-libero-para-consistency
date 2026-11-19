# ruff: noqa: SLF001
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.semantic_interface_test import TinyVision
from openpi.training import config as training_config


@pytest.fixture(params=["fixed", "dynamic"])
def language_models(monkeypatch, request):
    monkeypatch.setattr(pi0._siglip, "Module", lambda **kwargs: TinyVision())
    common = {
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "dtype": "float32",
        "action_dim": 7,
        "action_horizon": 3,
    }
    baseline = pi0_config.Pi0Config(**common).create(jax.random.key(0))
    config = pi0_config.Pi0Config(
        **common,
        language_cross_attention=True,
        language_cross_attention_gate=request.param,
    )
    branch = config.create(jax.random.key(0))
    obs = dataclasses.replace(
        config.fake_obs(batch_size=1),
        tokenized_prompt=jnp.full((1, config.max_token_len), 10, dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=bool),
    )
    return config, baseline, branch, obs


def test_language_branch_is_zero_initialized_identity(language_models):
    config, baseline, branch, obs = language_models
    extra = {
        path: value.value
        for path, value in nnx.state(branch, nnx.Param).flat_state().items()
        if "language" in str(path)
    }
    assert extra
    projection = [value for path, value in extra.items() if "language_attn_vec_einsum" in str(path)]
    assert projection
    assert all(np.count_nonzero(value) == 0 for value in projection)

    actions = config.fake_act(batch_size=1)
    time = jnp.array([0.5])
    np.testing.assert_array_equal(
        baseline._predict_velocity(obs, actions, time),
        branch._predict_velocity(obs, actions, time),
    )


def test_language_branch_receives_gradient(language_models):
    config, _, branch, obs = language_models
    actions = config.fake_act(batch_size=1)

    def objective(model):
        return jnp.mean(model._predict_velocity(obs, actions, jnp.array([0.5])))

    grads = nnx.grad(objective)(branch)
    projection_grads = [
        value.value for path, value in grads.flat_state().items() if "language_attn_vec_einsum" in str(path)
    ]
    assert projection_grads
    assert any(np.linalg.norm(value) > 0 for value in projection_grads)


def test_instruction_mask_selects_only_action_queries_and_content_keys():
    tokens = jnp.array([[2, 10, 108, 0]], dtype=jnp.int32)
    token_mask = jnp.array([[True, True, True, False]])
    mask = pi0.make_instruction_attention_mask(
        tokens,
        token_mask,
        prefix_length=7,
        query_length=10,
        key_length=10,
        action_horizon=3,
    )
    assert mask.shape == (1, 10, 10)
    assert not mask[:, :7].any()
    assert mask[:, 7:, 4].all()
    assert mask.sum() == 3


def test_paired_fm_language_config_is_e02_matched_single_variable_ablation():
    baseline = training_config.get_config("pi0_libero_goal_lora_para_aug")
    branch = training_config.get_config("pi0_libero_goal_paired_fm_language_fixed")

    assert branch.data == baseline.data
    assert branch.num_train_steps == baseline.num_train_steps
    assert branch.batch_size == baseline.batch_size
    assert branch.fsdp_devices == baseline.fsdp_devices
    assert branch.ema_decay == baseline.ema_decay
    assert branch.model == dataclasses.replace(
        baseline.model,
        language_cross_attention=True,
        language_cross_attention_gate="fixed",
    )
    assert branch.policy_metadata["matched_control"] == baseline.name
