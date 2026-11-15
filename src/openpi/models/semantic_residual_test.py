# ruff: noqa: SLF001
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.semantic_interface_test import TinyVision
from openpi.models.semantic_interface_test import activate_basis
from openpi.shared import nnx_utils
from openpi.training import config as training_config


@pytest.fixture
def residual_models(monkeypatch):
    monkeypatch.setattr(pi0._siglip, "Module", lambda **kwargs: TinyVision())
    original = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=7,
        action_horizon=3,
        paired_prompt_training=True,
        paired_loss_mode="teacher_consistency",
        consistency_loss_weight=1,
    )
    config = dataclasses.replace(original, semantic_rank=3, semantic_feature_dim=8, semantic_preserve_prompt=True)
    base = original.create(jax.random.key(0))
    model = config.create(jax.random.key(0))
    obs = dataclasses.replace(
        config.fake_obs(batch_size=1),
        tokenized_prompt=jnp.full((1, config.max_token_len), 10, dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=bool),
        tokenized_prompt_alt=jnp.full((1, config.max_token_len), 30, dtype=jnp.int32),
        tokenized_prompt_alt_mask=jnp.ones((1, config.max_token_len), dtype=bool),
        semantic_features=jax.random.normal(jax.random.key(1), (1, 8)),
        semantic_features_alt=jax.random.normal(jax.random.key(2), (1, 8)),
    )
    return config, base, model, obs


def test_zero_residual_preserves_original_parameters_loss_and_euler(residual_models):
    config, base, model, obs = residual_models
    original_params = nnx.state(base, nnx.Param).flat_state()
    new_params = nnx.state(model, nnx.Param).flat_state()
    for key, parameter in original_params.items():
        np.testing.assert_array_equal(parameter.value, new_params[key].value)
    actions = config.fake_act()
    key = jax.random.key(3)
    old_loss, _ = base.compute_loss_with_metrics(key, obs, actions)
    new_loss, _ = model.compute_loss_with_metrics(key, obs, actions)
    np.testing.assert_array_equal(old_loss, new_loss)
    np.testing.assert_array_equal(
        base.sample_actions(key, obs, noise=actions, num_steps=2),
        model.sample_actions(key, obs, noise=actions, num_steps=2),
    )


def test_raw_instruction_still_conditions_original_trunk(residual_models):
    config, base, model, obs = residual_models
    alt = dataclasses.replace(obs, tokenized_prompt=obs.tokenized_prompt_alt)
    x, time = config.fake_act(), jnp.array([0.5])
    hidden = model._predict_action_hidden(obs, x, time)
    other = model._predict_action_hidden(alt, x, time)
    assert not np.allclose(hidden, other)
    np.testing.assert_array_equal(hidden, base._predict_action_hidden(obs, x, time))


def test_two_language_branches_share_augmented_images_noise_and_time(residual_models, monkeypatch):
    config, _, model, obs = residual_models
    activate_basis(model)
    key = jax.random.key(5)
    pre_key, noise_key, time_key = jax.random.split(key, 3)
    actions = config.fake_act()
    noise = jax.random.normal(noise_key, actions.shape)
    time = jax.random.beta(time_key, 1.5, 1, (1,)) * 0.999 + 0.001
    x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    processed = model_lib.preprocess_observation(pre_key, obs, train=True)
    paired = model._make_paired_observation(processed)
    np.testing.assert_array_equal(paired.semantic_features[:1], obs.semantic_features)
    np.testing.assert_array_equal(paired.semantic_features[1:], obs.semantic_features_alt)
    alt = dataclasses.replace(
        processed,
        tokenized_prompt=processed.tokenized_prompt_alt,
        tokenized_prompt_mask=processed.tokenized_prompt_alt_mask,
        semantic_features=processed.semantic_features_alt,
    )
    expected, _ = pi0.paired_flow_matching_losses(
        model._predict_velocity(processed, x, time),
        model._predict_velocity(alt, x, time),
        noise - actions,
        1,
        "teacher_consistency",
    )
    calls = []
    original = pi0.Pi0._predict_action_hidden

    def capture(self, observation, x_t, t):
        calls.append((observation, x_t, t))
        return original(self, observation, x_t, t)

    monkeypatch.setattr(pi0.Pi0, "_predict_action_hidden", capture)
    actual, _ = model.compute_loss_with_metrics(key, obs, actions, train=True)
    # Separate B and combined 2B matmuls can differ by a few float32 ULPs.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
    assert len(calls) == 1
    actual_obs, actual_x, actual_t = calls[0]
    assert actual_x.shape[0] == 2
    np.testing.assert_array_equal(actual_x[:1], actual_x[1:])
    np.testing.assert_array_equal(actual_t[:1], actual_t[1:])
    for image in actual_obs.images.values():
        np.testing.assert_array_equal(image[:1], image[1:])
    assert not np.array_equal(actual_obs.tokenized_prompt[:1], actual_obs.tokenized_prompt[1:])


def test_switch_disables_trained_residual_but_preserves_language(residual_models):
    config, base, model, obs = residual_models
    activate_basis(model)
    noise = config.fake_act()
    key = jax.random.key(6)
    enabled = model.sample_actions(key, obs, noise=noise, num_steps=1)
    model.semantic_residual_scale = 0.0
    disabled = model.sample_actions(key, obs, noise=noise, num_steps=1)
    np.testing.assert_array_equal(disabled, base.sample_actions(key, obs, noise=noise, num_steps=1))
    assert not np.allclose(enabled, disabled)
    changed_features = dataclasses.replace(obs, semantic_features=obs.semantic_features_alt)
    np.testing.assert_array_equal(disabled, model.sample_actions(key, changed_features, noise=noise, num_steps=1))
    changed_text = dataclasses.replace(obs, tokenized_prompt=obs.tokenized_prompt_alt)
    assert not np.allclose(disabled, model.sample_actions(key, changed_text, noise=noise, num_steps=1))


def test_total_teacher_loss_trains_residual_heads(residual_models):
    config, _, model, obs = residual_models
    diff = nnx.DiffState(0, nnx_utils.PathRegex("semantic_.*"))

    def objective(m):
        return m.compute_loss(jax.random.key(7), obs, config.fake_act()).mean()

    gradients = nnx.grad(objective, argnums=diff)(model)
    assert jnp.linalg.norm(gradients["semantic_basis_head"]["kernel"].value) > 0
    np.testing.assert_array_equal(gradients["semantic_language_proj"]["kernel"].value, jnp.zeros((8, 3)))
    activate_basis(model)
    gradients = nnx.grad(objective, argnums=diff)(model)
    assert jnp.linalg.norm(gradients["semantic_language_proj"]["kernel"].value) > 0


def test_control_and_residual_have_matched_training_scope():
    control = training_config.get_config("pi0_libero_goal_teacher_prefix_control")
    residual = training_config.get_config("pi0_libero_goal_teacher_semantic_residual")
    off = training_config.get_config("pi0_libero_goal_teacher_semantic_residual_off")
    assert control.model == dataclasses.replace(residual.model, semantic_rank=0, semantic_preserve_prompt=False)
    assert off.model == dataclasses.replace(residual.model, semantic_residual_scale=0)
    assert control.seed == residual.seed == 42
    assert control.batch_size == residual.batch_size == 4
    assert control.num_train_steps == residual.num_train_steps == 30_000
    assert control.data.base_config.task_filter == residual.data.base_config.task_filter
    assert control.data.base_config.paired_prompt_path == residual.data.base_config.paired_prompt_path
    counts = []
    for cfg in (control, residual):
        shape = nnx.eval_shape(cfg.model.create, jax.random.key(0))
        params = nnx.state(shape, cfg.trainable_filter).flat_state()
        assert any(path[:2] == ("PaliGemma", "img") for path in params)
        counts.append(sum(p.value.size for p in params.values()))
    assert counts == [468_039_440, 468_580_640]
