# ruff: noqa: SLF001
import dataclasses

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import config as training_config
from openpi.training import weight_loaders


class TinyVision(nn.Module):
    @nn.compact
    def __call__(self, image, *, train=False):
        del train
        return nn.Dense(64)(jnp.mean(image, axis=(1, 2)))[:, None], None


@pytest.fixture
def semantic_model(monkeypatch):
    monkeypatch.setattr(_pi0._siglip, "Module", lambda **kwargs: TinyVision())
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=7,
        action_horizon=3,
        semantic_rank=3,
        semantic_feature_dim=8,
        paired_prompt_training=True,
        paired_loss_mode="teacher_consistency",
        consistency_loss_weight=1.0,
    )
    model = config.create(jax.random.key(0))
    obs = dataclasses.replace(
        config.fake_obs(batch_size=1),
        semantic_features=jax.random.normal(jax.random.key(1), (1, 8)),
        semantic_features_alt=jax.random.normal(jax.random.key(2), (1, 8)),
    )
    return config, model, obs


def activate_basis(model):
    model.semantic_basis_head.kernel.value = (
        jax.random.normal(jax.random.key(3), model.semantic_basis_head.kernel.shape) * 0.03
    )


def test_raw_text_ids_and_masks_cannot_change_trunk(semantic_model):
    config, model, obs = semantic_model
    altered = dataclasses.replace(
        obs,
        tokenized_prompt=jnp.full((1, 9), 500),
        tokenized_prompt_mask=jnp.zeros((1, 9), bool),
        tokenized_prompt_alt=None,
        tokenized_prompt_alt_mask=None,
    )
    noise = config.fake_act()
    hidden = model._predict_action_hidden(obs, noise, jnp.array([0.5]))
    hidden_alt = model._predict_action_hidden(altered, noise, jnp.array([0.5]))
    np.testing.assert_array_equal(hidden, hidden_alt)


def test_zero_initialization_and_semantic_edit_identity(semantic_model):
    _, model, obs = semantic_model
    hidden = jax.random.normal(jax.random.key(4), (1, 3, 64))
    np.testing.assert_array_equal(
        model._velocity_from_hidden(hidden, obs.semantic_features), model.action_out_proj(hidden)
    )
    activate_basis(model)
    base, basis = model._velocity_components(hidden)
    v1 = model._velocity_from_hidden(hidden, obs.semantic_features)
    v2 = model._velocity_from_hidden(hidden, obs.semantic_features_alt)
    expected = jnp.einsum(
        "bhdr,br->bhd",
        basis,
        model._semantic_code(obs.semantic_features_alt) - model._semantic_code(obs.semantic_features),
    )
    np.testing.assert_allclose(v2 - v1, expected, atol=2e-6)
    np.testing.assert_allclose(
        v1 - base, jnp.einsum("bhdr,br->bhd", basis, model._semantic_code(obs.semantic_features)), atol=1e-6
    )


def test_loss_matches_shared_flow_and_identical_prompts(semantic_model):
    config, model, obs = semantic_model
    activate_basis(model)
    actions = config.fake_act()
    rng = jax.random.key(5)
    pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    preprocessed = _model.preprocess_observation(pre_rng, obs, train=False)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, (1,)) * 0.999 + 0.001
    x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    v = model._predict_velocity(preprocessed, x, time)
    vp = model._predict_velocity(
        dataclasses.replace(preprocessed, semantic_features=obs.semantic_features_alt), x, time
    )
    expected, _ = _pi0.paired_flow_matching_losses(v, vp, noise - actions, 1, "teacher_consistency")
    actual, metrics = model.compute_loss_with_metrics(rng, obs, actions)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    assert metrics["loss_consistency"].mean() > 0
    identical = dataclasses.replace(obs, semantic_features_alt=obs.semantic_features)
    _, metrics = model.compute_loss_with_metrics(rng, identical, actions)
    np.testing.assert_array_equal(metrics["loss_consistency"], jnp.zeros((1, 3)))


def test_inference_uses_semantics_and_matches_manual_euler(semantic_model):
    config, model, obs = semantic_model
    activate_basis(model)
    noise = config.fake_act()
    expected = noise - model._predict_velocity(obs, noise, jnp.ones((1,)))
    actual = model.sample_actions(jax.random.key(6), obs, noise=noise, num_steps=1)
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    altered = dataclasses.replace(obs, semantic_features=obs.semantic_features_alt)
    other = model.sample_actions(jax.random.key(6), altered, noise=noise, num_steps=1)
    assert not np.allclose(actual, other)


def test_semantic_heads_receive_gradients(semantic_model):
    _, model, obs = semantic_model
    hidden = jax.random.normal(jax.random.key(7), (1, 3, 64))
    diff = nnx.DiffState(0, nnx_utils.PathRegex("semantic_.*"))

    def objective(m):
        return jnp.mean((m._velocity_from_hidden(hidden, obs.semantic_features) - 1) ** 2)

    _, initial_grad = nnx.value_and_grad(objective, argnums=diff)(model)
    assert jnp.linalg.norm(initial_grad["semantic_basis_head"]["kernel"].value) > 0
    np.testing.assert_array_equal(initial_grad["semantic_language_proj"]["kernel"].value, jnp.zeros((8, 3)))
    activate_basis(model)
    _, gradients = nnx.value_and_grad(objective, argnums=diff)(model)
    assert jnp.linalg.norm(gradients["semantic_language_proj"]["kernel"].value) > 0


def test_fm_control_does_not_supervise_paraphrase():
    v, vp, target = jnp.ones((1, 2, 3)), jnp.full((1, 2, 3), 8.0), jnp.zeros((1, 2, 3))
    loss, _ = _pi0.paired_flow_matching_losses(v, vp, target, 0, "canonical_fm")
    np.testing.assert_array_equal(loss, jnp.ones((1, 2)))
    derivative = jax.grad(lambda alt: _pi0.paired_flow_matching_losses(v, alt, target, 0, "canonical_fm")[0].sum())(vp)
    np.testing.assert_array_equal(derivative, jnp.zeros_like(vp))


def test_configs_have_matched_scope_and_initialization():
    fm = training_config.get_config("pi0_libero_goal_semantic_fm")
    teacher = training_config.get_config("pi0_libero_goal_semantic_teacher")
    assert (
        dataclasses.replace(fm.model, paired_loss_mode="teacher_consistency", consistency_loss_weight=1)
        == teacher.model
    )
    assert fm.seed == teacher.seed == 42
    assert fm.batch_size == teacher.batch_size == 4
    shape = nnx.eval_shape(fm.model.create, jax.random.key(0))
    trainable = nnx.state(shape, fm.trainable_filter).flat_state()
    assert sum(p.value.size for p in trainable.values()) == 468_580_640
    assert any(path[:2] == ("PaliGemma", "img") for path in trainable)
    assert any(path[0] == "semantic_basis_head" for path in trainable)


def test_weight_loader_only_allows_new_heads_and_lora():
    ref = {"backbone": {"weight": np.ones(2)}, "semantic_basis_head": {"kernel": np.zeros(3)}}
    loaded = {"backbone": {"weight": np.full(2, 2)}}
    merged = weight_loaders._merge_params(loaded, ref, missing_regex="semantic_basis_head/.*")
    np.testing.assert_array_equal(merged["semantic_basis_head"]["kernel"], np.zeros(3))
    assert "weight" not in weight_loaders._merge_params({}, ref, missing_regex="semantic_basis_head/.*").get(
        "backbone", {}
    )
