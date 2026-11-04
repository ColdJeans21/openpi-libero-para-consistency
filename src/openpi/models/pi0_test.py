import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models.pi0 import paired_flow_matching_losses
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_paired_flow_matching_losses_share_one_target():
    target = jnp.zeros((2, 3, 4))
    velocity = jnp.ones_like(target)
    velocity_alt = jnp.full_like(target, 2.0)

    total, metrics = paired_flow_matching_losses(velocity, velocity_alt, target, 0.1)

    assert jnp.allclose(metrics["loss_fm_original"], 1.0)
    assert jnp.allclose(metrics["loss_fm_paraphrase"], 4.0)
    assert jnp.allclose(metrics["loss_consistency"], 1.0)
    assert jnp.allclose(total, 2.6)


def test_paired_flow_matching_consistency_is_zero_for_equal_velocities():
    target = jnp.zeros((1, 2, 3))
    velocity = jnp.ones_like(target)

    total, metrics = paired_flow_matching_losses(velocity, velocity, target, 0.5)

    assert jnp.allclose(metrics["loss_consistency"], 0.0)
    assert jnp.allclose(total, metrics["loss_fm_original"])


def test_consistency_loss_produces_gradient():
    target = jnp.zeros((1, 2, 3))
    velocity_alt = jnp.zeros_like(target)

    def consistency_objective(velocity):
        _, metrics = paired_flow_matching_losses(velocity, velocity_alt, target, 1.0)
        return jnp.mean(metrics["loss_consistency"])

    gradient = jax.grad(consistency_objective)(jnp.ones_like(target))
    assert jnp.linalg.norm(gradient) > 0


def test_teacher_consistency_anchors_only_canonical_branch_with_flow_matching():
    target = jnp.zeros((1, 2, 3))
    velocity = jnp.ones_like(target)
    velocity_alt = jnp.full_like(target, 2.0)

    total, metrics = paired_flow_matching_losses(
        velocity,
        velocity_alt,
        target,
        0.1,
        loss_mode="teacher_consistency",
    )

    assert jnp.allclose(metrics["loss_fm_original"], 1.0)
    assert jnp.allclose(metrics["loss_fm_paraphrase"], 4.0)
    assert jnp.allclose(metrics["loss_consistency"], 1.0)
    assert jnp.allclose(total, 1.1)


def test_teacher_consistency_stops_consistency_gradient_into_canonical_branch():
    target = jnp.zeros((1, 2, 3))
    velocity = jnp.ones_like(target)
    velocity_alt = jnp.full_like(target, 2.0)

    def objective(canonical, paraphrase):
        total, _ = paired_flow_matching_losses(
            canonical,
            paraphrase,
            target,
            1.0,
            loss_mode="teacher_consistency",
        )
        return jnp.mean(total)

    canonical_grad, paraphrase_grad = jax.grad(objective, argnums=(0, 1))(velocity, velocity_alt)
    fm_grad = jax.grad(lambda canonical: jnp.mean(jnp.square(canonical - target)))(velocity)

    assert jnp.allclose(canonical_grad, fm_grad)
    assert jnp.linalg.norm(paraphrase_grad) > 0


def test_teacher_consistency_config_requires_paired_prompts_and_positive_weight():
    with pytest.raises(ValueError, match="paired_prompt_training"):
        _pi0_config.Pi0Config(paired_loss_mode="teacher_consistency", consistency_loss_weight=1.0)
    with pytest.raises(ValueError, match="positive consistency_loss_weight"):
        _pi0_config.Pi0Config(paired_prompt_training=True, paired_loss_mode="teacher_consistency")


def test_paired_prompt_config_exposes_second_prompt_spec():
    config = _pi0_config.Pi0Config(paired_prompt_training=True, consistency_loss_weight=0.1)
    observation, _ = config.inputs_spec(batch_size=2)

    assert observation.tokenized_prompt_alt.shape == (2, config.max_token_len)
    assert observation.tokenized_prompt_alt_mask.shape == (2, config.max_token_len)
