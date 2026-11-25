import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.verified_flow import verified_flow_losses


def test_correct_targets_zero_and_wrong_shared_offset_is_anchored():
    u = jnp.ones((2, 3, 7))
    alt = u.at[1].set(3.0)
    negative = jnp.array([False, True])
    loss, _ = verified_flow_losses(u, alt, u, alt, u, alt, negative, 1.0, 0.1)
    np.testing.assert_allclose(loss, 0.0)
    loss, metrics = verified_flow_losses(u + 2, alt + 2, u, alt, u + 2, alt + 2, negative, 1.0, 0.1)
    np.testing.assert_allclose(metrics["loss_differential_negative"], 0.0)
    np.testing.assert_allclose(loss, 4.0)


def test_auxiliary_gradients_stop_on_anchor_and_mask_other_edge():
    x = jnp.ones((2, 3, 7))
    negative = jnp.array([False, True])

    def auxiliary(v, other, ep, ep_other):
        _, m = verified_flow_losses(v, other, x, 2 * x, ep, ep_other, negative, 1.0, 0.1)
        return jnp.mean(m["loss_teacher_positive"] + m["loss_differential_negative"])

    g0, g1, g2, g3 = jax.grad(auxiliary, argnums=(0, 1, 2, 3))(x, 3 * x, x, 4 * x)
    np.testing.assert_allclose(g0, 0.0)
    np.testing.assert_allclose(g2, 0.0)
    np.testing.assert_allclose(g1[1], 0.0)
    np.testing.assert_allclose(g3[0], 0.0)
    assert jnp.linalg.norm(g1[0]) > 0
    assert jnp.linalg.norm(g3[1]) > 0


def test_endpoint_sign_and_coordinate_identity():
    a, b, eps = jnp.ones((1, 2, 7)), jnp.full((1, 2, 7), 3.0), jnp.full((1, 2, 7), 0.5)
    u, ub = eps - a, eps - b
    np.testing.assert_allclose(ub - u, a - b)
    # Only tau=1 removes the action label from x_tau.
    assert not jnp.allclose(0.5 * eps + 0.5 * a, 0.5 * eps + 0.5 * b)
    loss, _ = verified_flow_losses(u, ub, u, ub, u, ub, jnp.array([True]), 1.0, 0.1)
    np.testing.assert_allclose(loss, 0.0)


def test_model_forward_uses_two_paths_for_fm_and_same_endpoint_for_diff(monkeypatch):
    from openpi.models import model as observation_model
    from openpi.models.pi0 import Pi0
    from openpi.models.pi0_config import Pi0Config

    c = Pi0Config(action_horizon=2, paired_prompt_training=True, verified_pair_training=True)
    obs = c.fake_obs(2)
    actions = jnp.ones((2, 2, 32))
    alt = actions.at[1].set(3.0)
    obs = obs.replace(paired_actions=alt, pair_is_negative=jnp.array([False, True]))
    monkeypatch.setattr(observation_model, "preprocess_observation", lambda _rng, o, **_kw: o)

    class Probe:
        verified_pair_training = True
        differential_loss_weight = 0.1
        consistency_loss_weight = 1.0
        semantic_rank = 0
        _make_paired_observation = Pi0._make_paired_observation  # noqa: SLF001

        def __init__(self):
            self.calls = []

        def _predict_velocity(self, paired, x, time):
            self.calls.append((paired, x, time))
            return jnp.zeros_like(x)

    probe = Probe()
    loss, _ = Pi0.compute_loss_with_metrics(probe, jax.random.key(12), obs, actions)
    assert loss.shape == (2, 2)
    assert len(probe.calls) == 2
    _, x, t = probe.calls[0]
    np.testing.assert_allclose(x[:2] - x[2:], (1 - t[:2, None, None]) * (actions - alt), atol=1e-6)
    _, endpoint, t1 = probe.calls[1]
    np.testing.assert_array_equal(endpoint[:2], endpoint[2:])
    np.testing.assert_array_equal(t1, 1.0)
    assert probe.calls[0][0].state.shape[0] == 4
