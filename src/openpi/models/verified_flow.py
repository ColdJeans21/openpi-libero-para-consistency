"""R0/R1/R2 objectives. MSE uses all padded dimensions, matching historical FM.

Fixed mixture-frequency weighting is intentional: do not divide by the number
of negative examples in a batch (it varies, and is sometimes zero).
"""

import jax
import jax.numpy as jnp


def verified_flow_losses(v, v_alt, u, u_alt, endpoint, endpoint_alt, negative, teacher_weight, diff_weight):
    fm0 = jnp.mean(jnp.square(v - u), axis=-1)
    fm1 = jnp.mean(jnp.square(v_alt - u_alt), axis=-1)
    mask = negative[..., None]
    teacher = jnp.where(mask, 0.0, jnp.mean(jnp.square(v_alt - jax.lax.stop_gradient(v)), axis=-1))
    # u_alt-u = a-a_alt, since BOTH labels use the same epsilon.
    target = jax.lax.stop_gradient(endpoint + u_alt - u)
    differential = jnp.where(mask, jnp.mean(jnp.square(endpoint_alt - target), axis=-1), 0.0)
    fm = 0.5 * (fm0 + fm1)
    total = fm + teacher_weight * teacher + diff_weight * differential
    return total, {
        "loss_fm_original": fm0,
        "loss_fm_alt": fm1,
        "loss_fm": fm,
        "loss_teacher_positive": teacher,
        "loss_differential_negative": differential,
        "loss_total": total,
        "negative_pair_fraction": jnp.mean(negative.astype(jnp.float32)),
    }
