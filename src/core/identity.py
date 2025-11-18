from typing import Callable
import jax
import jax.numpy as jnp

def meanflow_target(u_apply_fn: Callable, params, zt, r, t, cls_idx, v_t, rng=None):
    """
    u_apply_fn: function(params, zt, r, t, cls_idx, rng) -> u
    params: model parameters
    zt: [B,H,W,C]
    r,t: [B]
    cls_idx: [B]
    v_t: instantaneous velocity = eps - x, same shape as zt
    Returns:
      u_pred, u_star (target)
    """
    # Pack inputs so we can take JVP with respect to (zt, t)
    def u_wrapped(zt, t):
        return u_apply_fn(params, zt, r, t, cls_idx, rng)

    # Tangent (v_t for z, and 1 for t)
    # We need to pass matching pytree tangents
    primals = (zt, t)
    tangents = (v_t, jnp.ones_like(t))

    u_pred, du_total = jax.jvp(u_wrapped, primals, tangents)  # total derivative d/dt u = v·∂_z u + ∂_t u

    # MeanFlow identity: u = v - (t - r) * d/dt u
    t_minus_r = (t - r).reshape(-1, 1, 1, 1)
    u_star = v_t - t_minus_r * du_total
    return u_pred, u_star
