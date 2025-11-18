import jax.numpy as jnp
import jax

def linear_path(x, eps, t):
    """z_t = (1-t) x + t eps"""
    a_t = 1.0 - t
    b_t = t
    zt = a_t[..., None, None, None] * x + b_t[..., None, None, None] * eps
    v_t = eps - x  # instantaneous velocity under linear path
    return zt, v_t

def sample_r_t(rng: jax.Array, batch: int):
    """Uniform 0 <= r < t <= 1."""
    rng1, rng2 = jax.random.split(rng)
    r = jax.random.uniform(rng1, (batch,), minval=0.0, maxval=1.0)
    t = jax.random.uniform(rng2, (batch,), minval=0.0, maxval=1.0)
    r, t = jnp.minimum(r, t), jnp.maximum(r, t)
    # avoid exact equality
    t = jnp.where(t == r, jnp.clip(t + 1e-4, 0.0, 1.0), t)
    return r, t
