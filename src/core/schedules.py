import jax.numpy as jnp
import jax

def linear_path(x, eps, t):
    """z_t = (1-t) x + t eps"""
    a_t = 1.0 - t
    b_t = t
    zt = a_t[..., None, None, None] * x + b_t[..., None, None, None] * eps
    v_t = eps - x  # instantaneous velocity under linear path
    return zt, v_t

def sample_r_t(rng: jax.Array, batch: int, use_logitnormal: bool = True,
               mu: float = -0.4, sigma: float = 1.0):
    """
    Sample time pairs (r, t) with 0 <= r < t <= 1.

    Args:
        rng: JAX random key
        batch: Batch size
        use_logitnormal: If True, use LogitNormal distribution (recommended by paper)
                        If False, use Uniform distribution
        mu: Mean for LogitNormal (-0.4 recommended for CIFAR-10)
        sigma: Std for LogitNormal (1.0 recommended)

    Returns:
        r, t: Time pairs with r < t
    """
    rng1, rng2 = jax.random.split(rng)

    if use_logitnormal:
        # LogitNormal(mu, sigma) as recommended in paper Section 4.3
        # Sample from normal, then apply sigmoid to get [0, 1]
        r_logit = jax.random.normal(rng1, (batch,)) * sigma + mu
        t_logit = jax.random.normal(rng2, (batch,)) * sigma + mu

        r = jax.nn.sigmoid(r_logit)
        t = jax.nn.sigmoid(t_logit)
    else:
        # Uniform fallback
        r = jax.random.uniform(rng1, (batch,), minval=0.0, maxval=1.0)
        t = jax.random.uniform(rng2, (batch,), minval=0.0, maxval=1.0)

    # Ensure r < t by sorting
    r, t = jnp.minimum(r, t), jnp.maximum(r, t)

    # Avoid exact equality
    t = jnp.where(t == r, jnp.clip(t + 1e-4, 0.0, 1.0), t)

    return r, t
