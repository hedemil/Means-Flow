import jax
import jax.numpy as jnp

def sample_1nfe(rng, apply_fn, params, shape, num_classes, cfg_scale=2.0):
    """
    Start z1 ~ N(0,I); one step using u_theta(z1, r=0, t=1, c, cfg_scale).
    For CIFAR-10 we sample unconditional or class-conditional.
    """
    rng, rgneps, rngcls = jax.random.split(rng, 3)
    z1 = jax.random.normal(rgneps, shape)

    # choose classes uniformly
    cls = jax.random.randint(rngcls, (shape[0],), 0, num_classes)
    r = jnp.zeros((shape[0],))
    t = jnp.ones((shape[0],))

    # CFG-as-field: we trained with class-drop; here we can emulate scaling
    # Simple trick: run once with class, once with null, then combine in-field.
    # Since our network expects cfg inside, a pragmatic approach here is:
    # call once with cls, once with null and linearly mix outputs.
    null = jnp.full_like(cls, fill_value=num_classes)  # null index

    u_c   = apply_fn(params, z1, r, t, cls, rng=None)
    u_null= apply_fn(params, z1, r, t, null, rng=None)

    u = u_null + cfg_scale * (u_c - u_null)

    z0 = z1 - u  # interval length is 1
    return z0
