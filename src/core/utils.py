from dataclasses import dataclass
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

@dataclass
class TrainConfig:
    lr: float
    wd: float
    ema: float
    grad_clip: float

def make_tx(cfg: TrainConfig):
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(cfg.lr, weight_decay=cfg.wd),
    )
    return tx

class EMA:
    def __init__(self, decay):
        self.decay = decay
    def init(self, params):
        return params
    def update(self, ema_params, new_params):
        return jax.tree_map(lambda e, p: self.decay*e + (1-self.decay)*p, ema_params, new_params)
