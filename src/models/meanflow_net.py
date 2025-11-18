import jax
import jax.numpy as jnp
import flax.linen as nn
from core.blocks import UNet, sinusoidal_embedding

class MeanFlowNet(nn.Module):
    in_ch: int = 3
    latent_hw: int = 32
    ch: int = 64
    num_classes: int = 10
    ch_mult: tuple = (1, 2, 4)
    num_res_blocks: int = 2
    
    @nn.compact
    def __call__(self, x, r, t, cls_idx, train_cfg_drop=0.1, rng=None):
        """
        x: [B, H, W, C] image
        r, t: [B] scalar conditioning values
        cls_idx: [B] class indices
        """
        B = x.shape[0]

        # Classifier-free guidance dropout
        # Always compute, but use dummy rng if not provided
        if rng is None:
            rng = self.make_rng('dropout') if self.is_mutable_collection('dropout') else jax.random.PRNGKey(0)

        drop = jax.random.bernoulli(rng, p=train_cfg_drop, shape=(B,))
        null_idx = self.num_classes  # Last index is unconditional
        cls_idx = jnp.where(drop, null_idx, cls_idx)

        # Sinusoidal embeddings for r and t
        r_emb = sinusoidal_embedding(r, dim=128)
        t_emb = sinusoidal_embedding(t, dim=128)

        # Class embedding (including null class)
        cls_emb = nn.Embed(self.num_classes + 1, 128)(cls_idx)

        # Combine all conditioning
        cond = jnp.concatenate([r_emb, t_emb, cls_emb], axis=-1)  # [B, 384]
        cond = nn.Dense(256)(cond)
        cond = nn.swish(cond)
        cond = nn.Dense(256)(cond)
        
        # Apply UNet
        out = UNet(
            ch=self.ch, 
            ch_mult=self.ch_mult,
            num_res_blocks=self.num_res_blocks
        )(x, cond)
        
        return out