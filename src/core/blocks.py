from typing import Callable, Optional, Tuple
import jax.numpy as jnp
import jax
import flax.linen as nn

def sinusoidal_embedding(t, dim=128):
    """Sinusoidal time embedding.

    Args:
        t: Time values, shape [B] or scalar
        dim: Embedding dimension

    Returns:
        Embeddings of shape [B, dim]
    """
    # Ensure t is at least 1D
    t = jnp.atleast_1d(t)

    half = dim // 2
    freqs = jnp.exp(jnp.arange(half) * (-jnp.log(10000.0) / half))
    args = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)

class ResBlock(nn.Module):
    """Residual block with time conditioning."""
    out_ch: int
    
    @nn.compact
    def __call__(self, x, temb):
        h = nn.GroupNorm(num_groups=8)(x)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, (3, 3), padding='SAME')(h)
        
        # Time conditioning
        temb_proj = nn.Dense(self.out_ch)(nn.swish(temb))
        h = h + temb_proj[:, None, None, :]
        
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, (3, 3), padding='SAME')(h)
        
        # Residual connection
        if x.shape[-1] != self.out_ch:
            x = nn.Conv(self.out_ch, (1, 1))(x)
        
        return x + h

class Downsample(nn.Module):
    out_ch: int
    
    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.out_ch, (3, 3), strides=(2, 2), padding='SAME')(x)

class Upsample(nn.Module):
    out_ch: int
    
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        # Nearest neighbor upsampling
        x = jax.image.resize(x, (B, H * 2, W * 2, C), method='nearest')
        x = nn.Conv(self.out_ch, (3, 3), padding='SAME')(x)
        return x

class UNet(nn.Module):
    """UNet for MeanFlow with proper time conditioning."""
    ch: int = 64
    ch_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    
    @nn.compact
    def __call__(self, x, cond):
        """
        x: [B, H, W, C] input image
        cond: [B, D] conditioning vector (combined r, t, class embeddings)
        """
        # Project conditioning to higher dimension
        temb = nn.Dense(self.ch * 4)(cond)
        temb = nn.swish(temb)
        temb = nn.Dense(self.ch * 4)(temb)
        
        # Initial convolution
        h = nn.Conv(self.ch, (3, 3), padding='SAME')(x)
        
        # Encoder
        skip_connections = [h]
        
        for i, mult in enumerate(self.ch_mult):
            out_ch = self.ch * mult
            
            # Residual blocks at this resolution
            for _ in range(self.num_res_blocks):
                h = ResBlock(out_ch)(h, temb)
                skip_connections.append(h)
            
            # Downsample (except at the last level)
            if i < len(self.ch_mult) - 1:
                h = Downsample(out_ch)(h)
                skip_connections.append(h)
        
        # Middle
        h = ResBlock(self.ch * self.ch_mult[-1])(h, temb)
        h = ResBlock(self.ch * self.ch_mult[-1])(h, temb)
        
        # Decoder
        for i, mult in enumerate(reversed(self.ch_mult)):
            out_ch = self.ch * mult
            
            for _ in range(self.num_res_blocks + 1):
                # Concatenate skip connection
                skip = skip_connections.pop()
                h = jnp.concatenate([h, skip], axis=-1)
                h = ResBlock(out_ch)(h, temb)
            
            # Upsample (except at the last level)
            if i < len(self.ch_mult) - 1:
                h = Upsample(out_ch)(h)
        
        # Output
        h = nn.GroupNorm(num_groups=8)(h)
        h = nn.swish(h)
        # Use default initialization - small init was preventing the model from learning
        h = nn.Conv(x.shape[-1], (3, 3), padding='SAME')(h)
        
        return h