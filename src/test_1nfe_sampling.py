#!/usr/bin/env python3
"""
Test the correct 1-NFE sampling method from core/sample.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import yaml
from ml_collections import ConfigDict

from models.meanflow_net import MeanFlowNet
from core.sample import sample_1nfe

# Load config
def load_config(config_path):
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigDict(raw)

# Load checkpoint
def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    print(f"  Epoch: {checkpoint_data.get('epoch', 'unknown')}")
    print(f"  Step: {checkpoint_data.get('step', 'unknown')}")
    return checkpoint_data

# Create model
def create_model(cfg):
    ch_mult = tuple(cfg.model.get('ch_mult', [1, 2, 4]))
    num_res_blocks = cfg.model.get('num_res_blocks', 2)

    model = MeanFlowNet(
        in_ch=cfg.model.in_ch,
        latent_hw=cfg.model.latent_hw,
        ch=cfg.model.ch,
        num_classes=cfg.model.num_classes,
        ch_mult=ch_mult,
        num_res_blocks=num_res_blocks
    )
    return model

def main():
    print("="*60)
    print("Testing 1-NFE Sampling (Correct MeanFlow Method)")
    print("="*60)

    # Paths
    config_path = "../configs/cifar10_test_improved.yaml"
    checkpoint_path = "../checkpoints/checkpoint_epoch_8.pkl"

    # Load
    cfg = load_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint['ema']

    # Create model
    model = create_model(cfg)

    # Create apply_fn that matches what sample_1nfe expects
    def apply_fn(params, z, r, t, cls_idx, rng):
        return model.apply(
            {"params": params},
            z, r, t, cls_idx,
            train_cfg_drop=0.0,
            rng=rng
        )

    # Generate samples using 1-NFE
    print("\nGenerating 16 samples using 1-NFE (single forward pass)...")
    rng = jax.random.PRNGKey(42)

    samples = sample_1nfe(
        rng=rng,
        apply_fn=apply_fn,
        params=params,
        shape=(16, 32, 32, 3),
        num_classes=cfg.model.num_classes,
        cfg_scale=2.0
    )

    # Convert to numpy and clip
    samples_np = np.array(samples)
    samples_np = np.clip(samples_np, 0, 1)

    print(f"\nGenerated samples shape: {samples_np.shape}")
    print(f"Value range: [{samples_np.min():.3f}, {samples_np.max():.3f}]")

    # Visualize
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(16):
        axes[i].imshow(samples_np[i])
        axes[i].axis('off')

    plt.suptitle('1-NFE Sampling (Correct MeanFlow Method)', fontsize=14)
    plt.tight_layout()

    output_path = "../checkpoints/test_1nfe_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved samples to: {output_path}")

    print("\n" + "="*60)
    print("Done! Check the samples to see if they look better.")
    print("="*60)

if __name__ == "__main__":
    main()
