#!/usr/bin/env python3
"""
Generate samples from a trained MeanFlow checkpoint.

Usage:
    python generate_samples.py --checkpoint checkpoints/checkpoint_epoch_10.pkl --num_samples 16
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from ml_collections import ConfigDict

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.meanflow_net import MeanFlowNet

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigDict(raw)


def load_checkpoint(checkpoint_path):
    """Load checkpoint from pickle file."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)

    print(f"  Epoch: {checkpoint_data.get('epoch', 'unknown')}")
    print(f"  Step: {checkpoint_data.get('step', 'unknown')}")

    return checkpoint_data


def create_model(cfg):
    """Create the MeanFlowNet model."""
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


def sample_images(model, params, rng, cfg, num_samples=16, class_labels=None,
                 num_steps=100, cfg_scale=1.0):
    """
    Generate samples from the trained model.

    Args:
        model: MeanFlowNet model
        params: Model parameters
        rng: JAX random key
        cfg: Configuration
        num_samples: Number of samples to generate
        class_labels: Optional class labels (if None, random)
        num_steps: Number of ODE integration steps
        cfg_scale: Classifier-free guidance scale

    Returns:
        Generated images and their labels
    """

    # Initialize with noise in IMAGE space (32x32)
    rng, noise_rng, cls_rng = jax.random.split(rng, 3)
    zt = jax.random.normal(noise_rng,
                          (num_samples, 32, 32, cfg.model.in_ch))

    # Sample class labels if not provided
    if class_labels is None:
        class_labels = jax.random.randint(cls_rng, (num_samples,),
                                         0, cfg.model.num_classes)
    else:
        class_labels = jnp.array(class_labels)

    # Set r=0 for sampling ODE
    r_batch = jnp.zeros((num_samples,))

    # Sampling loop - integrate from t=1 (noise) to t=0 (data)
    dt = 1.0 / num_steps

    print(f"Generating {num_samples} samples with {num_steps} ODE steps...")
    for step in tqdm(range(num_steps), desc="Sampling"):
        t = 1.0 - step * dt
        t_batch = jnp.full((num_samples,), t)

        # Get velocity prediction with optional CFG
        if cfg_scale != 1.0:
            # Classifier-free guidance
            null_labels = jnp.full_like(class_labels, cfg.model.num_classes)

            # Conditional velocity
            u_cond = model.apply(
                {"params": params},
                zt, r_batch, t_batch, class_labels,
                train_cfg_drop=0.0,
                rng=None
            )

            # Unconditional velocity
            u_uncond = model.apply(
                {"params": params},
                zt, r_batch, t_batch, null_labels,
                train_cfg_drop=0.0,
                rng=None
            )

            # Apply CFG
            velocity = u_uncond + cfg_scale * (u_cond - u_uncond)
        else:
            # No CFG, just conditional
            velocity = model.apply(
                {"params": params},
                zt, r_batch, t_batch, class_labels,
                train_cfg_drop=0.0,
                rng=None
            )

        # ODE integration: dz/dt = -u, so z_{t-dt} = z_t - dt*(-u) = z_t + u*dt
        zt = zt + velocity * dt

    return zt, class_labels


def visualize_samples(samples, labels, save_path=None):
    """Visualize generated samples in a grid."""
    num_samples = min(16, samples.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_samples):
        img = samples[i]
        axes[i].imshow(img)
        axes[i].set_title(f"Generated: {CLASS_NAMES[labels[i]]}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Generated Samples from MeanFlow Model',
                fontsize=14, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate samples from MeanFlow checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pkl)')
    parser.add_argument('--config', type=str,
                       default='../configs/cifar10_few_classes.yaml',
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of ODE integration steps')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                       help='Classifier-free guidance scale')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                       help='Specific classes to generate (e.g., --classes 0 1 2)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (if not specified, will display)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load config
    print("="*60)
    print("MeanFlow Sample Generation")
    print("="*60)

    cfg = load_config(args.config)
    print(f"Config loaded from: {args.config}")

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint)
    params = checkpoint['ema']  # Use EMA parameters for best quality

    # Create model
    print("\nCreating model...")
    model = create_model(cfg)
    print(f"  Model: MeanFlowNet")
    print(f"  Channels: {cfg.model.ch}")
    print(f"  Channel multipliers: {cfg.model.get('ch_mult', [1, 2, 4])}")

    # Prepare class labels
    if args.classes is not None:
        # Repeat specified classes to fill num_samples
        class_labels = jnp.array(args.classes * (args.num_samples // len(args.classes) + 1))
        class_labels = class_labels[:args.num_samples]
        print(f"\nGenerating classes: {args.classes}")
    else:
        class_labels = None
        print(f"\nGenerating random classes")

    # Generate samples
    print(f"\nGeneration settings:")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  ODE steps: {args.num_steps}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Random seed: {args.seed}")
    print()

    rng = jax.random.PRNGKey(args.seed)
    samples, labels = sample_images(
        model, params, rng, cfg,
        num_samples=args.num_samples,
        class_labels=class_labels,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale
    )

    # Convert to numpy and clip
    samples_np = np.array(samples)
    samples_np = np.clip(samples_np, 0, 1)

    print(f"\nGenerated samples shape: {samples_np.shape}")
    print(f"Value range: [{samples_np.min():.3f}, {samples_np.max():.3f}]")

    # Visualize
    print("\nVisualizing samples...")
    visualize_samples(samples_np, labels, save_path=args.output)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
