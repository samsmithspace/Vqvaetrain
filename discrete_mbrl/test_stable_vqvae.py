#!/usr/bin/env python3
"""
Diagnostic VQ-VAE Test - Analyze Reconstruction Issues
This script helps diagnose why reconstructions are black and provides detailed analysis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from argparse import Namespace

# Add the discrete_mbrl directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_helpers import make_env
from model_construction import construct_ae_model


def create_test_args():
    """Create arguments for model testing"""
    args = Namespace()

    # Model configuration
    args.env_name = "MiniGrid-Empty-6x6-v0"
    args.ae_model_type = "vqvae"
    args.ae_model_version = "2"
    args.embedding_dim = 64
    args.codebook_size = 256
    args.latent_dim = 32
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Required attributes
    args.filter_size = 8
    args.extra_info = None
    args.repr_sparsity = 0
    args.sparsity_type = 'random'
    args.wandb = False
    args.comet_ml = False
    args.tags = None
    args.load = False

    # FTA parameters
    args.fta_tiles = 20
    args.fta_bound_low = -2
    args.fta_bound_high = 2
    args.fta_eta = 0.2

    # Additional parameters
    args.stochastic = None
    args.trans_model_type = 'continuous'
    args.trans_model_version = '1'
    args.trans_hidden = 256
    args.trans_depth = 3
    args.vq_trans_1d_conv = False
    args.learning_rate = 1e-4
    args.ae_grad_clip = 0

    return args


def collect_real_observations(env_name, n_samples=8, device='cpu'):
    """Collect real observations from the environment"""
    print(f"Collecting {n_samples} real observations from {env_name}...")

    env = make_env(env_name)
    observations = []

    for i in range(n_samples):
        # Reset environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        observations.append(obs)

        # Take a few random steps for variety
        for _ in range(np.random.randint(0, 3)):
            action = env.action_space.sample()
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            if done:
                break

            observations[-1] = obs

    env.close()

    # Convert to tensor
    obs_array = np.stack(observations)
    obs_tensor = torch.from_numpy(obs_array).float().to(device)

    # Normalize to [0, 1] if needed
    if obs_tensor.max() > 1.0:
        obs_tensor = obs_tensor / 255.0

    return obs_tensor


def detailed_reconstruction_analysis(model, real_obs):
    """Perform detailed analysis of reconstruction process"""
    print("\n🔍 DETAILED RECONSTRUCTION ANALYSIS")
    print("=" * 50)

    with torch.no_grad():
        # Step 1: Analyze input
        print(f"📥 INPUT ANALYSIS:")
        print(f"   Shape: {real_obs.shape}")
        print(f"   Range: [{real_obs.min():.6f}, {real_obs.max():.6f}]")
        print(f"   Mean: {real_obs.mean():.6f}")
        print(f"   Std: {real_obs.std():.6f}")
        print(f"   Non-zero pixels: {(real_obs > 0).sum().item()}/{real_obs.numel()}")

        # Step 2: Analyze encoding
        print(f"\n🔢 ENCODING ANALYSIS:")
        encoded = model.encode(real_obs)
        print(f"   Encoded shape: {encoded.shape}")
        print(f"   Index range: [{encoded.min()}, {encoded.max()}]")
        print(f"   Unique indices: {torch.unique(encoded).numel()}")

        # Check if indices are valid
        valid_indices = (encoded >= 0) & (encoded < model.n_embeddings)
        print(f"   Valid indices: {valid_indices.sum().item()}/{encoded.numel()}")

        # Step 3: Analyze decoder input (quantized vectors)
        print(f"\n🎯 QUANTIZER ANALYSIS:")
        if hasattr(model, 'quantizer'):
            # Access quantizer embeddings
            quantizer = model.quantizer
            embedding_weight = quantizer._embedding.weight
            print(f"   Embedding weight shape: {embedding_weight.shape}")
            print(f"   Embedding weight range: [{embedding_weight.min():.6f}, {embedding_weight.max():.6f}]")
            print(f"   Embedding weight mean: {embedding_weight.mean():.6f}")
            print(f"   Embedding weight std: {embedding_weight.std():.6f}")

            # Get the actual quantized vectors being passed to decoder
            flat_indices = encoded.flatten()
            selected_embeddings = embedding_weight[flat_indices]
            print(f"   Selected embeddings shape: {selected_embeddings.shape}")
            print(f"   Selected embeddings range: [{selected_embeddings.min():.6f}, {selected_embeddings.max():.6f}]")
            print(f"   Selected embeddings mean: {selected_embeddings.mean():.6f}")

        # Step 4: Analyze reconstruction
        print(f"\n📤 RECONSTRUCTION ANALYSIS:")
        reconstructed = model.decode(encoded)
        print(f"   Reconstructed shape: {reconstructed.shape}")
        print(f"   Reconstructed range: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")
        print(f"   Reconstructed mean: {reconstructed.mean():.6f}")
        print(f"   Reconstructed std: {reconstructed.std():.6f}")
        print(f"   Non-zero pixels: {(reconstructed > 0).sum().item()}/{reconstructed.numel()}")

        # Step 5: Analyze reconstruction values in detail
        print(f"\n📊 RECONSTRUCTION VALUE DISTRIBUTION:")
        recon_flat = reconstructed.flatten()
        print(f"   Zeros: {(recon_flat == 0).sum().item()}")
        print(f"   Very small (< 1e-6): {(recon_flat.abs() < 1e-6).sum().item()}")
        print(f"   Small (< 1e-3): {(recon_flat.abs() < 1e-3).sum().item()}")
        print(f"   Medium (< 0.1): {(recon_flat.abs() < 0.1).sum().item()}")
        print(f"   Large (>= 0.1): {(recon_flat.abs() >= 0.1).sum().item()}")

        # Show some actual values
        print(f"   First 10 values: {recon_flat[:10].tolist()}")
        print(f"   Max 10 values: {recon_flat.topk(10).values.tolist()}")
        print(f"   Min 10 values: {recon_flat.topk(10, largest=False).values.tolist()}")

        # Step 6: Check for gradient flow issues
        print(f"\n🔄 MODEL PARAMETER ANALYSIS:")
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   Encoder parameters: {encoder_params:,}")
        print(f"   Decoder parameters: {decoder_params:,}")

        # Check parameter ranges
        decoder_weights = []
        for name, param in model.decoder.named_parameters():
            if 'weight' in name:
                decoder_weights.append(param)

        if decoder_weights:
            all_decoder_weights = torch.cat([w.flatten() for w in decoder_weights])
            print(f"   Decoder weight range: [{all_decoder_weights.min():.6f}, {all_decoder_weights.max():.6f}]")
            print(f"   Decoder weight mean: {all_decoder_weights.mean():.6f}")
            print(f"   Decoder weight std: {all_decoder_weights.std():.6f}")

        return reconstructed


def enhanced_visualization(original, reconstructed, save_path="diagnostic_vqvae_results.png"):
    """Enhanced visualization with multiple scaling attempts"""
    print(f"\n🎨 Enhanced Visualization with Multiple Scaling...")

    def tensor_to_image(tensor):
        """Convert tensor to image with proper handling"""
        img = tensor.detach().cpu()
        if len(img.shape) == 4:
            img = img.permute(0, 2, 3, 1)
        return img.numpy()

    orig_imgs = tensor_to_image(original)
    recon_imgs = tensor_to_image(reconstructed)

    # Try different scaling approaches for reconstruction
    recon_scaled_01 = np.clip(recon_imgs, 0, 1)  # Clip to [0, 1]
    recon_normalized = (recon_imgs - recon_imgs.min()) / (recon_imgs.max() - recon_imgs.min() + 1e-8)  # Normalize
    recon_scaled_255 = np.clip(recon_imgs * 255, 0, 255) / 255  # Scale assuming [0, 1] range
    recon_abs = np.abs(recon_imgs)  # Take absolute value

    n_show = min(4, len(orig_imgs))

    fig, axes = plt.subplots(6, n_show, figsize=(3 * n_show, 12))
    if n_show == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_show):
        # Original
        if len(orig_imgs[i].shape) == 2:
            axes[0, i].imshow(orig_imgs[i], cmap='gray')
        else:
            axes[0, i].imshow(orig_imgs[i])
        axes[0, i].set_title(f"Original {i + 1}")
        axes[0, i].axis('off')

        # Raw reconstruction
        if len(recon_imgs[i].shape) == 2:
            axes[1, i].imshow(recon_imgs[i], cmap='gray')
        else:
            axes[1, i].imshow(recon_imgs[i])
        axes[1, i].set_title(f"Raw Recon {i + 1}")
        axes[1, i].axis('off')

        # Clipped [0, 1]
        if len(recon_scaled_01[i].shape) == 2:
            axes[2, i].imshow(recon_scaled_01[i], cmap='gray')
        else:
            axes[2, i].imshow(recon_scaled_01[i])
        axes[2, i].set_title(f"Clipped [0,1] {i + 1}")
        axes[2, i].axis('off')

        # Normalized
        if len(recon_normalized[i].shape) == 2:
            axes[3, i].imshow(recon_normalized[i], cmap='gray')
        else:
            axes[3, i].imshow(recon_normalized[i])
        axes[3, i].set_title(f"Normalized {i + 1}")
        axes[3, i].axis('off')

        # Scaled x255
        if len(recon_scaled_255[i].shape) == 2:
            axes[4, i].imshow(recon_scaled_255[i], cmap='gray')
        else:
            axes[4, i].imshow(recon_scaled_255[i])
        axes[4, i].set_title(f"Scaled x255 {i + 1}")
        axes[4, i].axis('off')

        # Absolute value
        if len(recon_abs[i].shape) == 2:
            axes[5, i].imshow(recon_abs[i], cmap='gray')
        else:
            axes[5, i].imshow(recon_abs[i])
        axes[5, i].set_title(f"Absolute {i + 1}")
        axes[5, i].axis('off')

    plt.suptitle('Diagnostic VQ-VAE Results - Multiple Scaling Attempts', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved diagnostic visualization to: {save_path}")
    plt.show()


def suggest_fixes(reconstructed):
    """Analyze reconstruction and suggest fixes"""
    print(f"\n💡 SUGGESTED FIXES:")
    print("=" * 30)

    recon_min = reconstructed.min().item()
    recon_max = reconstructed.max().item()
    recon_mean = reconstructed.mean().item()

    if recon_max < 1e-6:
        print("🔴 ISSUE: Reconstructed values are extremely small")
        print("   POSSIBLE CAUSES:")
        print("   - Decoder weights initialized too small")
        print("   - Vanishing gradients")
        print("   - Wrong activation function")
        print("   FIXES:")
        print("   - Increase decoder weight initialization")
        print("   - Check for ReLU dying neurons")
        print("   - Try different activation functions")

    elif recon_min < 0 and recon_max > 0:
        print("🟡 ISSUE: Reconstructed values span negative and positive")
        print("   POSSIBLE CAUSES:")
        print("   - Missing final activation (sigmoid/tanh)")
        print("   - Decoder outputting raw values")
        print("   FIXES:")
        print("   - Add sigmoid activation to decoder output")
        print("   - Ensure input normalization matches output range")

    elif recon_max < 0.1 and recon_min >= 0:
        print("🟡 ISSUE: Reconstructed values are very small but positive")
        print("   POSSIBLE CAUSES:")
        print("   - Decoder learning slowly")
        print("   - Need more training")
        print("   - Learning rate too small")
        print("   FIXES:")
        print("   - This is normal for untrained model")
        print("   - Try training for a few epochs")

    else:
        print("🟢 Values look reasonable for untrained model")
        print("   - Try normalizing reconstruction to [0,1] for visualization")
        print("   - Consider training the model")


def main():
    """Main diagnostic function"""
    print("🔍 VQ-VAE DIAGNOSTIC ANALYSIS")
    print("=" * 50)

    try:
        # Setup
        args = create_test_args()

        # Create model
        model, _ = construct_ae_model([3, 48, 48], args, load=False)
        model = model.to(args.device)
        model.eval()

        print(f"✅ Model created: {type(model).__name__}")

        # Collect real observations
        real_obs = collect_real_observations(args.env_name, n_samples=8, device=args.device)

        # Detailed analysis
        reconstructed = detailed_reconstruction_analysis(model, real_obs)

        # Enhanced visualization
        enhanced_visualization(real_obs, reconstructed)

        # Suggest fixes
        suggest_fixes(reconstructed)

        print(f"\n📋 SUMMARY:")
        print(f"   Model structure: ✅ Working")
        print(f"   Data pipeline: ✅ Working")
        print(f"   Visualization: ✅ Enhanced")
        print(f"   Generated: diagnostic_vqvae_results.png")

    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()