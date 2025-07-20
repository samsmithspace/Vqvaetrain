#!/usr/bin/env python3
"""
Encoder Model Evaluation Script
Focused evaluation of autoencoder models without transition model components.
"""

import gc
import psutil
import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import torch
from tqdm import tqdm

from shared.models import *
from shared.trainers import *
from data_helpers import *
from data_logging import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from utils import *

sns.set()

# Configuration constants
SEED = 0  # Should be same as seed used for training
PRELOAD_TEST = False
TEST_WORKERS = 0
N_RAND_LATENT_SAMPLES = 500
N_EXAMPLE_IMGS = 15
DISCRETE_ENCODER_TYPES = ('vqvae', 'dae', 'softmax_ae', 'hard_fta_ae')
CONTINUOUS_ENCODER_TYPES = ('ae', 'vae', 'soft_vqvae', 'fta_ae')


def setup_evaluation_environment(args):
    """Setup data loaders and basic environment for evaluation"""
    print('Loading test data...')

    test_loader = prepare_dataloader(
        args.env_name, 'test', batch_size=args.batch_size, preprocess=args.preprocess,
        randomize=False, n=args.max_transitions, n_preload=TEST_WORKERS, preload=args.preload_data,
        extra_buffer_keys=args.extra_buffer_keys)

    test_sampler = create_fast_loader(
        test_loader.dataset, batch_size=1, shuffle=True, num_workers=TEST_WORKERS, n_step=1)

    rev_transform = test_loader.dataset.flat_rev_obs_transform

    print(f'Test dataset size: {len(test_loader.dataset)}')

    return test_loader, test_sampler, rev_transform


def load_and_setup_encoder(args, test_sampler):
    """Load and setup the encoder model"""
    print('Loading encoder model...')

    # Get sample observation shape
    sample_obs = next(iter(test_sampler))[0]

    # Load the encoder
    encoder_model, trainer = construct_ae_model(sample_obs.shape[1:], args, load=args.load)
    encoder_model = encoder_model.to(args.device)
    freeze_model(encoder_model)
    encoder_model.eval()

    if hasattr(encoder_model, 'enable_sparsity'):
        encoder_model.enable_sparsity()

    print(f'Loaded encoder: {type(encoder_model).__name__}')
    print(f'Encoder parameters: {sum(p.numel() for p in encoder_model.parameters()):,}')

    # Determine encoder type
    if args.ae_model_type in DISCRETE_ENCODER_TYPES:
        encoder_type = 'discrete'
    elif args.ae_model_type in CONTINUOUS_ENCODER_TYPES:
        encoder_type = 'continuous'
    else:
        encoder_type = 'other'

    return encoder_model, trainer, encoder_type


def evaluate_reconstruction_performance(encoder_model, test_loader, args):
    """Evaluate autoencoder reconstruction performance"""
    print('\n🔍 EVALUATING RECONSTRUCTION PERFORMANCE')
    print('=' * 50)

    # Calculate reconstruction loss
    n_samples = 0
    total_recon_loss = 0.0
    mse_losses = []

    device = next(encoder_model.parameters()).device

    for batch_data in tqdm(test_loader, desc="Computing reconstruction losses"):
        obs_data = batch_data[0]
        batch_size = obs_data.shape[0]
        n_samples += batch_size

        with torch.no_grad():
            # Encode and decode
            encoded = encoder_model.encode(obs_data.to(device))
            decoded = encoder_model.decode(encoded)

            # Calculate MSE loss
            mse_loss = torch.mean((decoded.cpu() - obs_data) ** 2, dim=(1, 2, 3))
            mse_losses.extend(mse_loss.tolist())

            total_recon_loss += torch.sum(mse_loss).item()

    avg_recon_loss = total_recon_loss / n_samples
    mse_std = np.std(mse_losses)

    print(f'📊 Reconstruction Statistics:')
    print(f'   Average MSE Loss: {avg_recon_loss:.6f}')
    print(f'   MSE Std Dev: {mse_std:.6f}')
    print(f'   Min MSE: {min(mse_losses):.6f}')
    print(f'   Max MSE: {max(mse_losses):.6f}')
    print(f'   Samples evaluated: {n_samples:,}')

    # Log metrics
    log_metrics({
        'encoder_avg_mse_loss': avg_recon_loss,
        'encoder_mse_std': mse_std,
        'encoder_min_mse': min(mse_losses),
        'encoder_max_mse': max(mse_losses),
        'encoder_samples_evaluated': n_samples
    }, args)

    return avg_recon_loss, mse_losses


def analyze_latent_space(encoder_model, test_loader, encoder_type, args):
    """Analyze the learned latent space"""
    print('\n🎯 ANALYZING LATENT SPACE')
    print('=' * 50)

    device = next(encoder_model.parameters()).device
    all_latents = []

    # Collect latent representations
    print('Collecting latent representations...')
    for batch_data in tqdm(test_loader, desc="Encoding observations"):
        obs_data = batch_data[0]
        with torch.no_grad():
            latents = encoder_model.encode(obs_data.to(device))
            all_latents.append(latents.cpu())

    all_latents = torch.cat(all_latents, dim=0)

    print(f'📊 Latent Space Statistics:')
    print(f'   Latent shape: {all_latents.shape}')
    print(f'   Latent range: [{all_latents.min():.4f}, {all_latents.max():.4f}]')
    print(f'   Latent mean: {all_latents.mean():.4f}')
    print(f'   Latent std: {all_latents.std():.4f}')

    # Type-specific analysis
    if encoder_type == 'discrete':
        print(f'   Unique values: {torch.unique(all_latents).numel()}')
        if hasattr(encoder_model, 'n_embeddings'):
            print(f'   Codebook size: {encoder_model.n_embeddings}')
            print(f'   Codebook utilization: {torch.unique(all_latents).numel() / encoder_model.n_embeddings:.2%}')

    elif encoder_type == 'continuous':
        if len(all_latents.shape) > 2:
            all_latents = all_latents.reshape(all_latents.shape[0], -1)

        print(f'   Latent dimensions: {all_latents.shape[1]}')

        # Per-dimension statistics
        dim_means = all_latents.mean(dim=0)
        dim_stds = all_latents.std(dim=0)

        print(f'   Dimension mean range: [{dim_means.min():.4f}, {dim_means.max():.4f}]')
        print(f'   Dimension std range: [{dim_stds.min():.4f}, {dim_stds.max():.4f}]')

        # Check for dead dimensions (very low variance)
        dead_dims = (dim_stds < 0.01).sum().item()
        if dead_dims > 0:
            print(f'   ⚠️  Dead dimensions (std < 0.01): {dead_dims}')

    # Log metrics
    log_metrics({
        'latent_min': all_latents.min().item(),
        'latent_max': all_latents.max().item(),
        'latent_mean': all_latents.mean().item(),
        'latent_std': all_latents.std().item(),
        'latent_unique_values': torch.unique(all_latents).numel()
    }, args)

    return all_latents


def sample_from_latent_space(encoder_model, all_latents, encoder_type, args, rev_transform):
    """Sample from the latent space and generate images"""
    print('\n🎲 SAMPLING FROM LATENT SPACE')
    print('=' * 50)

    device = next(encoder_model.parameters()).device

    if encoder_type == 'continuous':
        print('Sampling from continuous latent space...')

        # Reshape latents if needed
        if hasattr(encoder_model, 'latent_dim'):
            latent_dim = encoder_model.latent_dim
            all_latents = all_latents.reshape(all_latents.shape[0], latent_dim)
        else:
            latent_dim = all_latents.shape[1] if len(all_latents.shape) == 2 else np.prod(all_latents.shape[1:])
            all_latents = all_latents.reshape(all_latents.shape[0], latent_dim)

        # Uniform sampling
        latent_min = all_latents.min()
        latent_max = all_latents.max()
        latent_range = latent_max - latent_min
        uniform_sampled_latents = torch.rand((N_RAND_LATENT_SAMPLES, latent_dim))
        uniform_sampled_latents = uniform_sampled_latents * latent_range + latent_min

        with torch.no_grad():
            uniform_obs = encoder_model.decode(uniform_sampled_latents.to(device))

        uniform_imgs = obs_to_img(uniform_obs[:N_EXAMPLE_IMGS].cpu(), env_name=args.env_name,
                                  rev_transform=rev_transform)
        log_images({'uniform_continuous_samples': uniform_imgs}, args)
        print(f'   ✅ Generated {N_EXAMPLE_IMGS} uniform samples')

        # Normal sampling (using empirical mean and std)
        latent_means = all_latents.mean(dim=0)
        latent_stds = all_latents.std(dim=0)
        normal_sampled_latents = torch.normal(
            latent_means.repeat(N_RAND_LATENT_SAMPLES, 1),
            latent_stds.repeat(N_RAND_LATENT_SAMPLES, 1))

        with torch.no_grad():
            normal_obs = encoder_model.decode(normal_sampled_latents.to(device))

        normal_imgs = obs_to_img(normal_obs[:N_EXAMPLE_IMGS].cpu(), env_name=args.env_name, rev_transform=rev_transform)
        log_images({'normal_continuous_samples': normal_imgs}, args)
        print(f'   ✅ Generated {N_EXAMPLE_IMGS} normal samples')

    elif encoder_type == 'discrete':
        print('Sampling from discrete latent space...')

        if hasattr(encoder_model, 'n_latent_embeds') and hasattr(encoder_model, 'n_embeddings'):
            latent_dim = encoder_model.n_latent_embeds
            n_embeddings = encoder_model.n_embeddings

            # Uniform discrete sampling
            sampled_latents = torch.randint(
                0, n_embeddings, (N_RAND_LATENT_SAMPLES, latent_dim))

            with torch.no_grad():
                sampled_obs = encoder_model.decode(sampled_latents.to(device))

            sampled_imgs = obs_to_img(sampled_obs[:N_EXAMPLE_IMGS].cpu(), env_name=args.env_name,
                                      rev_transform=rev_transform)
            log_images({'uniform_discrete_samples': sampled_imgs}, args)
            print(f'   ✅ Generated {N_EXAMPLE_IMGS} discrete samples')
        else:
            print('   ⚠️  Cannot sample from this discrete encoder type')


def generate_reconstruction_examples(encoder_model, test_sampler, args, rev_transform):
    """Generate reconstruction example images"""
    print('\n🖼️  GENERATING RECONSTRUCTION EXAMPLES')
    print('=' * 50)

    device = next(encoder_model.parameters()).device
    example_imgs = []

    for i, sample_transition in enumerate(test_sampler):
        if i >= N_EXAMPLE_IMGS:
            break

        sample_obs = sample_transition[0]

        with torch.no_grad():
            # Get reconstruction
            recon_result = encoder_model(sample_obs.to(device))

            # Handle different return formats
            if isinstance(recon_result, tuple):
                recon_obs = recon_result[0]  # Take the reconstructed observation
            else:
                recon_obs = recon_result

        # Create side-by-side comparison
        both_obs = torch.cat([sample_obs, recon_obs.cpu()], dim=0)
        both_imgs = obs_to_img(both_obs, env_name=args.env_name, rev_transform=rev_transform)

        # Concatenate original and reconstruction horizontally
        if len(both_imgs.shape) == 4:  # Multiple images
            cat_img = np.concatenate([both_imgs[0], both_imgs[1]], axis=1)
        else:  # Single comparison
            cat_img = both_imgs

        example_imgs.append(cat_img)

    # Create a grid of all examples
    if example_imgs:
        grid_img = np.concatenate(example_imgs, axis=1)

        plt.figure(figsize=(N_EXAMPLE_IMGS * 2, 4))
        plt.imshow(grid_img.clip(0, 1))
        plt.title('Reconstruction Examples (Original | Reconstructed)')
        plt.axis('off')

        log_images({'reconstruction_examples': example_imgs}, args)
        print(f'   ✅ Generated {len(example_imgs)} reconstruction examples')

    return example_imgs


def analyze_reconstruction_quality(encoder_model, test_sampler, args):
    """Analyze reconstruction quality with detailed metrics"""
    print('\n📈 ANALYZING RECONSTRUCTION QUALITY')
    print('=' * 50)

    device = next(encoder_model.parameters()).device

    # Collect detailed metrics
    pixel_errors = []
    ssim_scores = []
    reconstruction_times = []

    n_analyzed = 0
    max_analyze = 1000  # Limit for detailed analysis

    for i, sample_transition in enumerate(test_sampler):
        if i >= max_analyze:
            break

        sample_obs = sample_transition[0]
        n_analyzed += 1

        # Time the reconstruction
        start_time = time.time()
        with torch.no_grad():
            recon_result = encoder_model(sample_obs.to(device))
            if isinstance(recon_result, tuple):
                recon_obs = recon_result[0]
            else:
                recon_obs = recon_result
        end_time = time.time()

        reconstruction_times.append(end_time - start_time)

        # Calculate pixel-wise error
        pixel_error = torch.mean(torch.abs(sample_obs - recon_obs.cpu())).item()
        pixel_errors.append(pixel_error)

    # Compute statistics
    avg_pixel_error = np.mean(pixel_errors)
    avg_recon_time = np.mean(reconstruction_times)

    print(f'📊 Quality Metrics (n={n_analyzed}):')
    print(f'   Average pixel error (L1): {avg_pixel_error:.6f}')
    print(f'   Pixel error std: {np.std(pixel_errors):.6f}')
    print(f'   Average reconstruction time: {avg_recon_time:.4f}s')
    print(f'   Reconstruction throughput: {1 / avg_recon_time:.1f} samples/sec')

    # Log metrics
    log_metrics({
        'avg_pixel_error_l1': avg_pixel_error,
        'pixel_error_std': np.std(pixel_errors),
        'avg_reconstruction_time': avg_recon_time,
        'reconstruction_throughput': 1 / avg_recon_time
    }, args)

    return avg_pixel_error, reconstruction_times


def main():
    """Main evaluation function"""
    print("🔍 ENCODER MODEL EVALUATION")
    print("=" * 60)

    # Parse arguments
    args = get_args()

    # Setup logging
    args = init_experiment('discrete-mbrl-encoder-eval', args)

    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    try:
        # Setup evaluation environment
        test_loader, test_sampler, rev_transform = setup_evaluation_environment(args)

        # Load and setup encoder
        encoder_model, trainer, encoder_type = load_and_setup_encoder(args, test_sampler)

        # Track model for logging
        track_model(encoder_model, args)

        print(f'\n🎯 Evaluating {encoder_type} encoder: {args.ae_model_type}')
        print(f'Environment: {args.env_name}')
        print(f'Device: {args.device}')

        # Evaluate reconstruction performance
        avg_recon_loss, mse_losses = evaluate_reconstruction_performance(
            encoder_model, test_loader, args)

        # Analyze latent space
        all_latents = analyze_latent_space(
            encoder_model, test_loader, encoder_type, args)

        # Sample from latent space
        sample_from_latent_space(
            encoder_model, all_latents, encoder_type, args, rev_transform)

        # Generate reconstruction examples
        example_imgs = generate_reconstruction_examples(
            encoder_model, test_sampler, args, rev_transform)

        # Analyze reconstruction quality
        avg_pixel_error, recon_times = analyze_reconstruction_quality(
            encoder_model, test_sampler, args)

        # Final summary
        print('\n' + '=' * 60)
        print('📋 EVALUATION SUMMARY')
        print('=' * 60)
        print(f'🎯 Model: {args.ae_model_type} ({encoder_type})')
        print(f'📊 Reconstruction MSE: {avg_recon_loss:.6f}')
        print(f'📊 Pixel Error (L1): {avg_pixel_error:.6f}')
        print(f'⚡ Throughput: {1 / np.mean(recon_times):.1f} samples/sec')
        print(f'🎨 Generated {len(example_imgs)} visualization examples')
        print(f'✅ Evaluation completed successfully!')

        # Log final summary
        log_metrics({
            'evaluation_summary': {
                'model_type': args.ae_model_type,
                'encoder_type': encoder_type,
                'final_mse': avg_recon_loss,
                'final_l1': avg_pixel_error,
                'throughput': 1 / np.mean(recon_times)
            }
        }, args)

    except KeyboardInterrupt:
        print('\n❌ Evaluation interrupted by user')
    except Exception as e:
        print(f'\n❌ Evaluation failed: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Clean up logging
        finish_experiment(args)

        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()