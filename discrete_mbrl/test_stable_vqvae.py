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

# Configuration constants - matching evaluate_model.py
SEED = 0  # Should be same as seed used for training
PRELOAD_TEST = False
TEST_WORKERS = 0
N_RAND_LATENT_SAMPLES = 500
N_EXAMPLE_IMGS = 15
DISCRETE_ENCODER_TYPES = ('vqvae', 'dae', 'softmax_ae', 'hard_fta_ae')
CONTINUOUS_ENCODER_TYPES = ('ae', 'vae', 'soft_vqvae', 'fta_ae')


def setup_evaluation_environment(args):
    """Setup data loaders and basic environment for evaluation - matching evaluate_model.py"""
    print('Loading test data...')

    # Use exact same data loading as evaluate_model.py
    test_loader = prepare_dataloader(
        args.env_name, 'test', batch_size=args.batch_size, preprocess=args.preprocess,
        randomize=False, n=args.max_transitions, n_preload=TEST_WORKERS, preload=args.preload_data,
        extra_buffer_keys=args.extra_buffer_keys)

    test_sampler = create_fast_loader(
        test_loader.dataset, batch_size=1, shuffle=True, num_workers=TEST_WORKERS, n_step=1)

    rev_transform = test_loader.dataset.flat_rev_obs_transform

    print(f'Test dataset size: {len(test_loader.dataset)}')

    return test_loader, test_sampler, rev_transform


def find_available_models(args):
    """Find all available model files for the environment"""
    model_dir = f'./models/{args.env_name}'
    if not os.path.exists(model_dir):
        return []

    model_files = []
    for filename in os.listdir(model_dir):
        if filename.startswith('model_') and filename.endswith('.pt'):
            model_path = os.path.join(model_dir, filename)
            # Extract hash from filename
            hash_part = filename.replace('model_', '').replace('.pt', '')
            model_files.append((model_path, hash_part))

    return model_files


def load_model_with_fallback(model, args):
    """Try to load model with fallback mechanisms"""
    from model_construction import make_model_hash, AE_MODEL_VARS, MODEL_SAVE_FORMAT

    # Check if user specified a specific model file
    if args.model_file:
        model_path = os.path.join(f'./models/{args.env_name}', args.model_file)
        if os.path.exists(model_path):
            print(f'🎯 Loading user-specified model: {args.model_file}')
            try:
                model.load_state_dict(torch.load(model_path, map_location=args.device))
                hash_from_filename = args.model_file.replace('model_', '').replace('.pt', '')
                return model, hash_from_filename
            except Exception as e:
                print(f'❌ Failed to load specified model: {e}')
                return model, None
        else:
            print(f'❌ Specified model file not found: {model_path}')
            return model, None

    # Check if user specified a specific hash
    target_hash = args.model_hash if args.model_hash else make_model_hash(args, model_vars=AE_MODEL_VARS,
                                                                          exp_type='encoder')
    expected_path = MODEL_SAVE_FORMAT.format(args.env_name, target_hash).replace(':', '-')

    print(f'🔍 Looking for model with hash: {target_hash}')
    print(f'🔍 Expected path: {expected_path}')

    if os.path.exists(expected_path):
        print(f'✅ Found expected model, loading...')
        try:
            model.load_state_dict(torch.load(expected_path, map_location=args.device))
            return model, target_hash
        except Exception as e:
            print(f'❌ Failed to load expected model: {e}')

    # Fallback: try to find any available models
    print(f'❌ Expected model not found, searching for alternatives...')
    available_models = find_available_models(args)

    if not available_models:
        print(f'❌ No models found in ./models/{args.env_name}/')
        print(f'💡 Tip: Use --list_models to see available models')
        print(f'💡 Tip: Use --debug_hash to see what hash is being generated')
        return model, None

    print(f'🔍 Found {len(available_models)} available models:')
    for i, (path, hash_val) in enumerate(available_models):
        print(f'   {i + 1}. {os.path.basename(path)} (hash: {hash_val})')

    # Try loading the first available model
    model_path, model_hash = available_models[0]
    print(f'🔄 Attempting to load: {os.path.basename(model_path)}')

    try:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f'✅ Successfully loaded model with hash: {model_hash}')
        return model, model_hash
    except Exception as e:
        print(f'❌ Failed to load model: {e}')
        return model, None


def load_and_setup_encoder(args, test_sampler):
    """Load and setup the encoder model - matching evaluate_model.py exactly"""
    print('Loading encoder model...')

    # Get sample observation shape - exactly like evaluate_model.py
    sample_obs = next(iter(test_sampler))[0]

    # Load the encoder - exactly like evaluate_model.py
    encoder_model = construct_ae_model(sample_obs.shape[1:], args)[0]
    encoder_model = encoder_model.to(args.device)
    freeze_model(encoder_model)
    encoder_model.eval()
    print(f'Loaded encoder')

    if hasattr(encoder_model, 'enable_sparsity'):
        encoder_model.enable_sparsity()

    print(f'📊 Encoder info:')
    print(f'   Type: {type(encoder_model).__name__}')
    print(f'   Parameters: {sum(p.numel() for p in encoder_model.parameters()):,}')

    # Determine encoder type
    if args.ae_model_type in DISCRETE_ENCODER_TYPES:
        encoder_type = 'discrete'
    elif args.ae_model_type in CONTINUOUS_ENCODER_TYPES:
        encoder_type = 'continuous'
    else:
        encoder_type = 'other'

    return encoder_model, None, encoder_type


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


def plot_and_save_reconstruction_comparison(encoder_model, test_sampler, args, rev_transform, n_examples=8):
    """Plot and save a comprehensive comparison of original vs reconstructed images"""
    print('\n🎨 CREATING RECONSTRUCTION COMPARISON PLOT')
    print('=' * 50)

    device = next(encoder_model.parameters()).device
    encoder_model.eval()

    # Collect examples
    original_images = []
    reconstructed_images = []
    mse_errors = []

    print(f'Collecting {n_examples} examples...')
    for i, sample_transition in enumerate(test_sampler):
        if i >= n_examples:
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

        # Calculate MSE for this example
        mse_error = torch.mean((sample_obs - recon_obs.cpu()) ** 2).item()
        mse_errors.append(mse_error)

        # Convert to images
        orig_img = obs_to_img(sample_obs, env_name=args.env_name, rev_transform=rev_transform)
        recon_img = obs_to_img(recon_obs.cpu(), env_name=args.env_name, rev_transform=rev_transform)

        # Handle different image formats
        if len(orig_img.shape) == 3:  # Single image
            original_images.append(orig_img)
            reconstructed_images.append(recon_img)
        elif len(orig_img.shape) == 4:  # Batch
            original_images.append(orig_img[0])
            reconstructed_images.append(recon_img[0])

    # Create the comparison plot
    fig, axes = plt.subplots(3, n_examples, figsize=(2 * n_examples, 6))
    if n_examples == 1:
        axes = axes.reshape(-1, 1)

    # Plot original images (top row)
    for i in range(n_examples):
        img = original_images[i]
        if len(img.shape) == 2:  # Grayscale
            axes[0, i].imshow(img, cmap='gray')
        else:  # RGB
            axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original {i + 1}', fontsize=10)
        axes[0, i].axis('off')

    # Plot reconstructed images (middle row)
    for i in range(n_examples):
        img = reconstructed_images[i]
        if len(img.shape) == 2:  # Grayscale
            axes[1, i].imshow(img, cmap='gray')
        else:  # RGB
            axes[1, i].imshow(img)
        axes[1, i].set_title(f'Reconstructed {i + 1}', fontsize=10)
        axes[1, i].axis('off')

    # Plot difference images (bottom row)
    for i in range(n_examples):
        orig = original_images[i]
        recon = reconstructed_images[i]

        # Calculate difference
        if len(orig.shape) == 3 and orig.shape[2] == 3:  # RGB
            # Convert to grayscale for difference
            orig_gray = np.mean(orig, axis=2)
            recon_gray = np.mean(recon, axis=2)
            diff = np.abs(orig_gray - recon_gray)
        else:  # Already grayscale or single channel
            diff = np.abs(orig - recon)

        im = axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[2, i].set_title(f'Diff (MSE: {mse_errors[i]:.4f})', fontsize=10)
        axes[2, i].axis('off')

    # Add colorbar for difference images
    plt.colorbar(im, ax=axes[2, :], orientation='horizontal', shrink=0.8, pad=0.1)

    # Add overall title and statistics
    avg_mse = np.mean(mse_errors)
    std_mse = np.std(mse_errors)
    fig.suptitle(f'Reconstruction Comparison - {args.ae_model_type.upper()}\n'
                 f'Environment: {args.env_name}\n'
                 f'Average MSE: {avg_mse:.6f} ± {std_mse:.6f}',
                 fontsize=14, y=0.98)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)

    # Save the plot
    os.makedirs('./results', exist_ok=True)
    save_path = f'./results/reconstruction_comparison_{args.env_name}_{args.ae_model_type}.png'
    save_path = save_path.replace(':', '_')  # Handle Windows-incompatible characters

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'📁 Saved reconstruction comparison to: {save_path}')

    # Also save a higher-level summary plot
    save_summary_plot(args, avg_mse, std_mse, mse_errors)

    # Log to experiment tracking if available
    if hasattr(args, 'wandb') and args.wandb:
        try:
            import wandb
            wandb.log({
                "reconstruction_comparison": wandb.Image(fig),
                "avg_reconstruction_mse": avg_mse,
                "reconstruction_mse_std": std_mse
            })
        except:
            pass

    plt.show()
    return save_path, avg_mse, mse_errors


def save_summary_plot(args, avg_mse, std_mse, mse_errors):
    """Save a summary statistics plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of MSE errors
    ax1.hist(mse_errors, bins=max(3, len(mse_errors) // 2), alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(avg_mse, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_mse:.6f}')
    ax1.set_xlabel('MSE Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Reconstruction Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Summary statistics
    stats_text = f"""
Model Type: {args.ae_model_type.upper()}
Environment: {args.env_name}

Reconstruction Statistics:
• Average MSE: {avg_mse:.6f}
• Std Dev: {std_mse:.6f}
• Min MSE: {min(mse_errors):.6f}
• Max MSE: {max(mse_errors):.6f}
• Samples: {len(mse_errors)}

Model Parameters:
• Embedding Dim: {getattr(args, 'embedding_dim', 'N/A')}
• Codebook Size: {getattr(args, 'codebook_size', 'N/A')}
• Latent Dim: {getattr(args, 'latent_dim', 'N/A')}
    """

    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax2.axis('off')
    ax2.set_title('Model Summary')

    plt.tight_layout()

    # Save summary
    summary_path = f'./results/reconstruction_summary_{args.env_name}_{args.ae_model_type}.png'
    summary_path = summary_path.replace(':', '_')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'📁 Saved reconstruction summary to: {summary_path}')
    plt.close()

    return summary_path


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


def add_evaluation_args(parser):
    """Add evaluation-specific arguments"""
    parser.add_argument('--model_hash', type=str, default=None,
                        help='Specific model hash to load (overrides automatic hash)')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Specific model file to load (e.g., model_abc123.pt)')
    parser.add_argument('--list_models', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--debug_hash', action='store_true',
                        help='Debug model hash generation')
    return parser


def debug_model_hash(args):
    """Debug the model hash generation process"""
    from model_construction import make_model_hash, AE_MODEL_VARS

    print("🔍 DEBUGGING MODEL HASH GENERATION")
    print("=" * 50)

    # Get the hash and parameters used
    hash_value = make_model_hash(args, model_vars=AE_MODEL_VARS, exp_type='encoder')

    print(f"Generated hash: {hash_value}")
    print(f"Model variables used for hashing:")

    args_dict = vars(args)
    for var in AE_MODEL_VARS:
        if var in args_dict:
            print(f"  {var}: {args_dict[var]}")
        else:
            print(f"  {var}: NOT SET")

    print(f"\nExpected model path: ./models/{args.env_name}/model_{hash_value}.pt")


def list_available_models(args):
    """List all available models for the environment"""
    print(f"🔍 AVAILABLE MODELS FOR {args.env_name}")
    print("=" * 50)

    available_models = find_available_models(args)

    if not available_models:
        print(f"❌ No models found in ./models/{args.env_name}/")
        return

    print(f"Found {len(available_models)} model(s):")
    for i, (path, hash_val) in enumerate(available_models):
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        mod_time = os.path.getmtime(path)
        mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))

        print(f"  {i + 1}. {os.path.basename(path)}")
        print(f"     Hash: {hash_val}")
        print(f"     Size: {file_size:.2f} MB")
        print(f"     Modified: {mod_time_str}")
        print()


def main():
    """Main evaluation function"""
    print("🔍 ENCODER MODEL EVALUATION")
    print("=" * 60)

    # Parse arguments with evaluation-specific options
    parser = make_argparser()
    parser = add_evaluation_args(parser)
    args = parser.parse_args()
    args = process_args(args)

    # Handle special modes
    if args.list_models:
        list_available_models(args)
        return

    if args.debug_hash:
        debug_model_hash(args)
        return

    # Setup logging
    args = init_experiment('discrete-mbrl-encoder-eval', args)

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

        # Generate and save reconstruction comparison plots
        save_path, avg_mse_plot, mse_errors = plot_and_save_reconstruction_comparison(
            encoder_model, test_sampler, args, rev_transform, n_examples=8)

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
        print(f'🎨 Saved reconstruction comparison plot: {save_path}')
        print(f'✅ Evaluation completed successfully!')

        # Log final summary
        log_metrics({
            'evaluation_summary': {
                'model_type': args.ae_model_type,
                'encoder_type': encoder_type,
                'final_mse': avg_recon_loss,
                'final_l1': avg_pixel_error,
                'throughput': 1 / np.mean(recon_times),
                'plot_saved_to': save_path
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