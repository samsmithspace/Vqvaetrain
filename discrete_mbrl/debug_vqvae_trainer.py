#!/usr/bin/env python3
"""
Encoder Model Evaluation Script - Plot Only
Simplified version that only generates reconstruction comparison plots.
"""

import gc
import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

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
TEST_WORKERS = 0


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


def find_available_models(args):
    """Find all available model files for the environment"""
    model_dir = f'./models/{args.env_name}'
    if not os.path.exists(model_dir):
        return []

    model_files = []
    for filename in os.listdir(model_dir):
        if filename.startswith('model_') and filename.endswith('.pt'):
            model_path = os.path.join(model_dir, filename)
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
    """Load and setup the encoder model"""
    print('Loading encoder model...')

    # Get sample observation shape
    sample_obs = next(iter(test_sampler))[0]

    # Load the encoder
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

    return encoder_model


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

    # Also save a summary plot
    save_summary_plot(args, avg_mse, std_mse, mse_errors)

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
    """Main evaluation function - plot only version"""
    print("🎨 ENCODER MODEL RECONSTRUCTION PLOT")
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
    args = init_experiment('discrete-mbrl-encoder-plot', args)

    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    try:
        # Setup evaluation environment
        test_loader, test_sampler, rev_transform = setup_evaluation_environment(args)

        # Load and setup encoder
        encoder_model = load_and_setup_encoder(args, test_sampler)

        print(f'\n🎯 Creating reconstruction plots for: {args.ae_model_type}')
        print(f'Environment: {args.env_name}')
        print(f'Device: {args.device}')

        # Generate and save reconstruction comparison plots
        save_path, avg_mse, mse_errors = plot_and_save_reconstruction_comparison(
            encoder_model, test_sampler, args, rev_transform, n_examples=8)

        # Final summary
        print('\n' + '=' * 60)
        print('📋 PLOT GENERATION SUMMARY')
        print('=' * 60)
        print(f'🎯 Model: {args.ae_model_type}')
        print(f'📊 Average MSE: {avg_mse:.6f}')
        print(f'🎨 Saved reconstruction comparison plot: {save_path}')
        print(f'✅ Plot generation completed successfully!')

    except KeyboardInterrupt:
        print('\n❌ Plot generation interrupted by user')
    except Exception as e:
        print(f'\n❌ Plot generation failed: {e}')
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