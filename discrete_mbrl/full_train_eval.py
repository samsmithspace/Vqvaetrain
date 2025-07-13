import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from data_logging import init_experiment, finish_experiment
from env_helpers import *
from training_helpers import *
from train_encoder import train_encoder
from train_transition_model import train_trans_model
from evaluate_model import eval_model
from train_rl_model import train_rl_model
from e2e_train import full_train
import torch
from training_helpers import optimize_gpu_memory, setup_efficient_model


# Add optimized training functions
def train_encoder_optimized(args):
    """Optimized version of train_encoder"""
    from train_encoder import train_encoder

    print("Starting optimized encoder training...")
    encoder_model = train_encoder(args)

    # Apply model optimizations
    encoder_model = setup_efficient_model(encoder_model, args)

    return encoder_model


def train_trans_model_optimized(args, encoder_model=None):
    """Optimized version of train_trans_model"""
    from train_transition_model import train_trans_model

    print("Starting optimized transition model training...")

    # Optimize encoder model if provided
    if encoder_model is not None:
        encoder_model = setup_efficient_model(encoder_model, args)

    trans_model = train_trans_model(args, encoder_model)

    # Apply model optimizations
    trans_model = setup_efficient_model(trans_model, args)

    return trans_model


def full_train_optimized(args):
    """Optimized version of full_train"""
    from e2e_train import full_train

    print("Starting optimized end-to-end training...")
    encoder_model, trans_model = full_train(args)

    # Apply model optimizations
    encoder_model = setup_efficient_model(encoder_model, args)
    trans_model = setup_efficient_model(trans_model, args)

    return encoder_model, trans_model


def optimize_training_args(args):
    """
    Optimize training arguments for better GPU utilization while fixing shape issues
    """

    # 1. Adjust batch size based on n_train_unroll to avoid memory issues
    if args.n_train_unroll > 1:
        # Reduce batch size proportionally to account for the unroll dimension
        effective_batch_size = args.batch_size
        args.batch_size = max(256, args.batch_size // args.n_train_unroll)
        print(
            f"Adjusted batch size from {effective_batch_size} to {args.batch_size} due to n_train_unroll={args.n_train_unroll}")

    # 2. Set up gradient accumulation to maintain effective batch size
    target_effective_batch = 4096  # Target effective batch size
    current_effective = args.batch_size * args.n_train_unroll
    args.accumulation_steps = max(1, target_effective_batch // current_effective)

    # 3. Enable mixed precision if available
    args.use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

    # 4. Optimize data loading
    args.n_preload = min(8, os.cpu_count())

    # 5. Reduce logging frequency to improve GPU utilization
    args.log_freq = max(500, args.log_freq)

    print(f"Training configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  N-step unroll: {args.n_train_unroll}")
    print(f"  Effective batch size per step: {args.batch_size * args.n_train_unroll}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Total effective batch: {args.batch_size * args.n_train_unroll * args.accumulation_steps}")
    print(f"  Mixed precision: {args.use_amp}")

    return args


# Update your main function in full_train_eval.py
def main():
    """Updated main function with proper configuration"""
    # Parse args
    args = get_args()

    # Apply optimizations
    args = optimize_training_args(args)

    # Setup logging
    args = init_experiment('discrete-mbrl-full', args)

    # Setup GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Your existing training pipeline
    if args.e2e_loss:
        encoder_model, trans_model = full_train(args)
    else:
        encoder_model = train_encoder(args)
        trans_model = train_trans_model(args, encoder_model)
        if args.rl_train_steps > 0:
            train_rl_model(args, encoder_model, trans_model)

    eval_model(args, encoder_model, trans_model)
    finish_experiment(args)


# Alternative: If you want to use smaller batch sizes initially, try this configuration
def conservative_config(args):
    """More conservative configuration that should work reliably"""

    # Conservative batch sizes that work well with unrolling
    if args.n_train_unroll <= 2:
        args.batch_size = 1024
    elif args.n_train_unroll <= 4:
        args.batch_size = 512
    else:
        args.batch_size = 256

    # Moderate accumulation
    args.accumulation_steps = 2

    # Enable mixed precision conservatively
    args.use_amp = False  # Disable initially to ensure stability

    # Conservative data loading
    args.n_preload = 4

    return args


# For immediate testing, you can also add this to your training scripts:
def debug_tensor_shapes(batch_data, step_name=""):
    """Debug function to print tensor shapes"""
    print(f"\n=== Debug {step_name} ===")
    for i, data in enumerate(batch_data):
        if torch.is_tensor(data):
            print(f"  Tensor {i}: {data.shape} {data.dtype}")
        else:
            print(f"  Data {i}: {type(data)}")
    print("=" * 30)


# Add this import to your train_transition_model.py if not already present
from contextlib import nullcontext
import numpy as np
from torch.cuda.amp import autocast, GradScaler


# Quick fix for immediate use - add this at the top of train_transition_model.py
def fix_batch_data_shapes(batch_data):
    """Quick fix function to handle shape issues"""
    obs, actions, next_obs, rewards, dones = batch_data[:5]
    extra_data = batch_data[5:] if len(batch_data) > 5 else []

    # Check if we have the unroll dimension
    if len(obs.shape) == 5:  # [batch, n_steps, channels, height, width]
        # Reshape to combine batch and time dimensions
        batch_size, n_steps = obs.shape[:2]

        obs = obs.reshape(batch_size * n_steps, *obs.shape[2:])
        next_obs = next_obs.reshape(batch_size * n_steps, *next_obs.shape[2:])
        actions = actions.reshape(batch_size * n_steps, *actions.shape[2:])
        rewards = rewards.reshape(batch_size * n_steps, *rewards.shape[2:])
        dones = dones.reshape(batch_size * n_steps, *dones.shape[2:])

        extra_data = [x.reshape(batch_size * n_steps, *x.shape[2:]) for x in extra_data]

    return [obs, actions, next_obs, rewards, dones] + extra_data


if __name__ == '__main__':
    main()