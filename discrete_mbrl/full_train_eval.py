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


def main():
    """Run all validation tests with proper Windows support and GPU optimizations"""
    # Parse args first
    args = get_args()

    # Setup logging
    args = init_experiment('discrete-mbrl-full', args)

    print(f"Using device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Apply GPU memory optimizations
    optimize_gpu_memory()

    if args.e2e_loss:
        # Train and test end-to-end model
        encoder_model, trans_model = full_train(args)
    else:
        # Train and test the encoder model
        print("🚀 Starting encoder training...")
        encoder_model = train_encoder(args)
        print(f"✅ Encoder training completed. Model type: {type(encoder_model)}")

        # Ensure we have a proper PyTorch model before continuing
        if not hasattr(encoder_model, 'parameters'):
            raise ValueError(f"train_encoder returned invalid object: {type(encoder_model)}")

        # Train and test the transition model
        print("🚀 Starting transition model training...")
        trans_model = train_trans_model(args, encoder_model)
        print(f"✅ Transition model training completed. Model type: {type(trans_model)}")

        # Train and evaluate an RL model with the learned model
        if args.rl_train_steps > 0:
            print("🚀 Starting RL model training...")
            train_rl_model(args, encoder_model, trans_model)

    # Evaluate the models
    print("🚀 Starting model evaluation...")
    eval_model(args, encoder_model, trans_model)

    # Clean up logging
    finish_experiment(args)


if __name__ == '__main__':
    main()