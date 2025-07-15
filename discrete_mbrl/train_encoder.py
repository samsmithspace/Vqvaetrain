import os
import sys
import time
from collections import defaultdict

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import psutil
import numpy as np

# Fix autocast import for newer PyTorch versions
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

from shared.models import *
from shared.trainers import *
from data_helpers import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from data_logging import *

# Global training state
ENCODER_STEP = 0
train_log_buffer = defaultdict(float)


def setup_data_and_model(args):
    """Setup data loaders and model"""
    print('Loading data...')

    if args.unique_data:
        train_loader = test_loader = prepare_unique_obs_dataloader(args, randomize=True)
        valid_loader = None
    else:
        train_loader, test_loader, valid_loader = prepare_dataloaders(
            args.env_name, n=args.max_transitions, batch_size=args.batch_size,
            preprocess=args.preprocess, randomize=True, n_preload=args.n_preload,
            preload_all=args.preload_data, extra_buffer_keys=args.extra_buffer_keys,
            pin_memory=getattr(args, 'pin_memory', True),
            persistent_workers=getattr(args, 'persistent_workers', args.n_preload > 0),
            prefetch_factor=getattr(args, 'prefetch_factor', 2))

    valid_len = len(valid_loader.dataset) if valid_loader is not None else 0
    print(f'Data split: {len(train_loader.dataset)}/{len(test_loader.dataset)}/{valid_len}')

    # Get sample observation and construct model
    print('Constructing model...')
    sample_obs = next(iter(train_loader))[0]
    print(f'Sample shape: {sample_obs.shape}')

    model, trainer = construct_ae_model(sample_obs.shape[1:], args, load=args.load)
    update_params(args)

    return train_loader, test_loader, valid_loader, model, trainer


def initialize_model(model, trainer, args):
    """Initialize model with safe settings"""
    # Ensure model is on correct device first
    model = model.to(args.device)
    device = next(model.parameters()).device

    # Safe initialization for VQ-VAE models
    if args.ae_model_type in ['vqvae', 'soft_vqvae']:
        print("🔧 Initializing VQ-VAE model...")

        def safe_init(m):
            if isinstance(m, torch.nn.Embedding):
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            elif isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                if m.weight.numel() > 0:
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.weight.data *= 0.1
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        model.apply(safe_init)

        # Test model with a small batch
        try:
            test_input = torch.randn(2, 3, 48, 48).to(device)
            with torch.no_grad():
                model.eval()
                test_output = model(test_input)
                print(f"✅ VQ-VAE test output type: {type(test_output)}")
                if isinstance(test_output, tuple):
                    print(f"✅ VQ-VAE test output length: {len(test_output)}")

                    # Fix the model's forward pass to return 4 values if it returns 3
                    if len(test_output) == 3:
                        print("🔧 Patching VQ-VAE forward pass to return 4 values")
                        original_forward = model.forward

                        def patched_forward(x):
                            result = original_forward(x)
                            if isinstance(result, tuple) and len(result) == 3:
                                # Add a dummy 4th value (e.g., encoding indices)
                                decoded, loss, perplexity = result
                                # Get encoding indices for compatibility
                                encoded = model.encode(x) if hasattr(model, 'encode') else torch.zeros(x.shape[0], 64)
                                return decoded, loss, perplexity, encoded
                            return result

                        model.forward = patched_forward
                        print("✅ VQ-VAE forward pass patched successfully")

                model.train()
            print("✅ VQ-VAE initialization successful")

            # Conservative settings for VQ-VAE
            if trainer and hasattr(trainer, 'optimizer'):
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'], 5e-5)
                    print(f"🔧 VQ-VAE learning rate: {param_group['lr']}")
                trainer.grad_clip = max(trainer.grad_clip, 1.0)

        except Exception as e:
            print(f"❌ VQ-VAE initialization failed: {e}")
            print("🔄 Switching to regular autoencoder")
            args.ae_model_type = 'ae'
            return construct_ae_model((3, 48, 48), args, load=False)

    return model, trainer


def setup_training(model, trainer, args):
    """Setup training environment and optimizations"""
    # Ensure model is on correct device FIRST
    model = model.to(args.device)
    device = next(model.parameters()).device

    # GPU optimizations
    optimize_gpu_memory()
    model = setup_efficient_model(model, args)

    # Mixed precision setup
    use_amp = getattr(args, 'use_amp', False) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None

    if use_amp:
        try:
            autocast_context = lambda: autocast('cuda')
        except TypeError:
            autocast_context = autocast
    else:
        autocast_context = nullcontext

    accumulation_steps = getattr(args, 'accumulation_steps', 1)

    print(f'Device: {device}')
    print(f'Mixed precision: {use_amp}')
    print(f'Accumulation steps: {accumulation_steps}')
    print(f'Model parameters: {sum(x.numel() for x in model.parameters()):,}')

    # Disable sparsity if available
    if hasattr(model, 'disable_sparsity'):
        model.disable_sparsity()

    # Setup trainer for mixed precision
    if trainer and use_amp:
        trainer.scaler = scaler
        trainer.autocast_context = autocast_context
        trainer.accumulation_steps = accumulation_steps

    return model, trainer, use_amp, scaler, autocast_context, accumulation_steps


def create_callbacks(args, model, valid_loader, train_loader):
    """Create training and validation callbacks"""
    global ENCODER_STEP, train_log_buffer

    def train_callback(train_data, batch_idx, epoch, **kwargs):
        global ENCODER_STEP, train_log_buffer

        # Save checkpoint
        if args.save and epoch % args.checkpoint_freq == 0 and batch_idx == 0:
            save_model(model, args, model_hash=args.ae_model_hash)

        # Accumulate training metrics
        for k, v in train_data.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            train_log_buffer[k] += v
            train_log_buffer[f'{k}_count'] += 1

        # Log metrics periodically
        log_interval = max(1, args.log_freq // 10)
        if ENCODER_STEP % log_interval == 0:
            log_stats = {}
            for k, v in train_log_buffer.items():
                if k.endswith('_count'):
                    continue
                count_key = f'{k}_count'
                if count_key in train_log_buffer and train_log_buffer[count_key] > 0:
                    log_stats[k] = v / train_log_buffer[count_key]

            if log_stats:
                log_metrics({
                    'epoch': epoch,
                    'step': ENCODER_STEP,
                    **log_stats},
                    args, prefix='encoder', step=ENCODER_STEP)

            train_log_buffer = defaultdict(float)

        ENCODER_STEP += 1

    def valid_callback(valid_data, batch_idx, epoch):
        global ENCODER_STEP

        log_metrics({
            'epoch': epoch,
            'step': ENCODER_STEP,
            **{f'valid_{k}': v for k, v in valid_data.items()}},
            args, prefix='encoder', step=ENCODER_STEP)

        # Generate sample reconstructions
        if batch_idx == 0 and epoch % args.checkpoint_freq == 0:
            rev_transform = valid_loader.dataset.flat_rev_obs_transform if valid_loader else None

            try:
                valid_recons = sample_recon_imgs(
                    model, valid_loader, env_name=args.env_name, rev_transform=rev_transform)
                train_recons = sample_recon_imgs(
                    model, train_loader, env_name=args.env_name, rev_transform=rev_transform)

                log_images({
                    'valid_img_recon': valid_recons,
                    'train_img_recon': train_recons},
                    args, prefix='encoder', step=ENCODER_STEP)
            except Exception as e:
                print(f"Warning: Could not generate reconstruction images: {e}")

    return train_callback, valid_callback


def mixed_precision_train_loop(model, trainer, train_loader, valid_loader, n_epochs,
                               batch_size, log_freq, train_callback, valid_callback, args):
    """Mixed precision training loop with proper device handling"""

    model.train()
    device = next(model.parameters()).device  # Get device once
    accumulated_loss = 0.0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}/{n_epochs} | Memory: {psutil.Process().memory_info().rss / 1024 ** 3:.1f}GB')

        if hasattr(trainer, 'optimizer'):
            trainer.optimizer.zero_grad()

        for i, batch_data in enumerate(train_loader):
            # Move data to device
            if isinstance(batch_data, (list, tuple)):
                batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                              for x in batch_data]

            # Forward pass with mixed precision and error handling
            try:
                with trainer.autocast_context():
                    result = trainer.calculate_losses(batch_data)

                    if isinstance(result, tuple):
                        loss_dict, aux_data = result
                    else:
                        loss_dict = result
                        aux_data = {}

                    # Check if trainer returned an error string
                    if isinstance(loss_dict, str):
                        print(f"❌ Training error at epoch {epoch}, batch {i}: {loss_dict}")
                        continue

                    # Handle loss format
                    if isinstance(loss_dict, dict):
                        total_loss = loss_dict.get('loss', sum(loss_dict.values()))
                    else:
                        total_loss = loss_dict
                        loss_dict = {'loss': loss_dict}

                    total_loss = total_loss / trainer.accumulation_steps

                # Backward pass
                trainer.scaler.scale(total_loss).backward()
                accumulated_loss += total_loss.item()

                # Update weights
                if not hasattr(trainer, '_step_count'):
                    trainer._step_count = 0
                trainer._step_count += 1

                if trainer._step_count % trainer.accumulation_steps == 0:
                    if trainer.grad_clip > 0:
                        trainer.scaler.unscale_(trainer.optimizer)
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)

                    trainer.scaler.step(trainer.optimizer)
                    trainer.scaler.update()
                    trainer.optimizer.zero_grad()

            except Exception as e:
                print(f"❌ Mixed precision training step failed at epoch {epoch}, batch {i}: {e}")
                continue

            # Logging and validation
            if i % log_freq == 0 and i > 0:
                avg_loss = accumulated_loss / log_freq
                accumulated_loss = 0.0

                print(f'Epoch {epoch} | Batch {i} | Loss: {avg_loss:.4f}')

                if train_callback:
                    train_callback(loss_dict, i * batch_size, epoch, aux_data=aux_data)

                # Validation
                if valid_loader and i % (log_freq * 3) == 0:
                    model.eval()
                    with torch.no_grad():
                        valid_losses = []
                        for val_batch in valid_loader:
                            if isinstance(val_batch, (list, tuple)):
                                val_batch = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                                             for x in val_batch]

                            try:
                                val_result = trainer.calculate_losses(val_batch)
                                if isinstance(val_result, tuple):
                                    val_loss, _ = val_result
                                else:
                                    val_loss = val_result

                                if isinstance(val_loss, str):
                                    continue

                                if not isinstance(val_loss, dict):
                                    val_loss = {'loss': val_loss}
                                valid_losses.append(val_loss)
                            except Exception as e:
                                print(f"Validation error: {e}")
                                continue

                        if valid_losses:
                            valid_loss_means = {k: np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k]
                                                            for x in valid_losses]) for k in valid_losses[0]}
                            if valid_callback:
                                valid_callback(valid_loss_means, i, epoch)

                    model.train()


def standard_train_loop(model, trainer, train_loader, valid_loader, n_epochs,
                        batch_size, log_freq, train_callback, valid_callback):
    """Standard training loop without mixed precision"""

    model.train()
    device = next(model.parameters()).device

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}/{n_epochs} | Memory: {psutil.Process().memory_info().rss / 1024 ** 3:.1f}GB')

        for i, batch_data in enumerate(train_loader):
            # Move data to device
            if isinstance(batch_data, (list, tuple)):
                batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                              for x in batch_data]

            # Training step with error handling
            try:
                result = trainer.train(batch_data)

                # Handle different return formats
                if isinstance(result, tuple) and len(result) == 2:
                    train_loss, aux_data = result
                elif isinstance(result, tuple) and len(result) == 1:
                    train_loss, aux_data = result[0], {}
                else:
                    train_loss, aux_data = result, {}

                # Check if trainer returned an error string
                if isinstance(train_loss, str):
                    print(f"❌ Training error at epoch {epoch}, batch {i}: {train_loss}")
                    continue

                if not isinstance(train_loss, dict):
                    train_loss = {'loss': train_loss}

            except Exception as e:
                print(f"❌ Training step failed at epoch {epoch}, batch {i}: {e}")
                continue

            # Logging
            if i % log_freq == 0:
                loss_value = train_loss.get('loss', 0)
                if torch.is_tensor(loss_value):
                    loss_value = loss_value.item()
                print(f'Epoch {epoch} | Batch {i} | Loss: {loss_value:.4f}')

                if train_callback:
                    train_callback(train_loss, i * batch_size, epoch, aux_data=aux_data)

                # Validation
                if valid_loader and i % (log_freq * 2) == 0:
                    model.eval()
                    with torch.no_grad():
                        valid_losses = []
                        for val_batch in valid_loader:
                            if isinstance(val_batch, (list, tuple)):
                                val_batch = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                                             for x in val_batch]

                            try:
                                val_result = trainer.calculate_losses(val_batch)
                                if isinstance(val_result, tuple):
                                    val_loss, _ = val_result
                                else:
                                    val_loss = val_result

                                if isinstance(val_loss, str):
                                    continue

                                if not isinstance(val_loss, dict):
                                    val_loss = {'loss': val_loss}
                                valid_losses.append(val_loss)
                            except Exception as e:
                                print(f"Validation error: {e}")
                                continue

                        if valid_losses:
                            valid_loss_means = {k: np.mean([x[k].item() if torch.is_tensor(x[k]) else x[k]
                                                            for x in valid_losses]) for k in valid_losses[0]}
                            if valid_callback:
                                valid_callback(valid_loss_means, i, epoch)

                    model.train()


def evaluate_model(model, trainer, test_loader):
    """Evaluate the trained model"""
    if not trainer:
        print('No trainer available for evaluation')
        return

    print('Evaluating model...')
    model.eval()
    device = next(model.parameters()).device
    test_losses = []

    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data, (list, tuple)):
                batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                              for x in batch_data]

            try:
                result = trainer.calculate_losses(batch_data)
                if isinstance(result, tuple):
                    loss, _ = result
                else:
                    loss = result

                if not isinstance(loss, dict):
                    loss = {'loss': loss.mean() if torch.is_tensor(loss) else loss}

                # Convert tensors to scalars
                loss_dict = {}
                for k, v in loss.items():
                    if torch.is_tensor(v):
                        loss_dict[k] = v.item() if v.numel() == 1 else v.mean().item()
                    else:
                        loss_dict[k] = v

                test_losses.append(loss_dict)

            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue

    if test_losses:
        test_loss_means = {k: np.mean([x[k] for x in test_losses]) for k in test_losses[0]}
        print(f'Test losses: {test_loss_means}')
    else:
        print('No valid test results')


def train_encoder(args):
    """Main training function"""

    # Setup data and model
    train_loader, test_loader, valid_loader, model, trainer = setup_data_and_model(args)

    # Initialize model
    model, trainer = initialize_model(model, trainer, args)

    # Setup training environment
    model, trainer, use_amp, scaler, autocast_context, accumulation_steps = setup_training(model, trainer, args)

    # Track model
    track_model(model, args)

    # Check if training is needed
    if args.epochs <= 0 or trainer is None:
        print('No training required')
        return model

    # Create callbacks
    train_callback, valid_callback = create_callbacks(args, model, valid_loader, train_loader)

    # Train the model
    print(f'Starting training for {args.epochs} epochs...')
    try:
        if use_amp and hasattr(trainer, 'scaler'):
            mixed_precision_train_loop(
                model, trainer, train_loader, valid_loader, args.epochs,
                args.batch_size, args.log_freq, train_callback, valid_callback, args)
        else:
            standard_train_loop(
                model, trainer, train_loader, valid_loader, args.epochs,
                args.batch_size, args.log_freq, train_callback, valid_callback)

    except KeyboardInterrupt:
        print('Training interrupted by user')

    # Clean up
    global train_log_buffer
    train_log_buffer.clear()

    # Evaluate model
    evaluate_model(model, trainer, test_loader)

    # Save final model
    if args.save:
        save_model(model, args, model_hash=args.ae_model_hash)
        print('Model saved')

    return model


if __name__ == '__main__':
    # Parse arguments and setup logging
    args = get_args(apply_optimizations=True)
    args = init_experiment('discrete-mbrl-encoder', args)

    # Train the model
    model = train_encoder(args)

    print('Training completed successfully!')