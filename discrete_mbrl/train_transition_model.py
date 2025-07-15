import os
import sys
import time
import functools
import psutil
import numpy as np
from collections import defaultdict
from contextlib import nullcontext

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from shared.models import *
from shared.trainers import *
from data_helpers import *
from data_logging import *
from env_helpers import *
from training_helpers import *
from model_construction import *

# Global training state
TRANS_STEP = 0
train_log_buffer = defaultdict(list)
aux_log_buffer = defaultdict(list)


def debug_tensor_shapes(batch_data, step_name="", detailed=False):
    """Debug function to print tensor shapes"""
    print(f"\n=== Debug {step_name} ===")
    names = ["obs", "actions", "next_obs", "rewards", "dones"] + [f"extra_{i}" for i in range(len(batch_data) - 5)]

    for i, (name, data) in enumerate(zip(names, batch_data)):
        if torch.is_tensor(data):
            shape_str = f"{data.shape} {data.dtype}"
            if detailed:
                shape_str += f" device={data.device}"
            print(f"  {name}: {shape_str}")
        else:
            print(f"  {name}: {type(data)}")
    print("=" * 30)


def fix_batch_data_shapes(batch_data):
    """Fix tensor shapes for multi-step training data"""
    obs, actions, next_obs, rewards, dones = batch_data[:5]
    extra_data = batch_data[5:] if len(batch_data) > 5 else []

    # Handle multi-step format: [batch, n_steps, ...] -> [batch*n_steps, ...]
    if len(obs.shape) == 5:  # [batch, n_steps, channels, height, width]
        batch_size, n_steps = obs.shape[:2]

        obs = obs.reshape(batch_size * n_steps, *obs.shape[2:])
        next_obs = next_obs.reshape(batch_size * n_steps, *next_obs.shape[2:])

        # Special handling for actions - keep 2D for discrete models
        if len(actions.shape) == 2:  # [batch, n_steps] -> [batch*n_steps, 1]
            actions = actions.reshape(batch_size * n_steps, 1)
        else:  # [batch, n_steps, action_dim] -> [batch*n_steps, action_dim]
            actions = actions.reshape(batch_size * n_steps, *actions.shape[2:])

        # Handle rewards and dones - ensure they're at least 1D
        if len(rewards.shape) == 2:  # [batch, n_steps] -> [batch*n_steps]
            rewards = rewards.reshape(batch_size * n_steps)
        else:
            rewards = rewards.reshape(batch_size * n_steps, *rewards.shape[2:])

        if len(dones.shape) == 2:  # [batch, n_steps] -> [batch*n_steps]
            dones = dones.reshape(batch_size * n_steps)
        else:
            dones = dones.reshape(batch_size * n_steps, *dones.shape[2:])

        extra_data = [x.reshape(batch_size * n_steps, *x.shape[2:]) for x in extra_data]

    elif len(obs.shape) == 4 and len(actions.shape) >= 2:  # Mixed dimensions
        if len(actions.shape) == 2:
            batch_size, n_steps = actions.shape
        else:
            batch_size, n_steps = actions.shape[:2]

        if obs.shape[0] == batch_size * n_steps:
            # obs already flattened, flatten others
            if len(actions.shape) == 2:  # [batch*n_steps] -> [batch*n_steps, 1]
                actions = actions.reshape(-1, 1)
            else:
                actions = actions.reshape(batch_size * n_steps, *actions.shape[2:])

            if len(rewards.shape) >= 2:
                rewards = rewards.reshape(batch_size * n_steps, *rewards.shape[2:])
            if len(dones.shape) >= 2:
                dones = dones.reshape(batch_size * n_steps, *dones.shape[2:])

            if next_obs.shape[0] != batch_size * n_steps:
                next_obs = next_obs.reshape(batch_size * n_steps, *next_obs.shape[2:])

            extra_data = [x.reshape(batch_size * n_steps, *x.shape[2:])
                          if len(x.shape) > 2 else x for x in extra_data]

    # Final check: ensure actions are 2D for discrete models
    if len(actions.shape) == 1:
        actions = actions.unsqueeze(-1)

    return [obs, actions, next_obs, rewards, dones] + extra_data


def apply_training_optimizations(args):
    """Apply optimizations for transition model training"""
    # Adjust batch size for multi-step training
    if args.n_train_unroll > 1:
        original_batch_size = args.batch_size
        args.batch_size = max(256, args.batch_size // args.n_train_unroll)
        print(f"Adjusted batch size from {original_batch_size} to {args.batch_size} "
              f"for n_train_unroll={args.n_train_unroll}")

    # Setup gradient accumulation
    if not hasattr(args, 'accumulation_steps'):
        target_effective_batch = 4096
        args.accumulation_steps = max(1, target_effective_batch // args.batch_size)

    # Setup mixed precision
    if not hasattr(args, 'use_amp'):
        args.use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

    print(f"Training optimizations:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Mixed precision: {args.use_amp}")

    return args


def setup_training_components(args, encoder_model=None):
    """Setup data loaders, models, and trainers"""
    print('Loading data...')
    train_loader, test_loader, valid_loader = prepare_dataloaders(
        args.env_name, n=args.max_transitions, batch_size=args.batch_size,
        n_step=args.n_train_unroll, preprocess=args.preprocess, randomize=True,
        n_preload=args.n_preload, preload_all=args.preload_data,
        extra_buffer_keys=args.extra_buffer_keys,
        pin_memory=getattr(args, 'pin_memory', True),
        persistent_workers=getattr(args, 'persistent_workers', args.n_preload > 0),
        prefetch_factor=getattr(args, 'prefetch_factor', 2))

    print(f'Data split: {len(train_loader.dataset)}/{len(test_loader.dataset)}/{len(valid_loader.dataset)}')

    # Setup encoder
    if encoder_model is None:
        print('Constructing encoder...')
        sample_obs = next(iter(train_loader))[0][0]
        if args.n_train_unroll > 1:
            sample_obs = sample_obs[0]
        encoder_model = construct_ae_model(sample_obs.shape, args)[0]

    encoder_model = setup_efficient_model(encoder_model, args)
    freeze_model(encoder_model)
    encoder_model.eval()

    if hasattr(encoder_model, 'enable_sparsity'):
        encoder_model.enable_sparsity()

    # Setup transition model
    print('Constructing transition model...')
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    trans_model, trans_trainer = construct_trans_model(
        encoder_model, args, env.action_space, load=args.load)

    trans_model = setup_efficient_model(trans_model, args)

    print('Transition model:', trans_model)
    print(f'# Transition model params: {sum(x.numel() for x in trans_model.parameters()):,}')

    env.close()

    return train_loader, test_loader, valid_loader, encoder_model, trans_model, trans_trainer


def setup_mixed_precision_training(args, trans_trainer):
    """Setup mixed precision training components"""
    use_amp = getattr(args, 'use_amp', False) and args.device == 'cuda'
    scaler = GradScaler() if use_amp else None

    if use_amp:
        try:
            autocast_context = lambda: autocast('cuda')
        except TypeError:
            autocast_context = autocast
    else:
        autocast_context = nullcontext

    accumulation_steps = getattr(args, 'accumulation_steps', 1)

    # Setup trainer for mixed precision
    if use_amp:
        trans_trainer.scaler = scaler
        trans_trainer.autocast_context = autocast_context
        trans_trainer.accumulation_steps = accumulation_steps

    return use_amp, scaler, autocast_context, accumulation_steps


def transition_trainer_train_mixed_precision(trainer, batch_data, accumulation_steps):
    """Mixed precision training step with shape fixing"""
    # Fix tensor shapes for multi-step data
    batch_data = fix_batch_data_shapes(batch_data)

    # Forward pass with autocast
    with trainer.autocast_context():
        result = trainer.calculate_losses(batch_data)

        if isinstance(result, tuple) and len(result) == 2:
            loss, aux_data = result
        else:
            loss = result
            aux_data = {}

        # Validate loss
        if isinstance(loss, str):
            raise ValueError(f"Loss calculation error: {loss}")

        if isinstance(loss, dict):
            if 'loss' in loss:
                total_loss = loss['loss']
            else:
                # Sum component losses
                total_loss = 0
                for k, v in loss.items():
                    if 'loss' in k.lower() and isinstance(v, (torch.Tensor, float, int)):
                        total_loss += v

                if total_loss == 0:
                    raise ValueError(f"No valid loss components found: {list(loss.keys())}")

                loss['loss'] = total_loss

            total_loss = total_loss / accumulation_steps
        else:
            if not isinstance(loss, (torch.Tensor, float, int)):
                raise ValueError(f"Invalid loss type: {type(loss)}")
            total_loss = loss / accumulation_steps
            loss = {'loss': loss}

    # Backward pass
    trainer.scaler.scale(total_loss).backward()

    # Update weights
    if not hasattr(trainer, '_step_count'):
        trainer._step_count = 0
    trainer._step_count += 1

    if trainer._step_count % accumulation_steps == 0:
        if hasattr(trainer, 'grad_clip') and trainer.grad_clip > 0:
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)

        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.optimizer.zero_grad()

    return loss, aux_data


def create_training_callbacks(args, trans_model, valid_loader, train_loader):
    """Create training and validation callbacks"""
    global TRANS_STEP, train_log_buffer, aux_log_buffer

    def train_callback(train_data, batch_idx, epoch, aux_data=None):
        global TRANS_STEP, train_log_buffer, aux_log_buffer

        # Save checkpoint
        if args.save and epoch % args.checkpoint_freq == 0 and batch_idx == 0:
            save_model(trans_model, args, model_hash=args.trans_model_hash)

        # Accumulate metrics
        for k, v in train_data.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            train_log_buffer[k].append(v)

        if aux_data is not None:
            for k, v in aux_data.items():
                aux_log_buffer[k].append(v)

        # Log periodically
        log_interval = max(1, args.log_freq // 10)
        if TRANS_STEP % log_interval == 0:
            log_stats = {}
            for k, v in train_log_buffer.items():
                if len(v) > 0:
                    log_stats[f'train_{k}'] = sum(v) / len(v)
            for k, v in aux_log_buffer.items():
                if len(v) > 0:
                    log_stats[k] = sum(v) / len(v)

            if log_stats:
                log_metrics({
                    'epoch': epoch,
                    'step': TRANS_STEP,
                    **log_stats
                }, args, prefix='trans', step=TRANS_STEP)

            train_log_buffer = defaultdict(list)
            aux_log_buffer = defaultdict(list)

        TRANS_STEP += 1

    def valid_callback(valid_data, batch_idx, epoch):
        global TRANS_STEP

        log_metrics({
            'epoch': epoch,
            'step': TRANS_STEP,
            **{f'valid_{k}': v for k, v in valid_data.items()}
        }, args, prefix='trans', step=TRANS_STEP)

        # Generate visualizations periodically
        if batch_idx == 0 and epoch % args.checkpoint_freq == 0:
            try:
                rev_transform = valid_loader.dataset.flat_rev_obs_transform

                valid_recons = sample_recon_seqs(
                    args.encoder_model, trans_model, valid_loader, args.n_train_unroll,
                    env_name=args.env_name, rev_transform=rev_transform, gif_format=True)
                train_recons = sample_recon_seqs(
                    args.encoder_model, trans_model, train_loader, args.n_train_unroll,
                    env_name=args.env_name, rev_transform=rev_transform, gif_format=True)

                log_videos({
                    'valid_seq_recon': valid_recons,
                    'train_seq_recon': train_recons
                }, args, prefix='trans', step=TRANS_STEP)
            except Exception as e:
                print(f"Warning: Could not generate visualizations: {e}")

    return train_callback, valid_callback


def optimized_train_loop(model, trainer, train_loader, valid_loader, n_epochs,
                         batch_size, log_freq, train_callback, valid_callback,
                         test_func, use_amp, accumulation_steps):
    """Optimized training loop with proper error handling"""
    model.train()
    device = next(model.parameters()).device
    accumulated_loss = 0.0

    for epoch in range(n_epochs):
        print(f'Starting epoch #{epoch}')
        print(f'Memory usage: {psutil.Process().memory_info().rss / 1024 ** 3:.1f} GB')

        if hasattr(trainer, 'optimizer'):
            trainer.optimizer.zero_grad()

        for i, batch_data in enumerate(train_loader):
            try:
                # Move data to device
                if isinstance(batch_data, (list, tuple)):
                    batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                                  for x in batch_data]
                else:
                    batch_data = batch_data.to(device, non_blocking=True)

                # Debug first batch
                if i == 0 and epoch == 0:
                    print("Before shape fixing:")
                    debug_tensor_shapes(batch_data, "Original")

                # Training step
                if use_amp and hasattr(trainer, 'scaler'):
                    train_loss, aux_data = transition_trainer_train_mixed_precision(
                        trainer, batch_data, accumulation_steps)
                else:
                    # Standard training with shape fixing
                    batch_data = fix_batch_data_shapes(batch_data)

                    # Debug first batch after fixing
                    if i == 0 and epoch == 0:
                        print("After shape fixing:")
                        debug_tensor_shapes(batch_data, "Fixed")

                    train_loss, aux_data = trainer.train(batch_data)

                if not isinstance(train_loss, dict):
                    train_loss = {'loss': train_loss}

                # Accumulate loss
                loss_value = train_loss['loss'].item() if torch.is_tensor(train_loss['loss']) else train_loss['loss']
                accumulated_loss += loss_value

            except Exception as e:
                print(f"Error in training step epoch {epoch}, batch {i}: {e}")
                if i == 0:  # Print shapes for first batch to help debug
                    print(f"Original batch shapes: {[x.shape if torch.is_tensor(x) else type(x) for x in batch_data]}")
                    try:
                        fixed_batch = fix_batch_data_shapes(batch_data)
                        print(
                            f"Fixed batch shapes: {[x.shape if torch.is_tensor(x) else type(x) for x in fixed_batch]}")
                    except Exception as fix_error:
                        print(f"Error in shape fixing: {fix_error}")
                continue

            # Logging and validation
            if i % log_freq == 0 and i > 0:
                avg_loss = accumulated_loss / log_freq
                accumulated_loss = 0.0

                print(f'Epoch {epoch} | Batch {i} | Loss: {avg_loss:.4f}')

                if train_callback:
                    train_callback(train_loss, i * batch_size, epoch, aux_data=aux_data)

                # Validation
                if valid_loader and i % (log_freq * 3) == 0:
                    model.eval()
                    with torch.no_grad():
                        valid_losses = test_model_optimized(model, test_func, valid_loader, device)
                        if valid_losses:
                            valid_loss_means = {k: np.mean([x[k] for x in valid_losses])
                                                for k in valid_losses[0]}
                            if valid_callback:
                                valid_callback(valid_loss_means, i, epoch)
                    model.train()


def create_test_function(trainer, n_steps):
    """Create a test function that handles both flattened and n-step data"""

    def test_function(batch_data):
        # Check if data is already in n-step format or needs to be unflattened
        obs = batch_data[0]

        # If data is flattened, try to unflatten it for n-step evaluation
        if len(obs.shape) == 4:  # Flattened format [batch*n_steps, ...]
            try:
                # Try to reshape back to n-step format
                batch_size = obs.shape[0] // n_steps
                if batch_size * n_steps == obs.shape[0]:
                    # Reshape back to [batch, n_steps, ...]
                    unflattened_data = []
                    for tensor in batch_data:
                        if torch.is_tensor(tensor) and len(tensor.shape) >= 1:
                            if len(tensor.shape) == 1:  # rewards, dones
                                reshaped = tensor.reshape(batch_size, n_steps)
                            elif len(tensor.shape) == 2 and tensor.shape[1] == 1:  # actions
                                reshaped = tensor.reshape(batch_size, n_steps)
                            elif len(tensor.shape) == 4:  # observations
                                reshaped = tensor.reshape(batch_size, n_steps, *tensor.shape[1:])
                            else:
                                reshaped = tensor.reshape(batch_size, n_steps, *tensor.shape[1:])
                            unflattened_data.append(reshaped)
                        else:
                            unflattened_data.append(tensor)

                    # Use the unflattened data
                    return trainer.calculate_losses(unflattened_data, n=n_steps)
                else:
                    # Can't unflatten properly, use single-step evaluation
                    return trainer.calculate_losses(batch_data)
            except Exception:
                # Fallback to single-step if unflattening fails
                return trainer.calculate_losses(batch_data)
        else:
            # Data is already in multi-step format
            return trainer.calculate_losses(batch_data, n=n_steps)

    return test_function


def test_model_optimized(model, test_func, data_loader, device):
    """Optimized model testing with proper n-step handling"""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_data in data_loader:
            try:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                                  for x in batch_data]
                else:
                    batch_data = batch_data.to(device, non_blocking=True)

                # Use the test function as-is (it should handle shape conversion internally)
                result = test_func(batch_data)

                if isinstance(result, tuple):
                    loss, _ = result
                else:
                    loss = result

                if isinstance(loss, str):
                    continue

                if not isinstance(loss, dict):
                    loss = {'loss': loss.mean() if torch.is_tensor(loss) else loss}

                # Convert tensors to scalars
                loss_dict = {}
                for k, v in loss.items():
                    if torch.is_tensor(v):
                        loss_dict[k] = v.item() if v.numel() == 1 else v.mean().item()
                    else:
                        loss_dict[k] = v

                losses.append(loss_dict)

            except Exception as e:
                print(f"Error in test batch: {e}")
                continue

    return losses


def train_trans_model(args, encoder_model=None):
    """Main transition model training function"""
    # Setup logging
    import_logger(args)

    # Apply optimizations
    args = apply_training_optimizations(args)

    # Setup GPU optimizations
    optimize_gpu_memory()

    # Setup components
    train_loader, test_loader, valid_loader, encoder_model, trans_model, trans_trainer = \
        setup_training_components(args, encoder_model)

    # Store encoder in args for callbacks
    args.encoder_model = encoder_model

    # Setup mixed precision
    use_amp, scaler, autocast_context, accumulation_steps = \
        setup_mixed_precision_training(args, trans_trainer)

    # Setup test function that can handle both formats
    test_func = create_test_function(trans_trainer, args.n_train_unroll)

    # Update the trainer's train method for n-step training
    trans_trainer.train = functools.partial(trans_trainer.train, n=args.n_train_unroll)

    # Update params and track model
    update_params(args)
    track_model(trans_model, args)

    # Create callbacks
    train_callback, valid_callback = create_training_callbacks(
        args, trans_model, valid_loader, train_loader)

    # Training
    n_epochs = args.trans_epochs if args.trans_epochs is not None else args.epochs

    print(f'Starting training for {n_epochs} epochs...')
    try:
        optimized_train_loop(
            trans_model, trans_trainer, train_loader, valid_loader, n_epochs,
            args.batch_size, args.log_freq, train_callback, valid_callback,
            test_func, use_amp, accumulation_steps)
    except KeyboardInterrupt:
        print('Training interrupted')

    # Cleanup
    global train_log_buffer, aux_log_buffer
    train_log_buffer.clear()
    aux_log_buffer.clear()

    # Final evaluation
    print('Starting model evaluation...')
    test_losses = test_model_optimized(trans_model, test_func, test_loader, args.device)
    if test_losses:
        test_loss_means = {k: np.mean([d[k] for d in test_losses]) for k in test_losses[0].keys()}
        print(f'Test losses: {test_loss_means}')

    # Save model
    if args.save:
        save_model(trans_model, args, model_hash=args.trans_model_hash)
        print('Transition model saved')

    return trans_model


if __name__ == '__main__':
    # Parse args
    args = get_args(apply_optimizations=True)

    # Setup logging
    args = init_experiment('discrete-mbrl-transition-models', args)

    # Train model
    trans_model = train_trans_model(args)

    print('Training completed successfully!')