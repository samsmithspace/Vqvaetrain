#train_transition_model.py
from collections import defaultdict
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import functools
import torch

# Fix the autocast import for newer PyTorch versions
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

from shared.models import *
from shared.trainers import *
from data_helpers import *
from data_logging import *
from env_helpers import *
from training_helpers import *
from model_construction import *

TRANS_STEP = 0
train_log_buffer = defaultdict(list)
aux_log_buffer = defaultdict(list)


def train_trans_model(args, encoder_model=None):
    """Optimized transition model training with GPU utilization improvements"""

    import_logger(args)

    # Setup GPU optimizations early
    optimize_gpu_memory()

    print('Loading data with optimizations...')
    # Data shape: (5, batch_size, n_steps, ...)
    train_loader, test_loader, valid_loader = prepare_dataloaders(
        args.env_name, n=args.max_transitions, batch_size=args.batch_size,
        n_step=args.n_train_unroll, preprocess=args.preprocess, randomize=True,
        n_preload=args.n_preload, preload_all=args.preload_data,
        extra_buffer_keys=args.extra_buffer_keys,
        pin_memory=getattr(args, 'pin_memory', True),
        persistent_workers=getattr(args, 'persistent_workers', args.n_preload > 0),
        prefetch_factor=getattr(args, 'prefetch_factor', 2))

    print(f'Data split: {len(train_loader.dataset)}/{len(test_loader.dataset)}/{len(valid_loader.dataset)}')
    print(f'Batch size: {args.batch_size}, N-step unroll: {args.n_train_unroll}, Workers: {args.n_preload}')

    if encoder_model is None:
        print('Constructing encoder...')
        sample_obs = next(iter(train_loader))[0][0]
        if args.n_train_unroll > 1:
            sample_obs = sample_obs[0]
        encoder_model = construct_ae_model(
            sample_obs.shape, args)[0]

    # Apply GPU optimizations to encoder
    encoder_model = setup_efficient_model(encoder_model, args)
    freeze_model(encoder_model)
    encoder_model.eval()

    if hasattr(encoder_model, 'enable_sparsity'):
        encoder_model.enable_sparsity()

    print('Constructing transition model...')
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    trans_model, trans_trainer = construct_trans_model(
        encoder_model, args, env.action_space, load=args.load)
    print('Transition model:', trans_model)

    # Apply GPU optimizations to transition model
    trans_model = setup_efficient_model(trans_model, args)

    # Setup mixed precision training
    use_amp = getattr(args, 'use_amp', False) and args.device == 'cuda'
    scaler = GradScaler() if use_amp else None
    # Fix autocast context for newer PyTorch versions
    if use_amp:
        try:
            autocast_context = lambda: autocast('cuda')
        except TypeError:
            autocast_context = autocast
    else:
        autocast_context = nullcontext
    accumulation_steps = getattr(args, 'accumulation_steps', 1)

    print(f'Using mixed precision: {use_amp}')
    print(f'Gradient accumulation steps: {accumulation_steps}')
    print('# Transition model params:', sum([x.numel() for x in trans_model.parameters()]))

    # Setup transition trainer for optimized training
    test_func = functools.partial(
        trans_trainer.calculate_losses, n=args.n_train_unroll)
    trans_trainer.train = functools.partial(trans_trainer.train, n=args.n_train_unroll)

    # Setup trainer for mixed precision
    if use_amp:
        trans_trainer.scaler = scaler
        trans_trainer.autocast_context = autocast_context
        trans_trainer.accumulation_steps = accumulation_steps

    update_params(args)
    track_model(trans_model, args)

    def train_callback(train_data, batch_idx, epoch, aux_data=None):
        global TRANS_STEP, train_log_buffer, aux_log_buffer
        if args.save and epoch % args.checkpoint_freq == 0 and batch_idx == 0:
            save_model(trans_model, args, model_hash=args.trans_model_hash)

        for k, v in train_data.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            train_log_buffer[k].append(v)
        if aux_data is not None:
            for k, v in aux_data.items():
                aux_log_buffer[k].append(v)

        # Reduced logging frequency for better GPU utilization
        log_interval = max(1, (args.log_freq // 10))
        if TRANS_STEP % log_interval == 0:
            log_stats = {}
            for k, v in train_log_buffer.items():
                if len(v) > 0:
                    log_stats[f'train_{k}'] = sum(v) / len(v)
            for k, v in aux_log_buffer.items():
                if len(v) > 0:
                    log_stats[k] = sum(v) / len(v)

            if log_stats:  # Only log if we have stats
                log_metrics({
                    'epoch': epoch,
                    'step': TRANS_STEP,
                    **log_stats},
                    args, prefix='trans', step=TRANS_STEP)

            train_log_buffer = defaultdict(list)
            aux_log_buffer = defaultdict(list)
        TRANS_STEP += 1

    # For reversing observation transformations
    rev_transform = valid_loader.dataset.flat_rev_obs_transform

    def valid_callback(valid_data, batch_idx, epoch):
        global TRANS_STEP
        log_metrics({
            'epoch': epoch,
            'step': TRANS_STEP,
            **{f'valid_{k}': v for k, v in valid_data.items()}},
            args, prefix='trans', step=TRANS_STEP)

        # Less frequent visualization generation to keep GPU busy
        if batch_idx == 0 and epoch % args.checkpoint_freq == 0:
            valid_recons = sample_recon_seqs(
                encoder_model, trans_model, valid_loader, args.n_train_unroll,
                env_name=args.env_name, rev_transform=rev_transform, gif_format=True)
            train_recons = sample_recon_seqs(
                encoder_model, trans_model, train_loader, args.n_train_unroll,
                env_name=args.env_name, rev_transform=rev_transform, gif_format=True)
            log_videos({
                'valid_seq_recon': valid_recons,
                'train_seq_recon': train_recons},
                args, prefix='trans', step=TRANS_STEP)

    n_epochs = args.trans_epochs if args.trans_epochs is not None else args.epochs
    try:
        # Use optimized training loop
        optimized_transition_train_loop(
            trans_model, trans_trainer, train_loader, valid_loader, n_epochs,
            args.batch_size, args.log_freq, callback=train_callback,
            valid_callback=valid_callback, test_func=test_func, use_amp=use_amp,
            accumulation_steps=accumulation_steps)
    except KeyboardInterrupt:
        print('Stopping training')

    # Get rid of any remaining log data
    global train_log_buffer
    del train_log_buffer

    # Test the model
    print('Starting model evaluation...')
    test_losses = test_model_optimized(trans_model, test_func, test_loader, args.device)
    if test_losses:
        test_losses = {k: np.mean([d[k] for d in test_losses]) for k in test_losses[0].keys()}
        print(f'Transition model test losses: {test_losses}')

    if args.save:
        save_model(trans_model, args, model_hash=args.trans_model_hash)
        print('Transition model saved')

    return trans_model


def optimized_transition_train_loop(model, trainer, train_loader, valid_loader=None, n_epochs=1,
                                    batch_size=128, log_freq=100, seed=0, callback=None,
                                    valid_callback=None, test_func=None, use_amp=True,
                                    accumulation_steps=1):
    """
    Optimized training loop specifically for transition models with mixed precision,
    gradient accumulation, and reduced CPU-GPU synchronization.
    """
    torch.manual_seed(seed)
    model.train()

    # Setup mixed precision training
    scaler = GradScaler() if use_amp else None
    # Fix autocast context for newer PyTorch versions
    if use_amp:
        try:
            autocast_context = lambda: autocast('cuda')
        except TypeError:
            autocast_context = autocast
    else:
        autocast_context = nullcontext

    # Pre-allocate tensors to reduce memory allocation overhead
    device = next(model.parameters()).device

    train_losses = []
    accumulated_loss = 0.0

    for epoch in range(n_epochs):
        print(f'Starting epoch #{epoch}')
        print('Memory usage: {:.1f} GB'.format(
            psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3))

        # Reset accumulated gradients at start of epoch
        if hasattr(trainer, 'optimizer'):
            trainer.optimizer.zero_grad()

        for i, batch_data in enumerate(train_loader):
            # Move data to GPU with non_blocking transfer
            if isinstance(batch_data, (list, tuple)):
                batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                              for x in batch_data]
            else:
                batch_data = batch_data.to(device, non_blocking=True)

            with autocast_context():
                # Modified trainer call for mixed precision
                if hasattr(trainer, 'scaler') and trainer.scaler is not None:
                    train_loss, aux_data = transition_trainer_train_mixed_precision(
                        trainer, batch_data, accumulation_steps)
                else:
                    train_loss, aux_data = trainer.train(batch_data)

                if not isinstance(train_loss, dict):
                    train_loss = {'loss': train_loss}

            # Accumulate losses for logging (avoid frequent CPU-GPU sync)
            loss_value = train_loss['loss'].item() if torch.is_tensor(train_loss['loss']) else train_loss['loss']
            accumulated_loss += loss_value

            # Reduced frequency operations
            if i % log_freq == 0 and i > 0:
                # Only sync when necessary for logging
                avg_loss = accumulated_loss / log_freq
                accumulated_loss = 0.0

                update_str = f'Epoch {epoch} | Batch {i} | train_loss: {avg_loss:.3f}'

                # Less frequent validation to keep GPU busy
                if valid_loader is not None and i % (log_freq * 3) == 0:
                    model.eval()
                    with torch.no_grad():
                        test_func = test_func or trainer.calculate_losses
                        valid_losses = test_model_optimized(model, test_func, valid_loader, device)
                        if valid_losses:
                            valid_loss_means = {k: np.mean([x[k] for x in valid_losses])
                                                for k in valid_losses[0]}
                            if valid_callback:
                                valid_callback(valid_loss_means, i, epoch)
                            for k, v in valid_loss_means.items():
                                update_str += f' | valid_{k}: {v:.3f}'
                    model.train()

                print(update_str)

                if callback:
                    callback(train_loss, i * batch_size, epoch, aux_data=aux_data)


def transition_trainer_train_mixed_precision(trainer, batch_data, accumulation_steps):
    """Modified trainer.train method for transition models with mixed precision and gradient accumulation"""

    # Forward pass with autocast
    with trainer.autocast_context():
        result = trainer.calculate_losses(batch_data)

        # Handle different return types from calculate_losses
        if isinstance(result, tuple) and len(result) == 2:
            loss, aux_data = result
        else:
            loss = result
            aux_data = {}

        # Validate that loss is numeric
        if isinstance(loss, str):
            raise ValueError(f"Loss calculation returned an error: {loss}")

        if isinstance(loss, dict):
            # Handle different loss dictionary formats
            if 'loss' in loss:
                # Standard case: single 'loss' key
                total_loss = loss['loss']
            else:
                # Handle component losses (e.g., VQ-VAE, transition models with multiple losses)
                total_loss = 0
                loss_components = []
                for k, v in loss.items():
                    if 'loss' in k.lower() and isinstance(v, (torch.Tensor, float, int)):
                        total_loss += v
                        loss_components.append(k)

                if total_loss == 0 or len(loss_components) == 0:
                    raise ValueError(f"No valid loss components found in loss dict. Got keys: {list(loss.keys())}")

                # Add the total loss to the dict for logging
                loss['loss'] = total_loss

            total_loss = total_loss / accumulation_steps
        else:
            # Validate that loss is numeric
            if not isinstance(loss, (torch.Tensor, float, int)):
                raise ValueError(f"Loss must be numeric, got {type(loss)}: {loss}")
            total_loss = loss / accumulation_steps
            loss = {'loss': loss}

    # Backward pass with gradient scaling
    trainer.scaler.scale(total_loss).backward()

    # Update weights every accumulation_steps
    if hasattr(trainer, '_step_count'):
        trainer._step_count += 1
    else:
        trainer._step_count = 1

    if trainer._step_count % accumulation_steps == 0:
        # Gradient clipping before step
        if hasattr(trainer, 'grad_clip') and trainer.grad_clip > 0:
            trainer.scaler.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)

        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.optimizer.zero_grad()

    return loss, aux_data


def test_model_optimized(model, test_func, data_loader, device):
    """Optimized model testing with batched processing for transition models"""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_data in data_loader:
            try:
                # Efficient data transfer
                if isinstance(batch_data, (list, tuple)):
                    batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                                  for x in batch_data]
                else:
                    batch_data = batch_data.to(device, non_blocking=True)

                result = test_func(batch_data)

                # Handle different return types
                if isinstance(result, tuple) and len(result) == 2:
                    loss, _ = result
                else:
                    loss = result

                # Validate loss
                if isinstance(loss, str):
                    print(f"Warning: Test function returned error: {loss}")
                    continue

                if not isinstance(loss, dict):
                    if isinstance(loss, (torch.Tensor, float, int)):
                        loss = {'loss': loss.mean() if torch.is_tensor(loss) else loss}
                    else:
                        print(f"Warning: Invalid loss type {type(loss)}, skipping batch")
                        continue
                else:
                    # Handle component loss dictionaries
                    if 'loss' not in loss:
                        # Sum up component losses to create total loss
                        total_loss = 0
                        for k, v in loss.items():
                            if 'loss' in k.lower() and isinstance(v, (torch.Tensor, float, int)):
                                total_loss += v.mean() if torch.is_tensor(v) else v
                        loss['loss'] = total_loss

                # Convert to CPU only when needed for storage
                loss_dict = {}
                for k, v in loss.items():
                    if torch.is_tensor(v):
                        if v.numel() == 1:
                            loss_dict[k] = v.item()
                        else:
                            loss_dict[k] = v.mean().item()
                    elif isinstance(v, (float, int)):
                        loss_dict[k] = v
                    else:
                        print(f"Warning: Invalid loss value type {type(v)} for key {k}, skipping")
                        continue

                losses.append(loss_dict)

            except Exception as e:
                print(f"Error in test batch: {e}")
                continue

    return losses


# Additional utility functions for monitoring
def monitor_gpu_utilization():
    """Monitor GPU utilization during training"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
            print(
                f"GPU Util: {gpu_util}% | Memory: {mem_used}/{mem_total} MB ({100 * int(mem_used) / int(mem_total):.1f}%)")
    except:
        pass


if __name__ == '__main__':
    # Parse args with optimizations
    args = get_args(apply_optimizations=True)
    # Setup logging
    args = init_experiment('discrete-mbrl-transition-models', args)
    # Train and test the model
    trans_model = train_trans_model(args)
    # Save the model
    if args.save:
        save_model(trans_model, args, model_hash=args.trans_model_hash)
        print('Model saved')