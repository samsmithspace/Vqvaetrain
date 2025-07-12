from collections import defaultdict
import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

from shared.models import *
from shared.trainers import *
from data_helpers import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from data_logging import *

ENCODER_STEP = 0
train_log_buffer = defaultdict(float)


def train_encoder(args):
    """Optimized encoder training with GPU utilization improvements"""

    # Setup GPU optimizations early
    optimize_gpu_memory()

    print('Loading data with optimizations...')
    if args.unique_data:
        # In this case there is no valid/test data split
        train_loader = test_loader = \
            prepare_unique_obs_dataloader(args, randomize=True)
        valid_loader = None
    else:
        # Use optimized data loading parameters
        train_loader, test_loader, valid_loader = prepare_dataloaders(
            args.env_name, n=args.max_transitions, batch_size=args.batch_size,
            preprocess=args.preprocess, randomize=True, n_preload=args.n_preload,
            preload_all=args.preload_data, extra_buffer_keys=args.extra_buffer_keys,
            pin_memory=getattr(args, 'pin_memory', True),
            persistent_workers=getattr(args, 'persistent_workers', args.n_preload > 0),
            prefetch_factor=getattr(args, 'prefetch_factor', 2))

    valid_len = len(valid_loader.dataset) if valid_loader is not None else 0
    print(f'Data split: {len(train_loader.dataset)}/{len(test_loader.dataset)}/{valid_len}')
    print(f'Batch size: {args.batch_size}, Workers: {args.n_preload}')

    print('Constructing model...')
    pre_sample_time = time.time()
    sample_obs = next(iter(train_loader))[0]
    print('Sample time:', time.time() - pre_sample_time)
    print('Sample shape:', sample_obs.shape)

    model, trainer = construct_ae_model(
        sample_obs.shape[1:], args, load=args.load)
    update_params(args)

    # Apply GPU optimizations to model
    model = setup_efficient_model(model, args)

    # Setup mixed precision training
    use_amp = getattr(args, 'use_amp', False) and args.device == 'cuda'
    scaler = GradScaler() if use_amp else None
    autocast_context = autocast if use_amp else nullcontext
    accumulation_steps = getattr(args, 'accumulation_steps', 1)

    print(f'Using mixed precision: {use_amp}')
    print(f'Gradient accumulation steps: {accumulation_steps}')
    print('# Params:', sum([x.numel() for x in model.parameters()]))
    print(model)

    track_model(model, args)

    if hasattr(model, 'disable_sparsity'):
        model.disable_sparsity()

    # Check if training is needed
    if args.epochs <= 0 or trainer is None:
        print('No training required for this model type or epochs=0')
        return model

    trainer.recon_loss_clip = args.recon_loss_clip

    # Setup trainer for mixed precision
    if trainer is not None and use_amp:
        trainer.scaler = scaler
        trainer.autocast_context = autocast_context
        trainer.accumulation_steps = accumulation_steps

    def train_callback(train_data, batch_idx, epoch, **kwargs):
        global ENCODER_STEP, train_log_buffer
        if args.save and epoch % args.checkpoint_freq == 0 and batch_idx == 0:
            save_model(model, args, model_hash=args.ae_model_hash)

        for k, v in train_data.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            train_log_buffer[k] += v
            train_log_buffer[f'{k}_count'] += 1

        # Reduced logging frequency for better GPU utilization
        log_interval = max(1, (args.log_freq // 10))
        if ENCODER_STEP % log_interval == 0:
            log_stats = {}
            for k, v in train_log_buffer.items():
                if k.endswith('_count'):
                    continue
                count_key = f'{k}_count'
                if count_key in train_log_buffer and train_log_buffer[count_key] > 0:
                    log_stats[k] = v / train_log_buffer[count_key]

            if log_stats:  # Only log if we have stats
                log_metrics({
                    'epoch': epoch,
                    'step': ENCODER_STEP,
                    **log_stats},
                    args, prefix='encoder', step=ENCODER_STEP)
            train_log_buffer = defaultdict(float)

        ENCODER_STEP += 1

    env = make_env(args.env_name, max_steps=args.env_max_steps)
    # For reversing observation transformations
    rev_transform = valid_loader.dataset.flat_rev_obs_transform if valid_loader else None

    def valid_callback(valid_data, batch_idx, epoch):
        global ENCODER_STEP
        log_metrics({
            'epoch': epoch,
            'step': ENCODER_STEP,
            **{f'valid_{k}': v for k, v in valid_data.items()}},
            args, prefix='encoder', step=ENCODER_STEP)

        if batch_idx == 0 and epoch % args.checkpoint_freq == 0:
            # Generate sample reconstructions less frequently
            valid_recons = sample_recon_imgs(
                model, valid_loader, env_name=args.env_name, rev_transform=rev_transform)
            train_recons = sample_recon_imgs(
                model, train_loader, env_name=args.env_name, rev_transform=rev_transform)
            log_images({
                'valid_img_recon': valid_recons,
                'train_img_recon': train_recons},
                args, prefix='encoder', step=ENCODER_STEP)

    try:
        # Use optimized training loop
        optimized_train_loop(
            model, trainer, train_loader, valid_loader, args.epochs,
            args.batch_size, args.log_freq, callback=train_callback,
            valid_callback=valid_callback, use_amp=use_amp,
            accumulation_steps=accumulation_steps)
    except KeyboardInterrupt:
        print('Stopping training')

    # Get rid of any remaining log data
    global train_log_buffer
    del train_log_buffer

    # Test the model (only if trainer exists)
    if trainer is not None:
        print('Starting model evaluation...')
        test_losses = test_model_optimized(model, trainer.calculate_losses, test_loader, args.device)
        test_losses = {k: np.mean([d[k] for d in test_losses]) for k in test_losses[0].keys()}
        print(f'Encoder test loss: {test_losses}')
    else:
        print('Skipping model evaluation (no trainer for this model type)')

    if args.save:
        save_model(model, args, model_hash=args.ae_model_hash)
        print('Encoder model saved')

    return model


def optimized_train_loop(model, trainer, train_loader, valid_loader=None, n_epochs=1,
                         batch_size=128, log_freq=100, seed=0, callback=None,
                         valid_callback=None, test_func=None, use_amp=True,
                         accumulation_steps=1):
    """
    Optimized training loop with mixed precision, gradient accumulation,
    and reduced CPU-GPU synchronization.
    """
    torch.manual_seed(seed)
    model.train()

    # Setup mixed precision training
    scaler = GradScaler() if use_amp else None
    autocast_context = autocast if use_amp else nullcontext

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
                    train_loss, aux_data = trainer_train_mixed_precision(
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


def trainer_train_mixed_precision(trainer, batch_data, accumulation_steps):
    """Modified trainer.train method for mixed precision and gradient accumulation"""

    # Forward pass with autocast
    with trainer.autocast_context():
        loss, aux_data = trainer.calculate_losses(batch_data)
        if isinstance(loss, dict):
            total_loss = loss['loss'] / accumulation_steps
        else:
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
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.optimizer.zero_grad()

    return loss, aux_data


def test_model_optimized(model, test_func, data_loader, device):
    """Optimized model testing with batched processing"""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_data in data_loader:
            # Efficient data transfer
            if isinstance(batch_data, (list, tuple)):
                batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                              for x in batch_data]
            else:
                batch_data = batch_data.to(device, non_blocking=True)

            loss = test_func(batch_data)
            if not isinstance(loss, dict):
                loss = {'loss': loss.mean()}

            # Convert to CPU only when needed for storage
            losses.append({k: v.item() if torch.is_tensor(v) else v for k, v in loss.items()})

    return losses


if __name__ == '__main__':
    # Parse args with optimizations
    args = get_args(apply_optimizations=True)
    # Setup logging
    args = init_experiment('discrete-mbrl-encoder', args)
    # Train and test the model
    model = train_encoder(args)
    # Save the model
    if args.save:
        save_model(model, args, model_hash=args.ae_model_hash)
        print('Model saved')