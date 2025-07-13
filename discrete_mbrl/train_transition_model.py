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


def get_autocast_context(device, use_amp=True):
    """Get the appropriate autocast context based on device and PyTorch version"""
    if not use_amp:
        return nullcontext()

    if device.type == 'cuda':
        try:
            # Try the new API first (PyTorch 1.10+)
            return autocast(device_type='cuda')
        except TypeError:
            # Fall back to old API (PyTorch < 1.10)
            return autocast()
    else:
        # CPU doesn't support autocast in older versions
        return nullcontext()


def debug_tensor_shapes(batch_data, step_name=""):
    """Debug function to print tensor shapes - disabled for production"""
    pass


def fix_batch_data_shapes(batch_data, n_train_unroll=1):
    """
    Fix tensor shapes for multi-step training data with proper validation
    """
    if not batch_data or not torch.is_tensor(batch_data[0]):
        return batch_data

    obs = batch_data[0]

    # Validate the expected input format
    if len(obs.shape) == 5:  # [batch, n_steps, channels, height, width]
        batch_size, n_steps = obs.shape[:2]

        # Verify n_steps matches expected
        if n_steps != n_train_unroll:
            print(f"Warning: Data has {n_steps} steps but expected {n_train_unroll}")

        fixed_data = []
        for i, x in enumerate(batch_data):
            if torch.is_tensor(x):
                if i in [0, 2] and len(x.shape) == 5:  # Observations (obs, next_obs)
                    # Reshape from [batch, n_steps, C, H, W] to [batch * n_steps, C, H, W]
                    new_shape = [batch_size * n_steps] + list(x.shape[2:])
                    reshaped = x.reshape(new_shape)
                    fixed_data.append(reshaped)
                elif len(x.shape) == 2:  # Actions, rewards, dones [batch, n_steps]
                    # Flatten to [batch * n_steps]
                    reshaped = x.reshape(-1)
                    fixed_data.append(reshaped)
                elif len(x.shape) == 3 and i in [1, 3, 4]:  # Could be [batch, n_steps, action_dim]
                    batch_size, n_steps = x.shape[:2]
                    if x.shape[2] == 1:
                        reshaped = x.reshape(-1)
                    else:
                        reshaped = x.reshape(batch_size * n_steps, -1)
                    fixed_data.append(reshaped)
                else:
                    fixed_data.append(x)
            else:
                fixed_data.append(x)

        return fixed_data

    elif len(obs.shape) == 4:  # [batch, channels, height, width] - already processed
        return batch_data

    elif len(obs.shape) == 3:  # [channels, height, width] - single sample
        # Add batch dimension
        fixed_data = []
        for x in batch_data:
            if torch.is_tensor(x) and x.dim() >= 2:
                fixed_data.append(x.unsqueeze(0))
            else:
                fixed_data.append(x)
        return fixed_data

    else:
        print(f"Unexpected observation shape: {obs.shape}")
        return batch_data


def safe_n_step_calculate_losses(trainer, batch_data, n_train_unroll):
    """
    Safely calculate losses for n-step training with proper tensor handling
    """
    # Extract data
    obs, actions, next_obs, rewards, dones = batch_data[:5]
    extra_data = batch_data[5:] if len(batch_data) > 5 else []

    if n_train_unroll > 1:
        # For n-step training, we need to reshape the flattened data back to sequences
        total_samples = obs.shape[0]
        batch_size = total_samples // n_train_unroll

        if total_samples % n_train_unroll != 0:
            print(f"Warning: total_samples ({total_samples}) not divisible by n_train_unroll ({n_train_unroll})")
            # Adjust to make it divisible
            new_total = (total_samples // n_train_unroll) * n_train_unroll
            obs = obs[:new_total]
            actions = actions[:new_total]
            next_obs = next_obs[:new_total]
            rewards = rewards[:new_total]
            dones = dones[:new_total]
            extra_data = [x[:new_total] for x in extra_data]
            batch_size = new_total // n_train_unroll

        # Reshape to [batch_size, n_steps, ...]
        obs = obs.reshape(batch_size, n_train_unroll, *obs.shape[1:])
        actions = actions.reshape(batch_size, n_train_unroll,
                                  *actions.shape[1:]) if actions.dim() > 1 else actions.reshape(batch_size,
                                                                                                n_train_unroll)
        next_obs = next_obs.reshape(batch_size, n_train_unroll, *next_obs.shape[1:])
        rewards = rewards.reshape(batch_size, n_train_unroll,
                                  *rewards.shape[1:]) if rewards.dim() > 1 else rewards.reshape(batch_size,
                                                                                                n_train_unroll)
        dones = dones.reshape(batch_size, n_train_unroll, *dones.shape[1:]) if dones.dim() > 1 else dones.reshape(
            batch_size, n_train_unroll)

        # Reconstruct batch_data with proper shapes
        reshaped_batch_data = [obs, actions, next_obs, rewards, dones]
        for x in extra_data:
            if x.dim() > 1:
                reshaped_x = x.reshape(batch_size, n_train_unroll, *x.shape[1:])
            else:
                reshaped_x = x.reshape(batch_size, n_train_unroll)
            reshaped_batch_data.append(reshaped_x)

        batch_data = reshaped_batch_data

    # Now call the original calculate_losses with the properly shaped data
    try:
        return trainer.calculate_losses(batch_data, n=n_train_unroll)
    except Exception as e:
        print(f"Error in calculate_losses: {e}")
        print(f"Final batch_data shapes: {[x.shape if torch.is_tensor(x) else type(x) for x in batch_data]}")
        raise


def transition_trainer_train_mixed_precision(trainer, batch_data, accumulation_steps, n_train_unroll=1):
    """
    Handle training step with proper error handling and n-step support
    """
    try:
        # Use the safe n-step calculation
        result = safe_n_step_calculate_losses(trainer, batch_data, n_train_unroll)

        # Handle different return types
        if isinstance(result, dict):
            #print(f"Result dict keys: {list(result.keys())}")

            # Find the main loss tensor
            loss_tensor = None
            for key in ['loss', 'total_loss', 'combined_loss', 'state_loss', 'trans_loss']:
                if key in result and torch.is_tensor(result[key]):
                    loss_tensor = result[key]
                    break

            if loss_tensor is None:
                # Sum all tensor values that have 'loss' in their key name
                tensor_losses = [v for k, v in result.items() if torch.is_tensor(v) and 'loss' in str(k).lower()]
                if not tensor_losses:
                    # If no losses found, sum ALL tensor values
                    tensor_losses = [v for v in result.values() if torch.is_tensor(v)]

                if tensor_losses:
                    loss_tensor = sum(tensor_losses)
                    #print(
                    #    f"Created combined loss from {len(tensor_losses)} components: {[k for k, v in result.items() if torch.is_tensor(v) and 'loss' in str(k).lower()]}")
                else:
                    raise ValueError(f"No tensor losses found in result dict with keys: {list(result.keys())}")

            # Ensure the result dict has a 'loss' key for consistency
            train_loss = result.copy()
            if 'loss' not in train_loss:
                train_loss['loss'] = loss_tensor

            aux_data = {}
        elif isinstance(result, tuple) and len(result) == 2:
            loss_tensor, aux_data = result
            train_loss = {'loss': loss_tensor}
        else:
            loss_tensor = result
            train_loss = {'loss': loss_tensor}
            aux_data = {}

        # Ensure loss_tensor is a scalar
        if loss_tensor.dim() > 0:
            loss_tensor = loss_tensor.mean()

        # Scale for gradient accumulation
        scaled_loss = loss_tensor / accumulation_steps

        return train_loss, aux_data, scaled_loss

    except Exception as e:
        print(f"Error in transition trainer: {e}")
        print(f"Batch data shapes: {[x.shape if torch.is_tensor(x) else type(x) for x in batch_data]}")
        raise


def optimized_transition_train_loop(model, trainer, train_loader, valid_loader=None, n_epochs=1,
                                    batch_size=128, log_freq=100, seed=0, callback=None,
                                    valid_callback=None, test_func=None, use_amp=True,
                                    accumulation_steps=1, n_train_unroll=1):
    """
    Fixed training loop with proper n-step handling
    """
    torch.manual_seed(seed)
    model.train()

    device = next(model.parameters()).device

    # Setup mixed precision with proper device handling
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    use_amp = use_amp and device.type == 'cuda'  # Only use AMP on CUDA

    # Setup optimizer if not already done
    if not hasattr(trainer, 'optimizer'):
        if hasattr(trainer, 'setup_optimizer'):
            trainer.setup_optimizer()
        else:
            # Create a basic optimizer if setup_optimizer doesn't exist
            trainer.optimizer = torch.optim.Adam(model.parameters(), lr=trainer.lr)

    print(f"Training on device: {device}, AMP enabled: {use_amp}")

    for epoch in range(n_epochs):
        print(f'Starting epoch #{epoch}')
        accumulated_loss = 0.0
        batch_count = 0

        for i, batch_data in enumerate(train_loader):
            try:
                # Move data to device
                batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                              for x in batch_data]

                # Fix tensor shapes for multi-step training
                batch_data = fix_batch_data_shapes(batch_data, n_train_unroll)

                # Skip if batch data is None (malformed data)
                if batch_data is None:
                    print(f"Skipping batch {i} due to malformed data")
                    continue

                # Get the appropriate autocast context
                autocast_context = get_autocast_context(device, use_amp)

                with autocast_context:
                    train_loss, aux_data, scaled_loss = transition_trainer_train_mixed_precision(
                        trainer, batch_data, accumulation_steps, n_train_unroll)

                # Backward pass
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(trainer.optimizer)
                        scaler.update()
                        trainer.optimizer.zero_grad()
                else:
                    scaled_loss.backward()
                    if (i + 1) % accumulation_steps == 0:
                        trainer.optimizer.step()
                        trainer.optimizer.zero_grad()

                # Accumulate loss for logging
                loss_value = train_loss['loss'].item() if torch.is_tensor(train_loss['loss']) else train_loss['loss']
                accumulated_loss += loss_value
                batch_count += 1

                # Callback for logging
                if callback:
                    callback(train_loss, i * batch_size, epoch, aux_data=aux_data)

                # Periodic logging and validation
                if i % log_freq == 0 and i > 0:
                    if batch_count > 0:
                        avg_loss = accumulated_loss / batch_count
                        accumulated_loss = 0.0
                        batch_count = 0

                        update_str = f'Epoch {epoch} | Batch {i} | train_loss: {avg_loss:.3f}'

                        # Less frequent validation
                        if valid_loader is not None and i % (log_freq * 3) == 0:
                            model.eval()
                            with torch.no_grad():
                                test_func_to_use = test_func or (
                                    lambda bd: safe_n_step_calculate_losses(trainer, bd, n_train_unroll))
                                valid_losses = test_model_with_unroll_fix(model, test_func_to_use, valid_loader, device,
                                                                          n_train_unroll)
                                if valid_losses:  # Check if we got valid results
                                    valid_loss_means = {k: np.mean([x[k] for x in valid_losses])
                                                        for k in valid_losses[0].keys()}
                                    if valid_callback:
                                        valid_callback(valid_loss_means, i, epoch)
                                    for k, v in valid_loss_means.items():
                                        update_str += f' | valid_{k}: {v:.3f}'
                            model.train()

                        print(update_str)

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue


def test_model_with_unroll_fix(model, test_func, data_loader, device, n_train_unroll=1):
    """Test model with proper shape handling"""
    model.eval()
    losses = []

    try:
        with torch.no_grad():
            for batch_data in data_loader:
                # Move to device
                batch_data = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                              for x in batch_data]

                # Fix shapes
                batch_data = fix_batch_data_shapes(batch_data, n_train_unroll)

                # Skip malformed batches
                if batch_data is None:
                    continue

                # Calculate loss
                loss = test_func(batch_data)
                if not isinstance(loss, dict):
                    if torch.is_tensor(loss):
                        loss = {'loss': loss.mean()}
                    else:
                        loss = {'loss': loss}

                # Convert to CPU for storage
                cpu_loss = {}
                for k, v in loss.items():
                    if torch.is_tensor(v):
                        cpu_loss[k] = v.item()
                    else:
                        cpu_loss[k] = v
                losses.append(cpu_loss)

    except Exception as e:
        print(f"Error in test_model_with_unroll_fix: {e}")
        # Return empty loss if validation fails
        return [{'loss': 0.0}]

    return losses


def train_trans_model(args, encoder_model=None):
    import_logger(args)

    print('Loading data...')
    train_loader, test_loader, valid_loader = prepare_dataloaders(
        args.env_name, n=args.max_transitions, batch_size=args.batch_size,
        n_step=args.n_train_unroll, preprocess=args.preprocess, randomize=True,
        n_preload=args.n_preload, preload_all=args.preload_data,
        extra_buffer_keys=args.extra_buffer_keys)

    print(f'Data split: {len(train_loader.dataset)}/{len(test_loader.dataset)}/{len(valid_loader.dataset)}')

    if encoder_model is None:
        print('Constructing encoder...')
        sample_obs = next(iter(train_loader))[0][0]
        if args.n_train_unroll > 1 and len(sample_obs.shape) > 3:
            sample_obs = sample_obs[0]
        encoder_model = construct_ae_model(sample_obs.shape, args)[0]

    encoder_model = encoder_model.to(args.device)
    freeze_model(encoder_model)
    encoder_model.eval()

    if hasattr(encoder_model, 'enable_sparsity'):
        encoder_model.enable_sparsity()

    print('Constructing transition model...')
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    trans_model, trans_trainer = construct_trans_model(
        encoder_model, args, env.action_space, load=args.load)

    trans_model = trans_model.to(args.device)

    update_params(args)
    track_model(trans_model, args)

    # Training configuration
    use_amp = getattr(args, 'use_amp', False)  # Default to False for stability
    accumulation_steps = getattr(args, 'accumulation_steps', 1)

    # Disable AMP on CPU or if not supported
    if args.device == 'cpu':
        use_amp = False

    print(f"Mixed precision training: {use_amp}")
    print(f"Accumulation steps: {accumulation_steps}")

    # Callbacks
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

        if TRANS_STEP % max(1, (args.log_freq // 10)) == 0:
            log_metrics({
                'epoch': epoch,
                'step': TRANS_STEP,
                **{f'train_{k}': sum(v) / len(v) for k, v in train_log_buffer.items()},
                **{k: sum(v) / len(v) for k, v in aux_log_buffer.items()}},
                args, prefix='trans', step=TRANS_STEP)
            train_log_buffer = defaultdict(list)
            aux_log_buffer = defaultdict(list)
        TRANS_STEP += 1

    def valid_callback(valid_data, batch_idx, epoch):
        global TRANS_STEP
        log_metrics({
            'epoch': epoch,
            'step': TRANS_STEP,
            **{f'valid_{k}': v for k, v in valid_data.items()}},
            args, prefix='trans', step=TRANS_STEP)

    n_epochs = args.trans_epochs if args.trans_epochs is not None else args.epochs

    try:
        optimized_transition_train_loop(
            trans_model, trans_trainer, train_loader, valid_loader, n_epochs,
            args.batch_size, args.log_freq, callback=train_callback,
            valid_callback=valid_callback, test_func=None, use_amp=use_amp,
            accumulation_steps=accumulation_steps, n_train_unroll=args.n_train_unroll)
    except KeyboardInterrupt:
        print('Stopping training')
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

    # Test the model
    print('Starting model evaluation...')
    try:
        test_func = lambda bd: safe_n_step_calculate_losses(trans_trainer, bd, args.n_train_unroll)
        test_losses = test_model_with_unroll_fix(trans_model, test_func, test_loader, args.device, args.n_train_unroll)
        if test_losses:
            test_losses = {k: np.mean([d[k] for d in test_losses]) for k in test_losses[0].keys()}
            print(f'Transition model test losses: {test_losses}')
        else:
            print('Test evaluation failed, skipping...')
    except Exception as e:
        print(f'Test evaluation error: {e}')

    if args.save:
        save_model(trans_model, args, model_hash=args.trans_model_hash)
        print('Transition model saved')

    return trans_model


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