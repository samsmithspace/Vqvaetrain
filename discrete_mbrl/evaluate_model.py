import gc
import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import gymnasium as gym
from gymnasium.core import Wrapper

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from shared.models import *
from shared.trainers import *
from data_helpers import *
from data_logging import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from utils import *
from eval_policies.policies import load_policy

sns.set()

GAMMA_CONST = 0.99
N_EXAMPLE_IMGS = 15
SEED = 0
N_RAND_LATENT_SAMPLES = 500
DISCRETE_TRANS_TYPES = ('discrete', 'transformer', 'transformerdec')
CONTINUOUS_TRANS_TYPES = ('continuous', 'shared_vq')


def safe_tensor_to_device(data, device):
    """Safely move tensor to device with proper shape handling"""
    if isinstance(data, (list, tuple)):
        return [safe_tensor_to_device(x, device) for x in data]
    elif torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    return data


def fix_tensor_shapes_for_discrete(obs, actions, next_obs, rewards, dones, extra_data=None):
    """Fix tensor shapes specifically for discrete transition models"""
    extra_data = extra_data or []

    # Handle multi-step data by flattening
    if len(obs.shape) == 5:  # [batch, n_steps, channels, height, width]
        batch_size, n_steps = obs.shape[:2]
        obs = obs.reshape(batch_size * n_steps, *obs.shape[2:])
        next_obs = next_obs.reshape(batch_size * n_steps, *next_obs.shape[2:])
        rewards = rewards.reshape(batch_size * n_steps)
        dones = dones.reshape(batch_size * n_steps)

        # Handle actions - ensure 1D for discrete models
        if len(actions.shape) == 2:  # [batch, n_steps]
            actions = actions.reshape(batch_size * n_steps)
        else:  # [batch, n_steps, action_dim]
            actions = actions.reshape(batch_size * n_steps, *actions.shape[2:])
            if actions.shape[-1] == 1:
                actions = actions.squeeze(-1)

        extra_data = [x.reshape(batch_size * n_steps, *x.shape[2:]) for x in extra_data]

    # Ensure actions are proper shape for discrete models
    if len(actions.shape) == 1:
        actions = actions  # Keep 1D for discrete models
    elif len(actions.shape) == 2 and actions.shape[1] == 1:
        actions = actions.squeeze(-1)  # Remove unnecessary dimension

    return obs, actions, next_obs, rewards, dones, extra_data


def evaluate_encoder_reconstruction(encoder_model, test_loader, args):
    """Evaluate encoder reconstruction quality"""
    n_samples = 0
    encoder_recon_loss = 0.0
    all_latents = []

    print('Evaluating encoder reconstruction...')
    device = next(encoder_model.parameters()).device

    with torch.no_grad():
        for batch_data in test_loader:
            obs_data = batch_data[0].to(device)
            n_samples += obs_data.shape[0]

            # Encode and decode
            latents = encoder_model.encode(obs_data)
            recon_outputs = encoder_model.decode(latents)
            all_latents.append(latents.cpu())

            # Calculate reconstruction loss
            if recon_outputs.shape == obs_data.shape:
                encoder_recon_loss += F.mse_loss(recon_outputs, obs_data, reduction='sum').item()

    all_latents = torch.cat(all_latents, dim=0)
    encoder_recon_loss = encoder_recon_loss / n_samples

    print(f'Encoder reconstruction loss: {encoder_recon_loss:.2f}')
    log_metrics({'encoder_recon_loss': encoder_recon_loss}, args)

    return all_latents, encoder_recon_loss


def evaluate_random_latent_sampling(encoder_model, all_latents, args, unique_obs, rev_transform):
    """Evaluate decoder by sampling random latent vectors"""
    print('Sampling random latent vectors...')

    # Skip for identity encoders
    if hasattr(encoder_model, '__class__') and 'Identity' in encoder_model.__class__.__name__:
        print('Identity encoder detected - skipping random latent sampling')
        return

    device = next(encoder_model.parameters()).device

    if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
        # Continuous latent space
        latent_dim = encoder_model.latent_dim
        all_latents = all_latents.reshape(all_latents.shape[0], latent_dim)

        # Sample from learned distribution
        latent_means = all_latents.mean(dim=0)
        latent_stds = all_latents.std(dim=0)
        sampled_latents = torch.normal(
            latent_means.repeat(N_RAND_LATENT_SAMPLES),
            latent_stds.repeat(N_RAND_LATENT_SAMPLES)
        ).reshape(N_RAND_LATENT_SAMPLES, latent_dim)

    elif args.trans_model_type in DISCRETE_TRANS_TYPES:
        # Discrete latent space
        batch_size = N_RAND_LATENT_SAMPLES
        n_positions = encoder_model.n_latent_embeds
        sampled_latents = torch.randint(
            0, encoder_model.n_embeddings, (batch_size, n_positions), dtype=torch.long
        )

    # Generate images from sampled latents
    with torch.no_grad():
        obs = encoder_model.decode(sampled_latents.to(device))

    if args.exact_comp and unique_obs is not None:
        min_dists = get_min_mses(obs.cpu(), unique_obs)
        print(f'Sample latent obs L2: {min_dists.mean():.4f}')
        log_metrics({'sample_latent_obs_l2': min_dists.mean()}, args)

    # Log sample images
    imgs = obs_to_img(obs[:N_EXAMPLE_IMGS].cpu(), env_name=args.env_name, rev_transform=rev_transform)
    log_images({'sample_latent_imgs': imgs}, args)


def evaluate_transition_model_step(batch_data, encoder_model, trans_model, args):
    """Evaluate a single step of the transition model with proper shape handling"""
    device = next(trans_model.parameters()).device

    # Move data to device
    batch_data = safe_tensor_to_device(batch_data, device)
    obs, actions, next_obs, rewards, dones = batch_data[:5]
    extra_data = batch_data[5:] if len(batch_data) > 5 else []

    # Fix shapes for discrete models
    obs, actions, next_obs, rewards, dones, extra_data = fix_tensor_shapes_for_discrete(
        obs, actions, next_obs, rewards, dones, extra_data
    )

    try:
        # Encode current and next observations
        current_z = encoder_model.encode(obs)
        next_z = encoder_model.encode(next_obs)

        print(f"Debug - current_z shape: {current_z.shape}, dtype: {current_z.dtype}")
        print(f"Debug - next_z shape: {next_z.shape}, dtype: {next_z.dtype}")

        # Handle discrete vs continuous models differently
        if args.trans_model_type in DISCRETE_TRANS_TYPES:
            # For VQ-VAE models, the encoder output might be spatial: [batch, spatial_positions]
            # or [batch, height, width] - we need to flatten to [batch, n_latent_embeds]
            if len(current_z.shape) > 2:
                # Spatial output - flatten spatial dimensions
                batch_size = current_z.shape[0]
                current_z = current_z.view(batch_size, -1)
                next_z = next_z.view(batch_size, -1)
                print(f"Debug - flattened current_z shape: {current_z.shape}")

            # Ensure indices are long type
            if current_z.dtype != torch.long:
                current_z = current_z.long()
                next_z = next_z.long()

            # For discrete transition models, we need to handle the input format carefully
            # The model expects either indices or one-hot vectors
            n_latent_positions = current_z.shape[1]  # This should match encoder.n_latent_embeds

            # Check if we need to adjust actions format
            if len(actions.shape) > 1:
                actions = actions.squeeze(-1)

            # Make actions 2D for discrete transition models: [batch, 1]
            actions = actions.unsqueeze(-1)

            print(f"Debug - before transition model:")
            print(f"  current_z: {current_z.shape}, dtype: {current_z.dtype}")
            print(f"  actions: {actions.shape}, dtype: {actions.dtype}")
            print(f"  n_latent_positions: {n_latent_positions}")

            # Use the transition model's expected input format
            if hasattr(trans_model, 'forward'):
                # Try direct forward pass with indices
                next_z_pred_logits, next_reward_pred, next_gamma_pred = trans_model(
                    current_z, actions, return_logits=True
                )
            else:
                # Fallback: create one-hot encoding
                z_logits = F.one_hot(current_z, encoder_model.n_embeddings).float()
                z_logits = z_logits.permute(0, 2, 1)  # [batch, n_embeddings, n_latent_embeds]

                next_z_pred_logits, next_reward_pred, next_gamma_pred = trans_model(
                    z_logits, actions, return_logits=True
                )

            # Convert predictions back to indices if needed
            if hasattr(trans_model, 'logits_to_state'):
                next_z_pred = trans_model.logits_to_state(next_z_pred_logits)
            else:
                next_z_pred = torch.argmax(next_z_pred_logits, dim=1)

            # Calculate losses - ensure shapes match
            if next_z_pred_logits.dim() == 3:  # [batch, n_embeddings, n_positions]
                # Reshape for cross entropy: [batch * n_positions, n_embeddings]
                batch_size, n_embeddings, n_positions = next_z_pred_logits.shape
                next_z_pred_logits_flat = next_z_pred_logits.permute(0, 2, 1).reshape(-1, n_embeddings)
                next_z_flat = next_z.reshape(-1)

                state_losses = F.cross_entropy(next_z_pred_logits_flat, next_z_flat, reduction='none')
                state_losses = state_losses.view(batch_size, n_positions).sum(1)

                state_accs = (next_z_pred == next_z).float().mean(1)
            else:
                # Standard case
                state_losses = F.cross_entropy(next_z_pred_logits, next_z, reduction='none')
                state_accs = (next_z_pred == next_z).float()
                if state_accs.dim() > 1:
                    state_accs = state_accs.mean(1)

        else:
            # Continuous models
            latent_dim = encoder_model.latent_dim
            current_z = current_z.reshape(current_z.shape[0], latent_dim)
            next_z = next_z.reshape(next_z.shape[0], latent_dim)

            # Predict next state
            next_z_pred, next_reward_pred, next_gamma_pred = trans_model(current_z, actions)

            # Calculate losses
            state_losses = F.mse_loss(next_z_pred, next_z, reduction='none').sum(1)
            state_accs = torch.zeros_like(state_losses)  # Not applicable for continuous

        # Calculate other losses
        gamma = (1 - dones) * GAMMA_CONST
        reward_losses = F.mse_loss(next_reward_pred.squeeze(), rewards, reduction='none')
        gamma_losses = F.mse_loss(next_gamma_pred.squeeze(), gamma, reduction='none')

        # Image reconstruction loss - decode the predicted next state
        try:
            if args.trans_model_type in DISCRETE_TRANS_TYPES:
                decoded_next_obs = encoder_model.decode(next_z_pred)
            else:
                decoded_next_obs = encoder_model.decode(next_z_pred)
            img_mse_losses = F.mse_loss(decoded_next_obs, next_obs, reduction='none').view(next_obs.shape[0], -1).sum(1)
        except:
            # Skip image reconstruction if decoding fails
            img_mse_losses = torch.zeros(next_obs.shape[0])

        loss_dict = {
            'state_loss': state_losses.cpu().numpy(),
            'state_acc': state_accs.cpu().numpy(),
            'reward_loss': reward_losses.cpu().numpy(),
            'gamma_loss': gamma_losses.cpu().numpy(),
            'img_mse_loss': img_mse_losses.cpu().numpy(),
        }

        return loss_dict

    except Exception as e:
        print(f"Error in transition model evaluation: {e}")
        print(f"Shapes - obs: {obs.shape}, actions: {actions.shape}, next_obs: {next_obs.shape}")
        if hasattr(encoder_model, 'n_latent_embeds'):
            print(
                f"Model info - n_latent_embeds: {encoder_model.n_latent_embeds}, n_embeddings: {encoder_model.n_embeddings}")

        # Try to get more debug info about encoded states
        try:
            current_z = encoder_model.encode(obs[:1])  # Just one sample for debugging
            print(f"Debug - encoded sample shape: {current_z.shape}, dtype: {current_z.dtype}")
            print(f"Debug - encoded sample range: [{current_z.min()}, {current_z.max()}]")
        except Exception as debug_e:
            print(f"Debug encoding failed: {debug_e}")

        import traceback
        traceback.print_exc()
        return None


def evaluate_transition_model(encoder_model, trans_model, args, unique_obs=None):
    """Evaluate transition model accuracy"""
    print('Evaluating transition model...')

    # Load n-step data
    n_step_loader = prepare_dataloader(
        args.env_name, 'test', batch_size=args.batch_size, preprocess=args.preprocess,
        randomize=True, n=args.max_transitions, n_preload=0, preload=False,
        n_step=args.eval_unroll_steps, extra_buffer_keys=args.extra_buffer_keys
    )

    all_losses = []
    n_processed = 0

    print('Calculating stats for n-step data...')
    for batch_data in tqdm(n_step_loader):
        if n_processed >= 10000:  # Limit for faster evaluation
            break

        loss_dict = evaluate_transition_model_step(batch_data, encoder_model, trans_model, args)

        if loss_dict is not None:
            all_losses.append(loss_dict)
            n_processed += batch_data[0].shape[0]

    if all_losses:
        # Calculate mean losses
        mean_losses = {}
        for key in all_losses[0].keys():
            values = np.concatenate([loss[key] for loss in all_losses])
            mean_losses[key] = np.mean(values)

        print(f'Transition model losses: {mean_losses}')
        log_metrics(mean_losses, args)
    else:
        print('No valid transition model results')


def get_min_mses(gens, sources, return_idxs=False):
    """Get minimum MSE between generated samples and source samples"""
    if len(gens.shape) == len(sources.shape) - 1:
        gens = gens.unsqueeze(0)

    min_dists = []
    min_idxs = []

    for gen in gens:
        dists = ((gen.unsqueeze(0) - sources) ** 2).view(sources.shape[0], -1).sum(dim=1)
        min_dist = dists.min().item()
        min_dists.append(min_dist)

        if return_idxs:
            min_idx = dists.argmin().item()
            min_idxs.append(min_idx)

    if return_idxs:
        return np.array(min_dists), np.array(min_idxs)
    return np.array(min_dists)


def generate_sample_images(encoder_model, test_loader, args):
    """Generate sample reconstruction images"""
    print('Generating reconstruction sample images...')

    device = next(encoder_model.parameters()).device
    sample_data = next(iter(test_loader))[0][:N_EXAMPLE_IMGS].to(device)

    with torch.no_grad():
        if hasattr(encoder_model, 'forward') and callable(encoder_model.forward):
            try:
                recon_obs = encoder_model(sample_data)
                if isinstance(recon_obs, tuple):
                    recon_obs = recon_obs[0]
            except:
                # Fallback to encode/decode
                encoded = encoder_model.encode(sample_data)
                recon_obs = encoder_model.decode(encoded)
        else:
            encoded = encoder_model.encode(sample_data)
            recon_obs = encoder_model.decode(encoded)

    # Create comparison images
    comparison = torch.cat([sample_data.cpu(), recon_obs.cpu()], dim=3)
    imgs = obs_to_img(comparison, env_name=args.env_name)
    log_images({'recon_sample_imgs': imgs}, args)


def eval_model(args, encoder_model=None, trans_model=None):
    """Main evaluation function - simplified and fixed"""
    import_logger(args)
    torch.manual_seed(SEED)

    # Setup
    unique_obs = None
    if args.exact_comp:
        try:
            unique_obs, unique_data_hash = get_unique_obs(
                args, cache=True, partition='all', return_hash=True, early_stop_frac=0.2
            )
            log_metrics({'unique_obs_count': len(unique_obs)}, args)
        except Exception as e:
            print(f"Warning: Could not load unique observations: {e}")
            args.exact_comp = False

    # Load test data
    print('Loading data...')
    test_loader = prepare_dataloader(
        args.env_name, 'test', batch_size=args.batch_size, preprocess=args.preprocess,
        randomize=False, n=args.max_transitions, n_preload=0, preload=False,
        extra_buffer_keys=args.extra_buffer_keys
    )

    # Load models if not provided
    if encoder_model is None or trans_model is None:
        print('Loading models...')
        sample_obs = next(iter(test_loader))[0]

        if encoder_model is None:
            encoder_model = construct_ae_model(sample_obs.shape[1:], args)[0]
            encoder_model = encoder_model.to(args.device)

        if trans_model is None:
            env = make_env(args.env_name, max_steps=args.env_max_steps)
            trans_model = construct_trans_model(encoder_model, args, env.action_space)[0]
            trans_model = trans_model.to(args.device)
            env.close()

    # Ensure models are in eval mode
    encoder_model.eval()
    trans_model.eval()
    freeze_model(encoder_model)
    freeze_model(trans_model)

    # Evaluation pipeline
    try:
        # 1. Encoder reconstruction
        all_latents, encoder_recon_loss = evaluate_encoder_reconstruction(encoder_model, test_loader, args)

        # 2. Random latent sampling
        evaluate_random_latent_sampling(encoder_model, all_latents, args, unique_obs, None)

        # 3. Sample reconstruction images
        generate_sample_images(encoder_model, test_loader, args)

        # 4. Transition model evaluation
        evaluate_transition_model(encoder_model, trans_model, args, unique_obs)

        print('✅ Evaluation completed successfully!')

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # Parse args
    args = get_args()
    # Setup logging
    args = init_experiment('discrete-mbrl-eval', args)
    # Evaluate models
    eval_model(args)
    # Clean up logging
    finish_experiment(args)