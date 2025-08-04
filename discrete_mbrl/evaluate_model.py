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
SEED = 0  # Should be same as seed used for prior steps
PRELOAD_TEST = False
TEST_WORKERS = 0
EARLY_STOP_COUNT = 1000  # Reduced from 3000 for memory efficiency
DISCRETE_TRANS_TYPES = ('discrete', 'transformer', 'transformerdec')
CONTINUOUS_TRANS_TYPES = ('continuous', 'shared_vq')
N_RAND_LATENT_SAMPLES = 500
STATE_DISTRIB_SAMPLES = 10000  # Reduced from 20000
IMAGINE_DISTRIB_SAMPLES = 1000  # Reduced from 2000
UNIQUE_OBS_EARLY_STOP = 1.0  # 0.2s


def clear_gpu_memory():
    """Clear GPU memory and collect garbage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        return f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
    return "GPU not available"


def calculate_trans_losses_efficient(
        next_z, next_reward, next_gamma, next_z_pred_logits, next_z_pred, next_reward_pred,
        next_gamma_pred, next_obs, trans_model_type, encoder_model, rand_obs=None,
        init_obs=None, all_obs=None, all_trans=None, curr_z=None, acts=None):
    """Memory-efficient version of calculate_trans_losses with chunked processing"""

    # Calculate the transition reconstruction loss
    loss_dict = {}
    batch_size = next_z.shape[0]
    chunk_size = min(16, batch_size)  # Process in small chunks

    # Initialize lists to collect results
    state_losses_list = []
    state_accs_list = []
    img_mse_losses_list = []
    reward_losses_list = []
    gamma_losses_list = []
    rand_img_mse_losses_list = []
    init_img_mse_losses_list = []
    closest_img_mse_losses_list = []
    real_transition_frac_list = []

    # Process in chunks to manage memory
    for chunk_start in range(0, batch_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, batch_size)

        # Get chunks
        next_z_chunk = next_z[chunk_start:chunk_end]
        next_z_pred_logits_chunk = next_z_pred_logits[chunk_start:chunk_end]
        next_z_pred_chunk = next_z_pred[chunk_start:chunk_end]
        next_reward_chunk = next_reward[chunk_start:chunk_end]
        next_gamma_chunk = next_gamma[chunk_start:chunk_end]
        next_reward_pred_chunk = next_reward_pred[chunk_start:chunk_end]
        next_gamma_pred_chunk = next_gamma_pred[chunk_start:chunk_end]
        next_obs_chunk = next_obs[chunk_start:chunk_end]

        # State losses
        if trans_model_type in CONTINUOUS_TRANS_TYPES:
            assert next_z_chunk.shape == next_z_pred_logits_chunk.shape
            state_losses = torch.pow(next_z_chunk - next_z_pred_logits_chunk, 2)
            state_losses = state_losses.view(next_z_chunk.shape[0], -1).sum(1)
            state_losses_list.extend(state_losses.cpu().numpy())
            state_accs_list.extend([0] * next_z_chunk.shape[0])
        elif trans_model_type in DISCRETE_TRANS_TYPES:
            state_losses = F.cross_entropy(
                next_z_pred_logits_chunk, next_z_chunk, reduction='none')
            state_losses = state_losses.view(next_z_chunk.shape[0], -1).sum(1)
            state_accs = (next_z_pred_chunk == next_z_chunk).float().view(next_z_chunk.shape[0], -1).mean(1)
            state_losses_list.extend(state_losses.cpu().numpy())
            state_accs_list.extend(state_accs.cpu().numpy())

        # Image reconstruction losses
        with torch.no_grad():  # Don't need gradients for decoder during evaluation
            next_obs_pred = encoder_model.decode(next_z_pred_chunk).cpu()
        img_mse_losses = torch.pow(next_obs_chunk - next_obs_pred, 2)
        img_mse_losses_list.extend(img_mse_losses.view(next_obs_chunk.shape[0], -1).sum(1).numpy())

        # Reward and gamma losses
        reward_losses = F.mse_loss(next_reward_chunk, next_reward_pred_chunk.squeeze().cpu(), reduction='none')
        gamma_losses = F.mse_loss(next_gamma_chunk, next_gamma_pred_chunk.squeeze().cpu(), reduction='none')
        reward_losses_list.extend(reward_losses.numpy())
        gamma_losses_list.extend(gamma_losses.numpy())

        # Optional losses (with memory-efficient handling)
        if rand_obs is not None:
            rand_obs_chunk = rand_obs[chunk_start:chunk_end]
            rand_img_mse_losses = torch.pow(next_obs_chunk - rand_obs_chunk, 2)
            rand_img_mse_losses_list.extend(rand_img_mse_losses.view(next_obs_chunk.shape[0], -1).sum(1).numpy())
        else:
            rand_img_mse_losses_list.extend([np.nan] * next_obs_chunk.shape[0])

        if init_obs is not None:
            init_obs_chunk = init_obs[chunk_start:chunk_end]
            init_img_mse_losses = torch.pow(next_obs_chunk - init_obs_chunk, 2)
            init_img_mse_losses_list.extend(init_img_mse_losses.view(next_obs_chunk.shape[0], -1).sum(1).numpy())
        else:
            init_img_mse_losses_list.extend([np.nan] * next_obs_chunk.shape[0])

        # Skip expensive closest image computation to save memory
        closest_img_mse_losses_list.extend([np.nan] * next_obs_chunk.shape[0])
        real_transition_frac_list.extend([np.nan] * next_obs_chunk.shape[0])

        # Clear chunk tensors
        del (next_z_chunk, next_z_pred_logits_chunk, next_z_pred_chunk,
             next_reward_chunk, next_gamma_chunk, next_reward_pred_chunk,
             next_gamma_pred_chunk, next_obs_chunk, next_obs_pred)

        # Memory cleanup every few chunks
        if chunk_start % (chunk_size * 4) == 0:
            clear_gpu_memory()

    # Convert lists to numpy arrays
    loss_dict['state_loss'] = np.array(state_losses_list)
    loss_dict['state_acc'] = np.array(state_accs_list)
    loss_dict['img_mse_loss'] = np.array(img_mse_losses_list)
    loss_dict['reward_loss'] = np.array(reward_losses_list)
    loss_dict['gamma_loss'] = np.array(gamma_losses_list)
    loss_dict['rand_img_mse_loss'] = np.array(rand_img_mse_losses_list)
    loss_dict['init_img_mse_loss'] = np.array(init_img_mse_losses_list)
    loss_dict['closest_img_mse_loss'] = np.array(closest_img_mse_losses_list)
    loss_dict['real_transition_frac'] = np.array(real_transition_frac_list)

    return loss_dict


def prepare_dataloaders_memory_efficient(*args, **kwargs):
    """Memory-efficient wrapper for prepare_dataloaders"""
    # Force memory-efficient settings
    kwargs['pin_memory'] = False
    kwargs['persistent_workers'] = False
    kwargs['prefetch_factor'] = 1

    # Reduce batch sizes
    if 'batch_size' in kwargs:
        kwargs['batch_size'] = min(kwargs['batch_size'], 64)

    return prepare_dataloaders(*args, **kwargs)


def wandb_log(items, do_log, make_gif=False):
    if do_log:
        for k, v in items.items():
            if isinstance(v, list) and isinstance(v[0], np.ndarray) \
                    and len(v[0].shape) > 1:
                items[k] = [wandb.Image(x) for x in v]
            elif isinstance(v, np.ndarray) and len(v.shape) > 1:
                if make_gif:
                    items[k] = wandb.Video(v, fps=4, format='gif')
                else:
                    items[k] = wandb.Image(v)
            elif isinstance(v, Figure):
                items[k] = wandb.Image(v)
        wandb.log(items)


def save_and_log_imgs(imgs, label, results_dir, args):
    # if args.save:
    #   plt.savefig(os.path.join(results_dir,
    #     f'{args.ae_model_type}_v{args.ae_model_version}_{label}.png'))
    wandb_log({label: [img.clip(0, 1) for img in imgs]}, args.wandb)


def get_min_mses(gens, sources, return_idxs=False):
    """ Get the minimum distance between each generated sample and all sources. """
    if len(gens.shape) == len(sources.shape) - 1:
        gens = gens[None]
    assert gens.shape[1:] == sources.shape[1:], \
        f'gens.shape: {gens.shape}, sources.shape: {sources.shape}, but core dims need to be equal!'

    min_dists = []
    min_idxs = []
    for gen in gens:
        dists = (gen.unsqueeze(0) - sources) ** 2
        dists = dists.reshape(dists.shape[0], -1).sum(dim=1)
        min_dist = dists.min().item()
        min_dists.append(min_dist)
        if return_idxs:
            min_idx = dists.argmin().item()
            min_idxs.append(min_idx)
    if return_idxs:
        return np.array(min_dists), np.array(min_idxs)
    return np.array(min_dists)


def update_losses(losses, new_losses, args, step, log=True):
    for k, v in new_losses.items():
        losses[k].extend(v)
    n_losses = len(new_losses[list(new_losses.keys())[0]])
    losses['model'].extend([f'{args.trans_model_type}' \
                            + f'_v{args.trans_model_version}' for _ in range(n_losses)])
    losses['step'].extend([step for _ in range(n_losses)])


def eval_model(args, encoder_model=None, trans_model=None):
    import_logger(args)

    # Memory management setup
    clear_gpu_memory()
    print(f"🧠 Initial {get_gpu_memory_info()}")

    # Override batch sizes for memory efficiency
    original_batch_size = args.batch_size
    original_eval_batch_size = args.eval_batch_size

    # Reduce batch sizes significantly for evaluation
    args.batch_size = min(args.batch_size // 8, 64)  # Much smaller
    args.eval_batch_size = min(args.eval_batch_size // 8, 32)  # Much smaller
    print(f"🔧 Reduced batch sizes for evaluation - batch: {args.batch_size}, eval: {args.eval_batch_size}")

    # Force memory-efficient settings
    args.pin_memory = False
    args.persistent_workers = False
    args.preload_data = False

    # Reduce unroll steps if too high
    if args.eval_unroll_steps > 10:
        args.eval_unroll_steps = 5
        print(f"🔧 Reduced eval_unroll_steps to {args.eval_unroll_steps}")

    # Randomize pytorch seed
    torch.manual_seed(time.time())

    # Skip exact comparison if it's memory intensive
    if hasattr(args, 'exact_comp') and args.exact_comp:
        print("⚠️ Skipping exact_comp due to memory constraints")
        args.exact_comp = False

    env = make_env(args.env_name, max_steps=args.env_max_steps)

    ### Loading Models & Some Data ###

    print('Loading data...')
    test_loader = prepare_dataloader(
        args.env_name, 'test', batch_size=args.batch_size, preprocess=args.preprocess,
        randomize=False, n=min(args.max_transitions or 10000, 5000),
        n_preload=0, preload=False,  # Force no preloading
        extra_buffer_keys=args.extra_buffer_keys)
    test_sampler = create_fast_loader(
        test_loader.dataset, batch_size=1, shuffle=True, num_workers=0, n_step=1,
        pin_memory=False, persistent_workers=False)
    rev_transform = test_loader.dataset.flat_rev_obs_transform

    # Load the encoder
    if encoder_model is None:
        sample_obs = next(iter(test_sampler))[0]
        encoder_model = construct_ae_model(
            sample_obs.shape[1:], args)[0]

    try:
        encoder_model = encoder_model.to(args.device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ GPU OOM loading encoder, using CPU")
            args.device = 'cpu'
            encoder_model = encoder_model.to('cpu')
        else:
            raise e

    freeze_model(encoder_model)
    encoder_model.eval()
    print(f'Loaded encoder on {args.device}')

    if hasattr(encoder_model, 'enable_sparsity'):
        encoder_model.enable_sparsity()

    # Load the transition model
    if trans_model is None:
        trans_model = construct_trans_model(encoder_model, args, env.action_space)[0]

    try:
        trans_model = trans_model.to(args.device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ GPU OOM loading transition model, using CPU")
            args.device = 'cpu'
            trans_model = trans_model.to('cpu')
            encoder_model = encoder_model.to('cpu')
        else:
            raise e

    freeze_model(trans_model)
    trans_model.eval()
    print(f'Loaded transition model on {args.device}')

    # Hack for universal_vq to work with the current code
    if args.trans_model_type == 'universal_vq':
        if encoder_model.quantized_enc:
            global CONTINUOUS_TRANS_TYPES
            CONTINUOUS_TRANS_TYPES = CONTINUOUS_TRANS_TYPES + ('universal_vq',)
        else:
            global DISCRETE_TRANS_TYPES
            DISCRETE_TRANS_TYPES = DISCRETE_TRANS_TYPES + ('universal_vq',)

    clear_gpu_memory()
    print(f'Memory usage after model loading: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.2f} GB')
    print(f"🧠 {get_gpu_memory_info()}")

    torch.manual_seed(SEED)
    gc.collect()

    ### Encoder Testing ###

    # Calculate autoencoder reconstruction loss
    print("Testing encoder...")
    n_samples = 0
    encoder_recon_loss = torch.tensor(0, dtype=torch.float64)
    all_latents = []

    batch_count = 0
    for batch_data in test_loader:
        obs_data = batch_data[0]
        # Limit batch size further if needed
        if obs_data.shape[0] > 16:
            obs_data = obs_data[:16]

        n_samples += obs_data.shape[0]
        with torch.no_grad():
            latents = encoder_model.encode(obs_data.to(args.device))
            recon_outputs = encoder_model.decode(latents)
            all_latents.append(latents.cpu())
        encoder_recon_loss += torch.sum((recon_outputs.cpu() - obs_data) ** 2)

        batch_count += 1
        if batch_count % 10 == 0:
            clear_gpu_memory()

        # Early stopping for memory
        if batch_count >= 50:  # Limit number of batches
            break

    all_latents = torch.cat(all_latents, dim=0)
    encoder_recon_loss = (encoder_recon_loss / n_samples).item()
    print(f'Encoder reconstruction loss: {encoder_recon_loss:.2f}')
    log_metrics({'encoder_recon_loss': encoder_recon_loss}, args)

    clear_gpu_memory()
    print(f'Memory usage after encoder test: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3:.2f} GB')

    # Sample random latent vectors eval (reduced)
    print('Sampling random latent vectors...')
    N_RAND_SAMPLES = min(N_RAND_LATENT_SAMPLES, 100)  # Much smaller

    if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
        latent_dim = encoder_model.latent_dim
        all_latents = all_latents.reshape(all_latents.shape[0], latent_dim)

        latent_min = all_latents.min()
        latent_max = all_latents.max()
        latent_range = latent_max - latent_min
        uniform_sampled_latents = torch.rand((N_RAND_SAMPLES, latent_dim))
        uniform_sampled_latents = uniform_sampled_latents * latent_range + latent_min

        with torch.no_grad():
            obs = encoder_model.decode(uniform_sampled_latents.to(args.device))
        obs = obs.cpu()

        imgs = obs_to_img(obs[:min(N_EXAMPLE_IMGS, 5)], env_name=args.env_name, rev_transform=rev_transform)
        log_images({'uniform_cont_sample_latent_imgs': imgs}, args)

        latent_means = all_latents.mean(dim=0)
        latent_stds = all_latents.std(dim=0)
        normal_sampled_latents = torch.normal(
            latent_means.repeat(N_RAND_SAMPLES),
            latent_stds.repeat(N_RAND_SAMPLES))
        normal_sampled_latents = normal_sampled_latents.reshape(N_RAND_SAMPLES, latent_dim)

        with torch.no_grad():
            obs = encoder_model.decode(normal_sampled_latents.to(args.device))
        obs = obs.cpu()
        imgs = obs_to_img(obs[:min(N_EXAMPLE_IMGS, 5)], env_name=args.env_name, rev_transform=rev_transform)
        log_images({'normal_sample_latent_imgs': imgs}, args)

    elif args.trans_model_type in DISCRETE_TRANS_TYPES:
        latent_dim = encoder_model.n_latent_embeds
        sampled_latents = torch.randint(
            0, encoder_model.n_embeddings, (N_RAND_SAMPLES, latent_dim,))
        with torch.no_grad():
            obs = encoder_model.decode(sampled_latents.to(args.device))
        obs = obs.cpu()
        imgs = obs_to_img(obs[:min(N_EXAMPLE_IMGS, 5)], env_name=args.env_name, rev_transform=rev_transform)
        log_images({'uniform_disc_sample_latent_imgs': imgs}, args)

    # Generate reconstruction sample images (reduced)
    print('Generating reconstruction sample images...')
    example_imgs = []
    for i, sample_transition in enumerate(test_sampler):
        sample_obs = sample_transition[0]
        if i >= min(N_EXAMPLE_IMGS, 5):  # Much fewer examples
            break
        with torch.no_grad():
            recon_obs = encoder_model(sample_obs.to(args.device))
        if isinstance(recon_obs, tuple):
            recon_obs = recon_obs[0]
        both_obs = torch.cat([sample_obs, recon_obs.cpu()], dim=0)
        both_imgs = obs_to_img(both_obs, env_name=args.env_name, rev_transform=rev_transform)
        cat_img = np.concatenate([both_imgs[0], both_imgs[1]], axis=1)
        example_imgs.append(cat_img)

    log_images({'recon_sample_imgs': example_imgs}, args)

    del test_loader, test_sampler
    clear_gpu_memory()

    ### Transition Model Testing ###

    # Prepare n-step data with reduced parameters
    n_step_loader = prepare_dataloader(
        args.env_name, 'test', batch_size=min(args.batch_size, 16), preprocess=args.preprocess,
        randomize=True, n=min(args.max_transitions or 5000, 2000), n_preload=0, preload=False,
        n_step=args.eval_unroll_steps, extra_buffer_keys=args.extra_buffer_keys)

    print(f'Sampled {args.eval_unroll_steps}-step sub-trajectories')

    # Calculate n-step statistics with memory management
    n_step_stats = dict(
        state_loss=[], state_acc=[], reward_loss=[],
        gamma_loss=[], img_mse_loss=[], rand_img_mse_loss=[],
        init_img_mse_loss=[], step=[], model=[],
        closest_img_mse_loss=[], real_transition_frac=[])

    n_full_unroll_samples = 0
    print('Calculating stats for n-step data...')

    batch_count = 0
    for i, n_step_trans in tqdm(enumerate(n_step_loader), total=min(len(n_step_loader), 100)):
        # Memory cleanup every 5 batches
        if batch_count % 5 == 0:
            clear_gpu_memory()

        obs, acts, next_obs, rewards, dones = n_step_trans[:5]

        # Further reduce batch size if needed
        if obs.shape[0] > 8:
            obs = obs[:8]
            acts = acts[:8]
            next_obs = next_obs[:8]
            rewards = rewards[:8]
            dones = dones[:8]

        gammas = (1 - dones) * GAMMA_CONST
        z = encoder_model.encode(obs[:, 0].to(args.device))
        if args.trans_model_type in DISCRETE_TRANS_TYPES:
            z_logits = F.one_hot(z, encoder_model.n_embeddings).permute(0, 2, 1).float() * 1e6
        else:
            z = z.reshape(z.shape[0], encoder_model.latent_dim)
            z_logits = z

        # Use memory-efficient loss calculation
        loss_dict = calculate_trans_losses_efficient(
            z, rewards[:, 0], gammas[:, 0], z_logits, z,
            rewards[:, 0], gammas[:, 0], obs[:, 0], args.trans_model_type, encoder_model,
            init_obs=obs[:, 0])
        update_losses(n_step_stats, loss_dict, args, 0)

        keep_idxs = set(range(obs.shape[0]))
        for step in range(args.eval_unroll_steps):
            next_z = encoder_model.encode(next_obs[:, step].to(args.device))
            if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
                next_z = next_z.reshape(next_z.shape[0], encoder_model.latent_dim)

            next_z_pred_logits, next_reward_pred, next_gamma_pred = \
                trans_model(z, acts[:, step].to(args.device), return_logits=True)
            next_z_pred = trans_model.logits_to_state(next_z_pred_logits)
            if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
                next_z_pred_logits = next_z_pred_logits.reshape(
                    next_z_pred_logits.shape[0], encoder_model.latent_dim)

            loss_dict = calculate_trans_losses_efficient(
                next_z, rewards[:, step], gammas[:, step],
                next_z_pred_logits, next_z_pred, next_reward_pred, next_gamma_pred,
                next_obs[:, step], args.trans_model_type, encoder_model,
                init_obs=obs[:, 0], curr_z=z, acts=acts[:, step])
            update_losses(n_step_stats, loss_dict, args, step + 1)

            z = next_z_pred

            # Remove transitions with finished episodes
            keep_idxs = (dones[:, step] == 0).float().nonzero().squeeze()
            if keep_idxs.numel() == 0:
                break
            obs, acts, next_obs, rewards, dones = \
                [x[keep_idxs] for x in (obs, acts, next_obs, rewards, dones)]
            gammas = gammas[keep_idxs]
            z = z[keep_idxs]

        n_full_unroll_samples += keep_idxs.numel()
        batch_count += 1

        # Early stopping for memory
        if n_full_unroll_samples >= EARLY_STOP_COUNT or batch_count >= 100:
            print(f'Early stopping at {n_full_unroll_samples} samples, {batch_count} batches')
            break

    # Upload the stats to logging server
    print('Publishing n-step stats to cloud...')
    for step in range(args.eval_unroll_steps + 1):
        keep_idxs = [i for i, n_step in enumerate(n_step_stats['step']) \
                     if n_step == step]
        if len(keep_idxs) > 0:
            log_vars = {k: np.nanmean(np.array(v)[keep_idxs]) \
                        for k, v in n_step_stats.items() \
                        if k not in ('step', 'model')}
            log_metrics({
                'n_step': step,
                **log_vars
            }, args, step=step)

    if len(n_step_stats['img_mse_loss']) > 0:
        log_metrics({'img_mse_loss_mean': np.mean(n_step_stats['img_mse_loss'])}, args)

    clear_gpu_memory()

    # Create sample transition images (much reduced)
    print('Creating sample transition images...')
    n_step_sampler = create_fast_loader(
        n_step_loader.dataset, batch_size=1, shuffle=True, num_workers=0,
        n_step=args.eval_unroll_steps, pin_memory=False, persistent_workers=False)

    samples = []
    sample_count = 0
    for i, sample_rollout in enumerate(n_step_sampler):
        if len(samples) >= min(N_EXAMPLE_IMGS, 3):  # Much fewer samples
            break
        if sample_rollout[0].numel() > 0:
            samples.append(sample_rollout)
            sample_count += 1
        if sample_count >= 3:  # Limit samples
            break

    if len(samples) == 0:
        print("Warning: No valid samples found for transition images")
    else:
        print(f"Processing {len(samples)} transition samples...")
        sample_rollouts = [torch.stack([x[i] for x in samples]).squeeze(dim=1) \
                           for i in range(len(samples[0]))]

        all_obs = torch.cat((sample_rollouts[0][:, :1], sample_rollouts[2]), dim=1)
        acts = sample_rollouts[1]
        dones = sample_rollouts[4]
        z = encoder_model.encode(all_obs[:, 0].to(args.device))
        if args.trans_model_type in CONTINUOUS_TRANS_TYPES:
            z = z.reshape(z.shape[0], encoder_model.latent_dim)

        # Convert hidden states to observations (if necessary)
        all_obs = [states_to_imgs(o, args.env_name, transform=rev_transform) for o in all_obs]
        all_obs = torch.from_numpy(np.stack(all_obs))

        example_trans_imgs = []
        example_trans_imgs.append(torch.cat((
            all_obs[:, 0], torch.zeros_like(all_obs[:, 0])), dim=3))

        continue_mask = torch.ones(all_obs.shape[0])
        for step in range(min(args.eval_unroll_steps, 3)):  # Limit steps
            with torch.no_grad():
                z = trans_model(z, acts[:, step].to(args.device))[0]
                pred_obs = encoder_model.decode(z).cpu()

            pred_obs = states_to_imgs(pred_obs, args.env_name, transform=rev_transform)
            pred_obs = torch.from_numpy(pred_obs)

            example_trans_imgs.append(torch.cat((
                all_obs[:, step + 1], pred_obs), dim=3) \
                                      * continue_mask[:, None, None, None])
            continue_mask[dones[:, step].float().nonzero().squeeze()] = 0

        example_trans_imgs = [
            torch.stack([x[i] for x in example_trans_imgs])
            for i in range(len(example_trans_imgs[0]))
        ]

        # Create individual full-size images for each sample
        for i, img in enumerate(example_trans_imgs):
            print(f"Creating transition image {i + 1}/{len(example_trans_imgs)}")

            # img shape: [time_steps, channels, height, width]
            print(f"Debug: Processing sample {i}, img shape: {img.shape}")

            # Convert to numpy and ensure proper range
            img = img.clip(0, 1).cpu().numpy()

            if img.ndim == 4:  # [time_steps, channels, height, width]
                # Process each time step
                processed_frames = []

                for t in range(img.shape[0]):
                    frame = img[t]  # [channels, height, width]

                    # Convert to [height, width, channels] for display
                    if frame.ndim == 3:
                        frame = np.transpose(frame, (1, 2, 0))

                    # Handle different channel configurations
                    if frame.ndim == 3:
                        if frame.shape[2] == 1:
                            frame = frame.squeeze(2)  # Grayscale
                        elif frame.shape[2] == 2:
                            frame = frame[:, :, -1]  # Take last channel
                        elif frame.shape[2] > 3:
                            frame = frame[:, :, -1]  # Take last channel if framestack

                    processed_frames.append(frame)

                # Create figure showing temporal sequence
                n_frames = len(processed_frames)

                # Create horizontal layout for temporal sequence
                fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6))
                if n_frames == 1:
                    axes = axes.reshape(2, 1)  # Ensure 2D array

                fig.suptitle(f'Sample {i + 1}: {min(args.eval_unroll_steps, 3)}-step Transition Sequence',
                             fontsize=16, y=0.95)

                # Determine if images are grayscale
                is_grayscale = processed_frames[0].ndim == 2
                cmap = 'gray' if is_grayscale else None

                for t, frame in enumerate(processed_frames):
                    # Split the concatenated frame into ground truth and prediction
                    if frame.ndim == 2:  # Grayscale
                        height, width = frame.shape
                        mid_width = width // 2
                        ground_truth = frame[:, :mid_width]
                        prediction = frame[:, mid_width:]
                    else:  # Color
                        height, width, channels = frame.shape
                        mid_width = width // 2
                        ground_truth = frame[:, :mid_width, :]
                        prediction = frame[:, mid_width:, :]

                    # Plot ground truth on top row
                    axes[0, t].imshow(ground_truth, cmap=cmap)
                    if t == 0:
                        axes[0, t].set_title(f'Step {t}\n(Initial)', fontsize=12)
                    else:
                        axes[0, t].set_title(f'Step {t}\n(Ground Truth)', fontsize=12)
                    axes[0, t].axis('off')

                    # Plot prediction on bottom row
                    axes[1, t].imshow(prediction, cmap=cmap)
                    if t == 0:
                        axes[1, t].set_title('(Empty)', fontsize=12)
                    else:
                        axes[1, t].set_title('(Prediction)', fontsize=12)
                    axes[1, t].axis('off')

                # Add row labels
                fig.text(0.02, 0.75, 'Ground\nTruth', fontsize=14, ha='center', va='center',
                         rotation=90, weight='bold')
                fig.text(0.02, 0.25, 'Model\nPrediction', fontsize=14, ha='center', va='center',
                         rotation=90, weight='bold')

                # Adjust layout
                plt.tight_layout()
                plt.subplots_adjust(top=0.85, left=0.08)

            else:
                # Handle non-4D case (fallback)
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))

                if img.ndim == 3:  # [channels, height, width]
                    display_img = np.transpose(img, (1, 2, 0))
                    if display_img.shape[2] == 1:
                        display_img = display_img.squeeze(2)
                        cmap = 'gray'
                    elif display_img.shape[2] > 3:
                        display_img = display_img[:, :, -1]
                        cmap = 'gray'
                    else:
                        cmap = None
                else:
                    display_img = img
                    cmap = 'gray'

                ax.imshow(display_img, cmap=cmap)
                ax.set_title(f'Sample {i + 1}: Transition', fontsize=16)
                ax.axis('off')

            # Don't save individual images to reduce I/O overhead
            plt.show()
            plt.close()  # Clean up memory

            # Skip video logging to save memory

        print(f"Created {len(example_trans_imgs)} individual transition images")

    # Final cleanup
    clear_gpu_memory()

    # Restore original batch sizes
    args.batch_size = original_batch_size
    args.eval_batch_size = original_eval_batch_size

    print("✅ Evaluation completed successfully!")
    print(f"🧠 Final {get_gpu_memory_info()}")


if __name__ == '__main__':
    # Parse args
    args = get_args()

    # Memory-efficient argument overrides
    print("🧠 Applying memory-efficient settings...")

    # Override problematic settings
    args.batch_size = min(getattr(args, 'batch_size', 256), 64)
    args.eval_batch_size = min(getattr(args, 'eval_batch_size', 128), 32)
    args.eval_unroll_steps = min(getattr(args, 'eval_unroll_steps', 20), 5)
    args.max_transitions = min(getattr(args, 'max_transitions', None) or 10000, 5000)

    # Force disable memory-intensive features
    args.exact_comp = False
    args.log_state_reprs = False
    args.preload_data = False
    args.pin_memory = False
    args.persistent_workers = False

    print(f"🔧 Memory-efficient settings applied:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Eval batch size: {args.eval_batch_size}")
    print(f"   Eval unroll steps: {args.eval_unroll_steps}")
    print(f"   Max transitions: {args.max_transitions}")
    print(f"   Exact comp: {args.exact_comp}")

    # Setup logging
    args = init_experiment('discrete-mbrl-eval', args)

    try:
        # Evaluate the models
        eval_model(args)
        print("🎉 Evaluation completed successfully!")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ Still running out of GPU memory. Try:")
            print("   1. Use --device cpu")
            print("   2. Further reduce --batch_size to 16 or 8")
            print("   3. Reduce --eval_unroll_steps to 3")
            print("   4. Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256")
        raise e
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        raise e
    finally:
        # Clean up logging
        clear_gpu_memory()
        finish_experiment(args)