from collections import defaultdict
import os
import sys

from discrete_mbrl.env_helpers import preprocess_obs

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from torch.distributions import Categorical
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from data import SizedReplayBuffer as ReplayBuffer
from ppo import ortho_init, PPOTrainer
from data_logging import *
from shared.models import *
from shared.trainers import *
from env_helpers import *
from training_helpers import *
from model_construction import *
from rl_utils import *


def save_best_model(ae_model, policy, critic, optimizer, step, args, avg_reward, suffix="best"):
    """Save the best model"""
    checkpoint = {
        'step': step,
        'avg_reward': avg_reward,
        'ae_model_state_dict': ae_model.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'model_info': {
            'ae_params': sum(p.numel() for p in ae_model.parameters()),
            'policy_params': sum(p.numel() for p in policy.parameters()),
            'critic_params': sum(p.numel() for p in critic.parameters()),
        }
    }

    # Create save directory
    save_dir = f"./models/{args.env_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save best model
    save_path = f"{save_dir}/{suffix}_model_reward_{avg_reward:.4f}.pt"
    torch.save(checkpoint, save_path)
    print(f"🏆 Saved {suffix} model to: {save_path}")

    return save_path


def train(args, encoder_model=None):
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    # env = FreezeOnDoneWrapper(env, max_count=1)
    act_space = env.action_space
    act_dim = act_space.n
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        sample_obs, _ = reset_result
    else:
        sample_obs = reset_result
    sample_obs = preprocess_obs([sample_obs])

    freeze_encoder = False

    # Load the encoder
    if encoder_model is None:
        ae_model, ae_trainer = construct_ae_model(
            sample_obs.shape[1:], args, latent_activation=True, load=args.load)
        if ae_trainer is not None:
            ae_trainer.log_freq = -1
    else:
        ae_model = encoder_model
    torch.compile(ae_model)

    if freeze_encoder:
        freeze_model(ae_model)
        ae_model.eval()
    else:
        ae_model = ae_model.to(args.device)
        ae_model.train()
    print(f'Loaded encoder')

    update_params(args)

    mlp_kwargs = {
        'activation': args.rl_activation,
        'discrete_input': args.ae_model_type == 'vqvae',
    }
    if args.ae_model_type == 'vqvae':
        mlp_kwargs['n_embeds'] = args.codebook_size
        mlp_kwargs['embed_dim'] = args.embedding_dim
        input_dim = args.embedding_dim * ae_model.n_latent_embeds
    else:
        input_dim = ae_model.latent_dim

    policy = mlp(
        [input_dim] + args.policy_hidden + [act_dim],
        **mlp_kwargs)
    policy = policy.to(args.device)
    torch.compile(policy)

    critic = mlp(
        [input_dim] + args.critic_hidden + [1],
        **mlp_kwargs)
    critic = critic.to(args.device)
    torch.compile(critic)

    if args.ortho_init:
        ortho_init(ae_model, policy, critic)

    # Print number of params in ae_model, policy, and critic separately

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'AE Model Params: {count_params(ae_model)}')
    print(f'Policy Params: {count_params(policy)}')
    print(f'Critic Params: {count_params(critic)}')

    # Create the optimizer(s)

    all_params = list(policy.parameters())
    all_params += list(critic.parameters())
    if args.e2e_loss:
        all_params += list(ae_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.learning_rate, eps=1e-5)

    ppo = PPOTrainer(
        env, policy, critic, ae_model, optimizer,
        ppo_iters=args.ppo_iters,
        ppo_clip=args.ppo_clip,
        minibatch_size=args.ppo_batch_size,
        value_coef=args.ppo_value_coef,
        entropy_coef=args.ppo_entropy_coef,
        gae_lambda=args.ppo_gae_lambda,
        norm_advantages=args.ppo_norm_advantages,
        max_grad_norm=args.ppo_max_grad_norm,
        e2e_loss=args.e2e_loss,
    )

    if args.ae_er_train:
        replay_buffer = ReplayBuffer(args.replay_size)
    else:
        replay_buffer = None

    run_stats = defaultdict(list)
    ep_info = defaultdict(list)

    # Track best performance
    best_avg_reward = float('-inf')
    recent_rewards = []  # Keep track of recent episode rewards
    reward_window = 10  # Number of episodes to average over

    # Rollout loop

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        curr_obs, _ = reset_result
    else:
        curr_obs = reset_result

    curr_obs = torch.from_numpy(curr_obs).float()
    # Batch next_obs, rewards, acts, gammas
    ep_rewards = []
    n_batches = int(np.ceil(args.mf_steps / args.batch_size))
    step = 0
    for batch in tqdm(range(n_batches)):
        ae_model.eval()
        policy.eval()
        log_data = defaultdict(list)
        batch_data = {k: [] for k in [
            'obs', 'states', 'next_obs', 'rewards', 'acts', 'gammas']}

        # ae_model.cpu()
        # policy.cpu()

        for _ in range(args.batch_size):
            with torch.no_grad():
                env_change = \
                    isinstance(args.env_change_freq, int) and \
                    args.env_change_freq > 0 and \
                    (step + 1) % args.env_change_freq == 0

                model_device = next(ae_model.parameters()).device
                state = ae_model.encode(
                    curr_obs.unsqueeze(0).to(model_device), return_one_hot=True)

                act_logits = policy(state)
                act_dist = Categorical(logits=act_logits)
                act = act_dist.sample().cpu()

            batch_data['obs'].append(curr_obs)
            batch_data['states'].append(state.squeeze(0))
            batch_data['acts'].append(act)

            # Take the action
            step_result = env.step(act)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            done = done or env_change
            next_obs = torch.from_numpy(next_obs).float()
            ep_rewards.append(reward)

            batch_data['next_obs'].append(next_obs)
            batch_data['rewards'].append(torch.tensor(reward).float())
            batch_data['gammas'].append(
                torch.tensor(args.gamma * (1 - done)).float())

            # Log achievement info for crafter
            if 'achievements' in info:
                for k, v in info['achievements'].items():
                    run_stats[f'achievement/{k}'].append(v)
                    ep_info[f'achievement/{k}'].append(v)

            # Update replay buffer
            if replay_buffer is not None:
                replay_buffer.add_step(
                    curr_obs, act, next_obs, batch_data['rewards'][-1], batch_data['gammas'][-1])

            # Update the current obs
            if done:
                if replay_buffer is not None:
                    replay_buffer.add_step(
                        next_obs, act, next_obs, batch_data['rewards'][-1], batch_data['gammas'][-1])

                if env_change or args.env_change_freq == 'episode':
                    if args.env_change_type == 'random':
                        env.seeds = [np.random.randint(0, 1000000)]
                    elif args.env_change_type == 'next':
                        env.seeds = [env.seeds[0] + 1]
                    else:
                        raise ValueError(f'Invalid env change type: {args.env_change_type}')

                # Fixed reset handling
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    curr_obs, _ = reset_result
                else:
                    curr_obs = reset_result
                curr_obs = torch.from_numpy(curr_obs).float()

                # Calculate episode reward and update best tracking
                episode_reward = np.sum(ep_rewards)
                recent_rewards.append(episode_reward)

                # Keep only recent rewards
                if len(recent_rewards) > reward_window:
                    recent_rewards.pop(0)

                # Calculate current average
                current_avg_reward = np.mean(recent_rewards)

                # Update best if this is the best performance so far
                if current_avg_reward > best_avg_reward:
                    best_avg_reward = current_avg_reward
                    print(f"🎯 New best average reward: {best_avg_reward:.4f} (over {len(recent_rewards)} episodes)")

                run_stats['ep_length'].append(len(ep_rewards))
                run_stats['ep_reward'].append(episode_reward)

                # Compute score for crafter
                if 'crafter' in args.env_name.lower():
                    achievement_keys = [k for k in ep_info.keys() if 'achievement' in k]
                    percents = np.array([np.mean(ep_info[k]) * 100 for k in achievement_keys])
                    score = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
                    run_stats['achievement/score'].append(score)

                print('\n--- Episode Stats ---')
                print(f'Reward: {episode_reward:.4f}')
                print(f'Length: {len(ep_rewards)}')
                print(f'Avg Reward (last {len(recent_rewards)}): {current_avg_reward:.4f}')
                print(f'Best Avg Reward: {best_avg_reward:.4f}')

                ep_rewards = []
                ep_info = defaultdict(list)

            else:
                curr_obs = next_obs

            update_stats(run_stats, {
                'reward': reward,
            })

            # Log and reset logging data
            if step > 0 and step % args.log_freq == 0:
                # Compute score for crafter
                if 'crafter' in args.env_name.lower():
                    achievement_keys = [k for k in run_stats.keys() if 'achievement' in k]
                    percents = np.array([np.mean(run_stats[k]) * 100 for k in achievement_keys])
                    score = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
                    run_stats['achievement/score'].append(score)

                log_stats(run_stats, step, args)
                run_stats = defaultdict(list)

                # Only save image reconstructions, no model saving during training
                if step % (args.log_freq * args.checkpoint_freq) == 0:
                    recons = sample_recon_imgs(
                        ae_model, batch_data['obs'], env_name=args.env_name)
                    log_images({'img_recon': recons}, args, step=step)

            step += 1
        ae_model.train()
        policy.train()

        ae_model.to(args.device)
        policy.to(args.device)

        batch_data = {k: torch.stack(v).to(args.device) \
                      for k, v in batch_data.items()}

        ### PPO Updates ###

        if step >= args.rl_start_step:
            loss_dict = ppo.train(batch_data)

            for k, v in loss_dict.items():
                run_stats[k].append(v.item())

        ### AE Reconstruction Loss ###

        if args.ae_recon_loss:

            for _ in range(args.n_ae_updates):

                if args.ae_er_train:
                    # Sample a batch of data from the replay buffer
                    batch_data = replay_buffer.sample(args.ae_batch_size or args.batch_size)
                    batch_obs = batch_data[0]
                    batch_next_obs = batch_data[2]
                else:
                    # Use the most recent batch data
                    batch_obs = batch_data['obs']
                    batch_next_obs = batch_data['next_obs']

                # Calculate loss
                loss_dict, ae_stats = ae_trainer.train((batch_obs, None, batch_next_obs))

                for k, v in {**loss_dict, **ae_stats}.items():
                    run_stats[k].append(v.item())

    # Save only the final best model
    if args.save:
        # Calculate final average over all episodes if we have any
        if run_stats['ep_reward']:
            final_avg_reward = np.mean(run_stats['ep_reward'][-reward_window:])  # Last N episodes
            overall_avg_reward = np.mean(run_stats['ep_reward'])  # All episodes

            print(f"\n🏁 Training Complete!")
            print(f"   Final {reward_window}-episode average: {final_avg_reward:.4f}")
            print(f"   Overall average reward: {overall_avg_reward:.4f}")
            print(f"   Best rolling average: {best_avg_reward:.4f}")

            # Save the final model (regardless of whether it's the best)
            final_save_path = save_best_model(ae_model, policy, critic, optimizer, step, args,
                                              final_avg_reward, suffix="final")

            # If final performance is the best, also save as best
            if abs(final_avg_reward - best_avg_reward) < 0.001:  # Small tolerance for floating point
                print("🎉 Final model is also the best model!")
            else:
                print(f"ℹ️  Final model saved. Best rolling average was {best_avg_reward:.4f}")
        else:
            print("⚠️  No episodes completed, saving final model anyway")
            save_best_model(ae_model, policy, critic, optimizer, step, args, 0.0, suffix="final")

    return policy, critic


if __name__ == '__main__':
    # Parse args
    mf_arg_parser = make_mf_arg_parser()
    args = get_args(mf_arg_parser)

    if args.env_change_freq.isdecimal():
        args.env_change_freq = int(args.env_change_freq)

    # Setup logging
    args = init_experiment('discrete-mbrl-model-free', args)

    if args.wandb:
        args.update({'policy_hidden': interpret_layer_sizes(args.policy_hidden)},
                    allow_val_change=True)
        args.update({'critic_hidden': interpret_layer_sizes(args.critic_hidden)},
                    allow_val_change=True)
    else:
        args.policy_hidden = interpret_layer_sizes(args.policy_hidden)
        args.critic_hidden = interpret_layer_sizes(args.critic_hidden)

    # Train and test the model
    policy, critic = train(args)