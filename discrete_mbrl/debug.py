# 1. Environment Testing Script - Run this first to identify the issue

import os
import sys
import traceback
import torch
import numpy as np
from pathlib import Path


def test_environment_step_by_step(env_name):
    """Test each component that might cause hanging"""
    print(f"🧪 Testing environment: {env_name}")

    try:
        # Step 1: Test basic environment creation
        print("📍 Step 1: Testing basic environment creation...")
        from env_helpers import make_env
        env = make_env(env_name)
        print(f"✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")

        # Step 2: Test environment reset
        print("📍 Step 2: Testing environment reset...")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, info = obs
        print(f"✅ Environment reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation dtype: {obs.dtype}")
        print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

        # Step 3: Test environment step
        print("📍 Step 3: Testing environment step...")
        action = env.action_space.sample()
        step_result = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   Step result length: {len(step_result)}")

        # Step 4: Test multiple steps
        print("📍 Step 4: Testing multiple environment steps...")
        for i in range(10):
            action = env.action_space.sample()
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs, info = obs
                print(f"   Episode ended at step {i}, reset successful")
                break
        print(f"✅ Multiple environment steps successful")

        env.close()

    except Exception as e:
        print(f"❌ Environment test failed at current step: {e}")
        traceback.print_exc()
        return False

    return True


def test_data_loading(env_name):
    """Test data loading components"""
    print(f"📍 Testing data loading for: {env_name}")

    try:
        # Check if replay buffer exists
        sanitized_env_name = env_name.replace(':', '_')
        replay_buffer_path = f'./data/{sanitized_env_name}_replay_buffer.hdf5'

        if not os.path.exists(replay_buffer_path):
            print(f"❌ Replay buffer not found: {replay_buffer_path}")
            print("   This is likely the cause of hanging!")
            return False

        print(f"✅ Replay buffer found: {replay_buffer_path}")

        # Test opening the HDF5 file
        import h5py
        with h5py.File(replay_buffer_path, 'r') as f:
            print(f"   Data keys: {list(f.keys())}")
            if 'obs' in f:
                print(f"   Obs shape: {f['obs'].shape}")
                print(f"   Data index: {f.attrs.get('data_idx', 'Not found')}")

        return True

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        traceback.print_exc()
        return False


def collect_data_for_environment(env_name, n_steps=50000):
    """Collect replay buffer data for the environment"""
    print(f"🗂️  Collecting data for environment: {env_name}")

    try:
        # Import data collection script
        sys.path.append('.')
        from collect_data import setup_replay_buffer
        from env_helpers import make_env
        from stable_baselines3.common.vec_env import DummyVecEnv
        from threading import Lock

        # Create a simple args object
        class Args:
            def __init__(self):
                self.env_name = env_name
                self.train_steps = n_steps
                self.chunk_size = 2048
                self.compression_type = 'lzf'
                self.extra_info = []
                self.env_max_steps = None
                self.algorithm = 'random'
                self.n_envs = 4  # Use multiple environments for faster collection

        args = Args()

        # Setup replay buffer
        print("   Setting up replay buffer...")
        replay_buffer = setup_replay_buffer(args)
        buffer_lock = Lock()

        # Create vectorized environment for faster data collection
        print("   Creating environments...")
        venv = DummyVecEnv([
            lambda: make_env(env_name, replay_buffer, buffer_lock,
                             extra_info=args.extra_info, monitor=True,
                             max_steps=args.env_max_steps)
            for _ in range(args.n_envs)
        ])

        # Collect random data
        print(f"   Collecting {n_steps} random transitions...")
        from training_helpers import vec_env_random_walk
        vec_env_random_walk(venv, n_steps, progress=True)

        print("   Closing replay buffer...")
        replay_buffer.close()

        print(f"✅ Data collection completed for {env_name}")
        return True

    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        traceback.print_exc()
        return False


def debug_training_start(env_name):
    """Debug the training initialization process"""
    print(f"🔍 Debugging training initialization for: {env_name}")

    try:
        # Test data loader creation
        print("📍 Testing data loader creation...")
        from data_helpers import prepare_dataloaders

        # Use small batch size and limited data for testing
        train_loader, test_loader, valid_loader = prepare_dataloaders(
            env_name,
            n=1000,  # Limit to 1000 transitions for testing
            batch_size=32,  # Small batch size
            n_step=1,  # Single step for simplicity
            preprocess=False,
            randomize=True,
            n_preload=0,  # No multiprocessing to avoid hanging
            preload_all=False,
            extra_buffer_keys=[]
        )

        print(f"✅ Data loaders created successfully")
        print(f"   Train dataset size: {len(train_loader.dataset)}")
        print(f"   Test dataset size: {len(test_loader.dataset)}")
        print(f"   Valid dataset size: {len(valid_loader.dataset)}")

        # Test getting first batch
        print("📍 Testing first batch loading...")
        first_batch = next(iter(train_loader))
        print(f"✅ First batch loaded successfully")
        print(f"   Batch shape: {first_batch[0].shape}")

        return True

    except Exception as e:
        print(f"❌ Training initialization debug failed: {e}")
        traceback.print_exc()
        return False


# Main debugging function
def debug_minigrid_environment(env_name="MiniGrid-MultiRoom-N2-S4-v0"):
    """Complete debugging workflow"""
    print("🚀 Starting MiniGrid Environment Debugging")
    print("=" * 60)

    # Step 1: Test environment
    if not test_environment_step_by_step(env_name):
        print("🛑 Environment test failed - fix environment setup first")
        return False

    # Step 2: Test data loading
    if not test_data_loading(env_name):
        print("🔧 Data loading failed - collecting data...")
        if not collect_data_for_environment(env_name):
            print("🛑 Data collection failed")
            return False

    # Step 3: Test training initialization
    if not debug_training_start(env_name):
        print("🛑 Training initialization failed")
        return False

    print("✅ All debugging tests passed!")
    print("🎉 Environment should work now")
    return True


# 2. Quick fix for missing data - add to your training script

def ensure_data_exists(env_name, min_transitions=50000):
    """Ensure replay buffer data exists for the environment"""
    sanitized_env_name = env_name.replace(':', '_')
    replay_buffer_path = f'./data/{sanitized_env_name}_replay_buffer.hdf5'

    if not os.path.exists(replay_buffer_path):
        print(f"⚠️  No data found for {env_name}")
        print(f"🗂️  Collecting {min_transitions} transitions...")

        # Quick data collection
        os.system(f"""python collect_data.py \
            --env_name {env_name} \
            --train_steps {min_transitions} \
            --algorithm random \
            --n_envs 8""")

        if os.path.exists(replay_buffer_path):
            print(f"✅ Data collection completed")
        else:
            raise FileNotFoundError(f"Failed to create data for {env_name}")


# 3. Robust environment initialization

def robust_make_env(env_name, max_retries=3, **kwargs):
    """Robustly create environment with retries and error handling"""
    from env_helpers import make_env

    for attempt in range(max_retries):
        try:
            print(f"🔄 Creating environment {env_name} (attempt {attempt + 1}/{max_retries})")
            env = make_env(env_name, **kwargs)

            # Test the environment
            obs = env.reset()
            if isinstance(obs, tuple):
                obs, info = obs

            # Test a few steps
            for _ in range(3):
                action = env.action_space.sample()
                step_result = env.step(action)
                if len(step_result) >= 4:
                    obs = step_result[0]
                else:
                    raise ValueError(f"Unexpected step result: {step_result}")

            print(f"✅ Environment created and tested successfully")
            return env

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise

            # Wait before retry
            import time
            time.sleep(1)


# 4. Modified training script with debugging

def debug_and_train(env_name="MiniGrid-MultiRoom-N2-S4-v0"):
    """Training with comprehensive debugging"""

    print(f"🏋️  Starting training for {env_name}")

    # Step 1: Ensure data exists
    try:
        ensure_data_exists(env_name)
    except Exception as e:
        print(f"❌ Data setup failed: {e}")
        return False

    # Step 2: Test environment creation
    try:
        env = robust_make_env(env_name)
        env.close()
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return False

    # Step 3: Start training with timeout protection
    try:
        # Your existing training code here
        from full_train_eval import main
        main()

    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Run debugging
    debug_minigrid_environment("MiniGrid-MultiRoom-N2-S4-v0")