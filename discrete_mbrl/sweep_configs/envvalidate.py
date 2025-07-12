#!/usr/bin/env python3
"""
Environment Validation Program
Tests your environment setup to identify API compatibility issues.
"""

import traceback
import numpy as np
import gym
from collections import defaultdict

# Import your environment helpers
try:
    from env_helpers import make_env, Custom2DWrapper, MiniGridSimpleStochActionWrapper

    print("✓ Successfully imported env_helpers")
except ImportError as e:
    print(f"✗ Failed to import env_helpers: {e}")
    exit(1)


def test_basic_gym_apis():
    """Test basic gym API compatibility"""
    print("\n=== Testing Basic Gym APIs ===")

    # Test basic gym environment
    try:
        env = gym.make('CartPole-v1')
        print("✓ Basic gym.make() works")

        # Test reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
            print("✓ Reset returns tuple (new API)")
            print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        else:
            obs = reset_result
            print("✓ Reset returns observation only (old API)")
            print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")

        # Test step
        action = env.action_space.sample()
        step_result = env.step(action)
        print(f"✓ Step returns {len(step_result)} values")

        if len(step_result) == 4:
            obs, reward, done, info = step_result
            print("  Format: obs, reward, done, info (old API)")
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            print("  Format: obs, reward, terminated, truncated, info (new API)")
        else:
            print(f"  ✗ Unexpected format with {len(step_result)} values")

        env.close()

    except Exception as e:
        print(f"✗ Basic gym test failed: {e}")
        traceback.print_exc()


def test_minigrid_environment():
    """Test minigrid environment directly"""
    print("\n=== Testing MiniGrid Environment ===")

    try:
        import minigrid
        print("✓ Minigrid imported successfully")

        env = gym.make('MiniGrid-Empty-6x6-v0')
        print("✓ MiniGrid environment created")

        # Test reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
            print("✓ MiniGrid reset returns tuple (new API)")
        else:
            obs = reset_result
            print("✓ MiniGrid reset returns observation only (old API)")

        print(f"  Observation type: {type(obs)}")
        if hasattr(obs, 'keys'):
            print(f"  Observation keys: {list(obs.keys())}")
            if 'image' in obs:
                print(f"  Image shape: {obs['image'].shape}")

        # Test step
        action = env.action_space.sample()
        step_result = env.step(action)
        print(f"✓ MiniGrid step returns {len(step_result)} values")

        env.close()

    except Exception as e:
        print(f"✗ MiniGrid test failed: {e}")
        traceback.print_exc()


def test_wrapper_chain():
    """Test individual wrappers in the chain"""
    print("\n=== Testing Wrapper Chain ===")

    # Test each wrapper individually
    wrappers_to_test = [
        ("Base MiniGrid", lambda: gym.make('MiniGrid-Empty-6x6-v0')),
        ("With MiniGridSimpleStochActionWrapper",
         lambda: MiniGridSimpleStochActionWrapper(gym.make('MiniGrid-Empty-6x6-v0'), n_acts=3)),
    ]

    for wrapper_name, env_factory in wrappers_to_test:
        print(f"\n--- Testing {wrapper_name} ---")
        try:
            env = env_factory()

            # Test reset
            reset_result = env.reset()
            reset_format = "tuple" if isinstance(reset_result, tuple) else "single"
            print(f"✓ Reset works, returns {reset_format}")

            # Test step
            action = env.action_space.sample()
            step_result = env.step(action)
            print(f"✓ Step works, returns {len(step_result)} values")

            env.close()

        except Exception as e:
            print(f"✗ {wrapper_name} failed: {e}")
            traceback.print_exc()


def test_full_environment():
    """Test the full environment from make_env function"""
    print("\n=== Testing Full Environment ===")

    env_names_to_test = [
        'minigrid-crossing-stochastic',
        'MiniGrid-Empty-6x6-v0'
    ]

    for env_name in env_names_to_test:
        print(f"\n--- Testing {env_name} ---")
        try:
            env = make_env(env_name)
            print(f"✓ Environment {env_name} created successfully")

            # Test reset
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
                print("✓ Reset returns tuple")
            else:
                obs = reset_result
                print("✓ Reset returns observation only")

            print(f"  Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"  Observation dtype: {obs.dtype if hasattr(obs, 'dtype') else 'N/A'}")

            # Test multiple steps
            for i in range(5):
                action = env.action_space.sample()
                step_result = env.step(action)

                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                elif len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    print(f"✗ Unexpected step result length: {len(step_result)}")
                    break

                print(f"  Step {i + 1}: reward={reward:.3f}, done={done}")

                if done:
                    print("  Episode ended, resetting...")
                    reset_result = env.reset()
                    if isinstance(reset_result, tuple):
                        obs, info = reset_result
                    else:
                        obs = reset_result
                    break

            print(f"✓ Successfully completed test for {env_name}")
            env.close()

        except Exception as e:
            print(f"✗ {env_name} failed: {e}")
            traceback.print_exc()


def test_api_compatibility():
    """Test API compatibility detection"""
    print("\n=== Testing API Compatibility Detection ===")

    def detect_api_format(env):
        """Helper to detect which API format an environment uses"""
        try:
            obs = env.reset()
            reset_api = "new" if isinstance(obs, tuple) else "old"

            action = env.action_space.sample()
            step_result = env.step(action)

            if len(step_result) == 4:
                step_api = "old"
            elif len(step_result) == 5:
                step_api = "new"
            else:
                step_api = f"unknown ({len(step_result)} values)"

            return reset_api, step_api

        except Exception as e:
            return f"error: {e}", f"error: {e}"

    environments = [
        ("CartPole-v1", lambda: gym.make('CartPole-v1')),
        ("MiniGrid-Empty-6x6-v0", lambda: gym.make('MiniGrid-Empty-6x6-v0')),
    ]

    for env_name, env_factory in environments:
        try:
            env = env_factory()
            reset_api, step_api = detect_api_format(env)
            print(f"{env_name:25} | Reset: {reset_api:8} | Step: {step_api}")
            env.close()
        except Exception as e:
            print(f"{env_name:25} | Error: {e}")


def main():
    """Run all validation tests"""
    print("🔍 Environment Validation Program")
    print("=" * 50)

    test_basic_gym_apis()
    test_minigrid_environment()
    test_api_compatibility()
    test_wrapper_chain()
    test_full_environment()

    print("\n" + "=" * 50)
    print("✅ Validation complete! Check output above for any issues.")


if __name__ == "__main__":
    main()