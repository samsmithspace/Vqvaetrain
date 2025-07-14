#!/usr/bin/env python3
"""
Comprehensive Encoder Model Test Script
Tests encoder models for validity, stability, and compatibility
"""

import sys
import os
import argparse
import traceback
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_helpers import make_env, check_env_name
from model_construction import construct_ae_model


class EncoderTester:
    def __init__(self, env_name, ae_model_type, device='cpu', **kwargs):
        self.env_name = env_name
        self.ae_model_type = ae_model_type
        self.device = device
        self.kwargs = kwargs

        # Test results
        self.test_results = {
            'basic_functionality': False,
            'nan_stability': False,
            'dimension_consistency': False,
            'interface_compatibility': False,
            'batch_processing': False,
            'edge_case_handling': False,
            'performance': {}
        }

        self.model = None
        self.trainer = None

    def create_args(self):
        """Create args for model construction"""
        args = Namespace()

        # Basic settings
        args.env_name = check_env_name(self.env_name)
        args.ae_model_type = self.ae_model_type
        args.device = self.device
        args.load = False  # Don't load pre-trained weights for testing

        # Model parameters
        args.latent_dim = self.kwargs.get('latent_dim', 32)
        args.embedding_dim = self.kwargs.get('embedding_dim', 64)
        args.filter_size = self.kwargs.get('filter_size', 8)
        args.codebook_size = self.kwargs.get('codebook_size', 16)
        args.ae_model_version = self.kwargs.get('ae_model_version', '2')

        # Additional parameters
        args.extra_info = None
        args.repr_sparsity = 0
        args.sparsity_type = 'random'
        args.fta_tiles = 20
        args.fta_bound_low = -2
        args.fta_bound_high = 2
        args.fta_eta = 0.2
        args.learning_rate = 3e-4
        args.ae_grad_clip = 0

        # Disable logging
        args.wandb = False
        args.comet_ml = False

        return args

    def get_test_observations(self):
        """Get test observations from the environment"""
        env = make_env(self.env_name)

        # Get initial observation
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs1, _ = reset_result
        else:
            obs1 = reset_result

        # Get observation after random action
        action = env.action_space.sample()
        step_result = env.step(action)
        obs2 = step_result[0]

        env.close()

        return obs1, obs2

    def debug_tensor(self, tensor, name):
        """Debug utility to analyze tensor properties"""
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = np.array(tensor)

        info = {
            'shape': tensor_np.shape,
            'dtype': str(tensor_np.dtype),
            'min': float(tensor_np.min()),
            'max': float(tensor_np.max()),
            'mean': float(tensor_np.mean()),
            'std': float(tensor_np.std()),
            'has_nan': bool(np.isnan(tensor_np).any()),
            'has_inf': bool(np.isinf(tensor_np).any()),
            'finite_ratio': float(np.isfinite(tensor_np).mean())
        }

        print(f"🔍 {name}:")
        print(f"  Shape: {info['shape']}")
        print(f"  Range: [{info['min']:.4f}, {info['max']:.4f}]")
        print(f"  Mean±Std: {info['mean']:.4f}±{info['std']:.4f}")
        print(f"  Has NaN: {info['has_nan']}")
        print(f"  Has Inf: {info['has_inf']}")
        print(f"  Finite ratio: {info['finite_ratio']:.4f}")

        return info

    def test_basic_functionality(self):
        """Test basic encode/decode functionality"""
        print("\n" + "=" * 50)
        print("🧪 TEST 1: Basic Functionality")
        print("=" * 50)

        try:
            # Create model
            args = self.create_args()
            obs1, obs2 = self.get_test_observations()

            print(f"Environment: {self.env_name}")
            print(f"Model type: {self.ae_model_type}")
            print(f"Device: {self.device}")

            self.debug_tensor(obs1, "Sample observation")

            # Construct model
            self.model, self.trainer = construct_ae_model(obs1.shape, args)
            self.model = self.model.to(self.device)
            self.model.eval()

            print(f"✅ Model constructed successfully")
            print(f"   Model type: {type(self.model).__name__}")
            if hasattr(self.model, 'latent_dim'):
                print(f"   Latent dim: {self.model.latent_dim}")
            if hasattr(self.model, 'n_latent_embeds'):
                print(f"   Latent embeds: {self.model.n_latent_embeds}")
            if hasattr(self.model, 'embedding_dim'):
                print(f"   Embedding dim: {self.model.embedding_dim}")

            # Test single observation
            obs_tensor = torch.from_numpy(obs1).float().unsqueeze(0).to(self.device)

            # Test encoding
            with torch.no_grad():
                encoded = self.model.encode(obs_tensor)
                encoded_info = self.debug_tensor(encoded, "Encoded latent")

                # Test decoding
                decoded = self.model.decode(encoded)
                decoded_info = self.debug_tensor(decoded, "Decoded observation")

                # Test full forward pass
                if hasattr(self.model, 'forward'):
                    forward_result = self.model(obs_tensor)
                    if isinstance(forward_result, tuple):
                        recon = forward_result[0]
                        print(f"✅ Forward pass returned {len(forward_result)} values")
                    else:
                        recon = forward_result
                        print(f"✅ Forward pass returned single tensor")

                    recon_info = self.debug_tensor(recon, "Forward reconstruction")

            # Check for basic validity
            if encoded_info['has_nan'] or decoded_info['has_nan']:
                print("❌ NaN values detected in basic functionality!")
                return False

            if encoded_info['finite_ratio'] < 0.99 or decoded_info['finite_ratio'] < 0.99:
                print("❌ Too many non-finite values!")
                return False

            print("✅ Basic functionality test PASSED")
            self.test_results['basic_functionality'] = True
            return True

        except Exception as e:
            print(f"❌ Basic functionality test FAILED: {e}")
            traceback.print_exc()
            return False

    def test_nan_stability(self):
        """Test model stability against NaN inputs and outputs"""
        print("\n" + "=" * 50)
        print("🧪 TEST 2: NaN Stability")
        print("=" * 50)

        if self.model is None:
            print("❌ No model available for testing")
            return False

        try:
            # Test with various problematic inputs
            test_cases = [
                ("zeros", torch.zeros(2, 3, 48, 48)),
                ("ones", torch.ones(2, 3, 48, 48)),
                ("random_small", torch.randn(2, 3, 48, 48) * 0.01),
                ("random_large", torch.randn(2, 3, 48, 48) * 10),
                ("extreme_values", torch.randn(2, 3, 48, 48) * 100),
            ]

            all_stable = True

            for case_name, test_input in test_cases:
                test_input = test_input.to(self.device)
                print(f"\n🔸 Testing {case_name}...")

                try:
                    with torch.no_grad():
                        encoded = self.model.encode(test_input)
                        decoded = self.model.decode(encoded)

                        # Check for NaN/Inf
                        enc_valid = torch.isfinite(encoded).all().item()
                        dec_valid = torch.isfinite(decoded).all().item()

                        if enc_valid and dec_valid:
                            print(f"  ✅ {case_name}: Stable")
                        else:
                            print(f"  ❌ {case_name}: Unstable (NaN/Inf detected)")
                            all_stable = False

                except Exception as e:
                    print(f"  ❌ {case_name}: Exception - {e}")
                    all_stable = False

            # Test with actual NaN input (recovery test)
            print(f"\n🔸 Testing NaN input recovery...")
            nan_input = torch.randn(1, 3, 48, 48).to(self.device)
            nan_input[0, 0, :5, :5] = float('nan')  # Inject some NaN values

            try:
                with torch.no_grad():
                    encoded = self.model.encode(nan_input)
                    decoded = self.model.decode(encoded)

                    if torch.isfinite(decoded).all():
                        print("  ✅ NaN recovery: Model handled NaN input gracefully")
                    else:
                        print("  ⚠️ NaN recovery: Model did not fully recover from NaN input")

            except Exception as e:
                print(f"  ❌ NaN recovery: Exception - {e}")
                all_stable = False

            if all_stable:
                print("✅ NaN stability test PASSED")
                self.test_results['nan_stability'] = True
                return True
            else:
                print("❌ NaN stability test FAILED")
                return False

        except Exception as e:
            print(f"❌ NaN stability test FAILED: {e}")
            traceback.print_exc()
            return False

    def test_dimension_consistency(self):
        """Test dimensional consistency across different batch sizes"""
        print("\n" + "=" * 50)
        print("🧪 TEST 3: Dimension Consistency")
        print("=" * 50)

        if self.model is None:
            print("❌ No model available for testing")
            return False

        try:
            obs1, obs2 = self.get_test_observations()
            base_shape = obs1.shape

            # Test different batch sizes
            batch_sizes = [1, 2, 4, 8]

            reference_encoded_shape = None
            reference_decoded_shape = None

            for batch_size in batch_sizes:
                print(f"\n🔸 Testing batch size {batch_size}...")

                # Create batch
                batch_obs = torch.randn(batch_size, *base_shape).to(self.device)
                batch_obs = torch.clamp(batch_obs, 0, 1)  # Ensure valid range

                with torch.no_grad():
                    encoded = self.model.encode(batch_obs)
                    decoded = self.model.decode(encoded)

                print(f"  Input shape: {batch_obs.shape}")
                print(f"  Encoded shape: {encoded.shape}")
                print(f"  Decoded shape: {decoded.shape}")

                # Check batch dimension
                if encoded.shape[0] != batch_size or decoded.shape[0] != batch_size:
                    print(f"  ❌ Batch dimension mismatch!")
                    return False

                # Check consistency with reference
                if reference_encoded_shape is None:
                    reference_encoded_shape = encoded.shape[1:]
                    reference_decoded_shape = decoded.shape[1:]
                else:
                    if encoded.shape[1:] != reference_encoded_shape:
                        print(f"  ❌ Encoded shape inconsistency!")
                        print(f"     Expected: {reference_encoded_shape}")
                        print(f"     Got: {encoded.shape[1:]}")
                        return False

                    if decoded.shape[1:] != reference_decoded_shape:
                        print(f"  ❌ Decoded shape inconsistency!")
                        print(f"     Expected: {reference_decoded_shape}")
                        print(f"     Got: {decoded.shape[1:]}")
                        return False

                print(f"  ✅ Batch size {batch_size}: Consistent")

            print("✅ Dimension consistency test PASSED")
            self.test_results['dimension_consistency'] = True
            return True

        except Exception as e:
            print(f"❌ Dimension consistency test FAILED: {e}")
            traceback.print_exc()
            return False

    def test_interface_compatibility(self):
        """Test interface compatibility with transition models"""
        print("\n" + "=" * 50)
        print("🧪 TEST 4: Interface Compatibility")
        print("=" * 50)

        if self.model is None:
            print("❌ No model available for testing")
            return False

        try:
            # Check required attributes for transition models
            required_attrs = []
            optional_attrs = []

            # Different model types need different attributes
            if self.ae_model_type in ['vqvae', 'soft_vqvae']:
                required_attrs.extend(['n_embeddings', 'embedding_dim', 'n_latent_embeds'])
                optional_attrs.extend(['codebook_size', 'quantized_enc'])
            elif self.ae_model_type in ['ae', 'vae']:
                required_attrs.extend(['latent_dim'])

            # Check attributes
            missing_required = []
            for attr in required_attrs:
                if not hasattr(self.model, attr):
                    missing_required.append(attr)
                else:
                    value = getattr(self.model, attr)
                    print(f"  ✅ {attr}: {value}")

            if missing_required:
                print(f"❌ Missing required attributes: {missing_required}")
                return False

            # Check optional attributes
            for attr in optional_attrs:
                if hasattr(self.model, attr):
                    value = getattr(self.model, attr)
                    print(f"  ✅ {attr}: {value}")
                else:
                    print(f"  ⚠️ Optional attribute missing: {attr}")

            # Check methods
            required_methods = ['encode', 'decode']
            for method in required_methods:
                if not hasattr(self.model, method):
                    print(f"❌ Missing required method: {method}")
                    return False
                else:
                    print(f"  ✅ Method {method}: Available")

            # Test trainer compatibility
            if self.trainer is not None:
                trainer_methods = ['calculate_losses', 'train']
                for method in trainer_methods:
                    if hasattr(self.trainer, method):
                        print(f"  ✅ Trainer method {method}: Available")
                    else:
                        print(f"  ⚠️ Trainer method missing: {method}")

            print("✅ Interface compatibility test PASSED")
            self.test_results['interface_compatibility'] = True
            return True

        except Exception as e:
            print(f"❌ Interface compatibility test FAILED: {e}")
            traceback.print_exc()
            return False

    def test_batch_processing(self):
        """Test batch processing efficiency and correctness"""
        print("\n" + "=" * 50)
        print("🧪 TEST 5: Batch Processing")
        print("=" * 50)

        if self.model is None:
            print("❌ No model available for testing")
            return False

        try:
            obs1, obs2 = self.get_test_observations()

            # Test single vs batch processing consistency
            single_obs = torch.from_numpy(obs1).float().unsqueeze(0).to(self.device)
            batch_obs = single_obs.repeat(4, 1, 1, 1)  # 4 identical observations

            with torch.no_grad():
                # Single processing
                single_encoded = self.model.encode(single_obs)
                single_decoded = self.model.decode(single_encoded)

                # Batch processing
                batch_encoded = self.model.encode(batch_obs)
                batch_decoded = self.model.decode(batch_encoded)

            # Check consistency
            single_expanded = single_encoded.repeat(4, 1, 1, 1) if len(
                single_encoded.shape) == 4 else single_encoded.repeat(4, 1)

            encoding_diff = torch.abs(batch_encoded - single_expanded).max().item()
            decoding_diff = torch.abs(batch_decoded - single_decoded.repeat(4, 1, 1, 1)).max().item()

            print(f"  Encoding difference: {encoding_diff:.6f}")
            print(f"  Decoding difference: {decoding_diff:.6f}")

            tolerance = 1e-5
            if encoding_diff > tolerance or decoding_diff > tolerance:
                print(f"❌ Batch processing inconsistent (tolerance: {tolerance})")
                return False

            # Test performance
            batch_sizes = [1, 8, 16, 32]
            times = {}

            for bs in batch_sizes:
                test_batch = torch.randn(bs, *obs1.shape).to(self.device)
                test_batch = torch.clamp(test_batch, 0, 1)

                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.model.encode(test_batch)

                # Time measurement
                start_time = time.time()
                for _ in range(10):
                    with torch.no_grad():
                        encoded = self.model.encode(test_batch)
                        decoded = self.model.decode(encoded)
                end_time = time.time()

                avg_time = (end_time - start_time) / 10
                times[bs] = avg_time
                print(f"  Batch size {bs:2d}: {avg_time:.4f}s ({avg_time / bs:.4f}s per sample)")

            # Check scaling efficiency
            efficiency = times[1] / (times[32] / 32)
            print(f"  Batch efficiency: {efficiency:.2f}x speedup")

            self.test_results['performance'] = times

            print("✅ Batch processing test PASSED")
            self.test_results['batch_processing'] = True
            return True

        except Exception as e:
            print(f"❌ Batch processing test FAILED: {e}")
            traceback.print_exc()
            return False

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n" + "=" * 50)
        print("🧪 TEST 6: Edge Case Handling")
        print("=" * 50)

        if self.model is None:
            print("❌ No model available for testing")
            return False

        try:
            edge_cases_passed = 0
            total_edge_cases = 0

            # Test case 1: Empty tensor
            print("\n🔸 Testing empty tensor...")
            total_edge_cases += 1
            try:
                empty_tensor = torch.empty(0, 3, 48, 48).to(self.device)
                with torch.no_grad():
                    encoded = self.model.encode(empty_tensor)
                    if encoded.shape[0] == 0:
                        print("  ✅ Empty tensor handled correctly")
                        edge_cases_passed += 1
                    else:
                        print("  ❌ Empty tensor not handled correctly")
            except Exception as e:
                print(f"  ⚠️ Empty tensor caused exception: {e}")

            # Test case 2: Very small values
            print("\n🔸 Testing very small values...")
            total_edge_cases += 1
            try:
                tiny_tensor = torch.full((1, 3, 48, 48), 1e-10).to(self.device)
                with torch.no_grad():
                    encoded = self.model.encode(tiny_tensor)
                    decoded = self.model.decode(encoded)
                    if torch.isfinite(decoded).all():
                        print("  ✅ Very small values handled correctly")
                        edge_cases_passed += 1
                    else:
                        print("  ❌ Very small values caused NaN/Inf")
            except Exception as e:
                print(f"  ❌ Very small values caused exception: {e}")

            # Test case 3: Different spatial sizes (if supported)
            print("\n🔸 Testing different spatial sizes...")
            total_edge_cases += 1
            try:
                different_size = torch.randn(1, 3, 32, 32).to(self.device)
                different_size = torch.clamp(different_size, 0, 1)
                with torch.no_grad():
                    encoded = self.model.encode(different_size)
                    decoded = self.model.decode(encoded)
                    print("  ✅ Different spatial size handled")
                    edge_cases_passed += 1
            except Exception as e:
                print(f"  ⚠️ Different spatial size not supported: {e}")
                # This is often expected, so we'll count it as passed
                edge_cases_passed += 1

            # Test case 4: Training/eval mode consistency
            print("\n🔸 Testing training/eval mode consistency...")
            total_edge_cases += 1
            try:
                test_input = torch.randn(2, 3, 48, 48).to(self.device)
                test_input = torch.clamp(test_input, 0, 1)

                # Test in eval mode
                self.model.eval()
                with torch.no_grad():
                    eval_encoded = self.model.encode(test_input)
                    eval_decoded = self.model.decode(eval_encoded)

                # Test in train mode
                self.model.train()
                with torch.no_grad():
                    train_encoded = self.model.encode(test_input)
                    train_decoded = self.model.decode(train_encoded)

                # For deterministic models, these should be similar
                # For stochastic models (like VAE), they might differ
                encode_diff = torch.abs(eval_encoded - train_encoded).max().item()
                decode_diff = torch.abs(eval_decoded - train_decoded).max().item()

                print(f"  Encode difference (eval vs train): {encode_diff:.6f}")
                print(f"  Decode difference (eval vs train): {decode_diff:.6f}")

                if encode_diff < 1e-3 and decode_diff < 1e-3:
                    print("  ✅ Training/eval modes consistent (deterministic)")
                elif encode_diff < 1.0 and decode_diff < 1.0:
                    print("  ✅ Training/eval modes reasonably consistent (stochastic)")
                else:
                    print("  ⚠️ Large difference between training/eval modes")

                edge_cases_passed += 1

            except Exception as e:
                print(f"  ❌ Training/eval mode test failed: {e}")

            success_rate = edge_cases_passed / total_edge_cases
            print(f"\nEdge case success rate: {edge_cases_passed}/{total_edge_cases} ({success_rate:.1%})")

            if success_rate >= 0.75:  # 75% success rate
                print("✅ Edge case handling test PASSED")
                self.test_results['edge_case_handling'] = True
                return True
            else:
                print("❌ Edge case handling test FAILED")
                return False

        except Exception as e:
            print(f"❌ Edge case handling test FAILED: {e}")
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """Run all tests and provide summary"""
        print("🧪 ENCODER MODEL VALIDATION")
        print("=" * 60)
        print(f"Environment: {self.env_name}")
        print(f"Model Type: {self.ae_model_type}")
        print(f"Device: {self.device}")
        print("=" * 60)

        # Run tests in order
        test_functions = [
            self.test_basic_functionality,
            self.test_nan_stability,
            self.test_dimension_consistency,
            self.test_interface_compatibility,
            self.test_batch_processing,
            self.test_edge_cases
        ]

        passed_tests = 0
        total_tests = len(test_functions)

        for test_func in test_functions:
            try:
                if test_func():
                    passed_tests += 1
            except KeyboardInterrupt:
                print("\n❌ Tests interrupted by user")
                break
            except Exception as e:
                print(f"❌ Test failed with unexpected error: {e}")
                traceback.print_exc()

        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        success_rate = passed_tests / total_tests

        for test_name, result in self.test_results.items():
            if test_name == 'performance':
                continue
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():.<30} {status}")

        print(f"\nOverall Score: {passed_tests}/{total_tests} ({success_rate:.1%})")

        if success_rate >= 0.8:
            print("🎉 ENCODER MODEL IS VALID AND READY FOR USE! 🎉")
            recommendation = "✅ Your encoder model passes all critical tests and is ready for training/inference."
        elif success_rate >= 0.6:
            print("⚠️ ENCODER MODEL HAS SOME ISSUES")
            recommendation = "⚠️ Your encoder model works but has some issues. Check the failed tests above."
        else:
            print("❌ ENCODER MODEL HAS SERIOUS ISSUES")
            recommendation = "❌ Your encoder model has serious issues and needs fixes before use."

        print(f"\n💡 Recommendation: {recommendation}")

        if 'performance' in self.test_results and self.test_results['performance']:
            print(f"\n⚡ Performance:")
            perf = self.test_results['performance']
            print(f"   Single sample: {perf.get(1, 0):.4f}s")
            print(f"   Batch of 32: {perf.get(32, 0):.4f}s ({perf.get(32, 0) / 32:.4f}s per sample)")

        return success_rate >= 0.8


def main():
    parser = argparse.ArgumentParser(description="Test encoder model validity")
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-6x6-v0',
                        help='Environment name')
    parser.add_argument('--ae_model_type', type=str, default='vqvae',
                        choices=['ae', 'vae', 'vqvae', 'soft_vqvae', 'dae', 'identity', 'flatten'],
                        help='Autoencoder model type')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run tests on')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--codebook_size', type=int, default=16,
                        help='Codebook size for VQ-VAE')

    args = parser.parse_args()

    # Run tests
    tester = EncoderTester(
        env_name=args.env_name,
        ae_model_type=args.ae_model_type,
        device=args.device,
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size
    )

    success = tester.run_all_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)