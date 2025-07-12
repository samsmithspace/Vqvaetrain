#!/usr/bin/env python3
"""
Debug version to find why model predictions are black
"""

import sys
import os
import argparse
from argparse import Namespace
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Add the parent directory to path to import discrete_mbrl modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_helpers import make_env, check_env_name
from model_construction import construct_ae_model, construct_trans_model


def create_simple_args(env_name, ae_model_type, latent_dim, device='cpu', **kwargs):
    """Create a simple args object without complex processing."""
    args = Namespace()

    # Basic settings
    args.env_name = check_env_name(env_name)
    args.ae_model_type = ae_model_type
    args.trans_model_type = kwargs.get('trans_model_type', 'continuous')
    args.device = device
    args.load = True

    # Model architecture - USE EXACT SAME DEFAULTS AS TRAINING
    args.latent_dim = latent_dim
    args.embedding_dim = kwargs.get('embedding_dim', 64)
    args.filter_size = kwargs.get('filter_size', 8)
    args.codebook_size = kwargs.get('codebook_size', 16)
    args.ae_model_version = kwargs.get('ae_model_version', '2')
    args.trans_model_version = kwargs.get('trans_model_version', '1')
    args.trans_hidden = kwargs.get('trans_hidden', 256)
    args.trans_depth = kwargs.get('trans_depth', 3)
    args.stochastic = kwargs.get('stochastic', 'simple')

    # Hash-relevant parameters that must match training
    args.extra_info = kwargs.get('extra_info', None)
    args.repr_sparsity = kwargs.get('repr_sparsity', 0)
    args.sparsity_type = kwargs.get('sparsity_type', 'random')
    args.vq_trans_1d_conv = kwargs.get('vq_trans_1d_conv', False)

    # FTA parameters
    args.fta_tiles = kwargs.get('fta_tiles', 20)
    args.fta_bound_low = kwargs.get('fta_bound_low', -2)
    args.fta_bound_high = kwargs.get('fta_bound_high', 2)
    args.fta_eta = kwargs.get('fta_eta', 0.2)

    # Training parameters
    args.learning_rate = kwargs.get('learning_rate', 3e-4)
    args.trans_learning_rate = kwargs.get('trans_learning_rate', 3e-4)
    args.ae_grad_clip = kwargs.get('ae_grad_clip', 0)
    args.log_freq = kwargs.get('log_freq', 100)

    # Additional parameters
    args.vq_trans_loss_type = kwargs.get('vq_trans_loss_type', 'mse')
    args.vq_trans_state_snap = kwargs.get('vq_trans_state_snap', False)
    args.use_soft_embeds = kwargs.get('use_soft_embeds', False)
    args.e2e_loss = kwargs.get('e2e_loss', False)
    args.log_norms = kwargs.get('log_norms', False)

    # Disable logging
    args.wandb = False
    args.comet_ml = False

    return args


class DebugModelGUI:
    def __init__(self, env_name, ae_model_type, latent_dim, device='cpu', **kwargs):
        print(f"🔍 DEBUG: Initializing GUI with: {env_name}, {ae_model_type}, latent_dim={latent_dim}")

        self.args = create_simple_args(env_name, ae_model_type, latent_dim, device, **kwargs)
        self.setup_models()
        self.setup_environments()
        self.setup_gui()

        # State tracking
        self.real_obs = None
        self.model_obs = None
        self.model_state = None
        self.step_count = 0

        # Initialize environments
        self.reset_environments()

    def debug_tensor(self, tensor, name):
        """Debug utility to print tensor information."""
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
        else:
            tensor_np = tensor

        print(f"🔍 {name}:")
        print(f"  Shape: {tensor_np.shape}")
        print(f"  Dtype: {tensor_np.dtype}")
        print(f"  Min: {tensor_np.min():.4f}")
        print(f"  Max: {tensor_np.max():.4f}")
        print(f"  Mean: {tensor_np.mean():.4f}")
        print(f"  Has NaN: {np.isnan(tensor_np).any()}")
        print(f"  Has Inf: {np.isinf(tensor_np).any()}")
        if tensor_np.size < 20:  # Only print values for small tensors
            print(f"  Values: {tensor_np.flatten()}")
        print()

    def setup_models(self):
        """Load the trained encoder and transition models."""
        print("\n🔍 DEBUG: Setting up models...")

        try:
            # Create a dummy observation to get the shape
            temp_env = make_env(self.args.env_name)
            temp_obs = temp_env.reset()
            if isinstance(temp_obs, tuple):
                temp_obs = temp_obs[0]
            temp_env.close()

            print(f"🔍 Original observation shape: {temp_obs.shape}")
            self.debug_tensor(temp_obs, "Original observation")

            # Load encoder model
            self.encoder_model = construct_ae_model(temp_obs.shape, self.args)[0]
            self.encoder_model = self.encoder_model.to(self.args.device)
            self.encoder_model.eval()

            # Test encoder with dummy observation
            print("🔍 Testing encoder with dummy observation...")
            with torch.no_grad():
                dummy_obs = torch.from_numpy(temp_obs).float().unsqueeze(0).to(self.args.device)
                self.debug_tensor(dummy_obs, "Encoder input")

                encoded = self.encoder_model.encode(dummy_obs)
                self.debug_tensor(encoded, "Encoded latent")

                decoded = self.encoder_model.decode(encoded)
                self.debug_tensor(decoded, "Decoded observation")

            # Load transition model
            temp_env = make_env(self.args.env_name)
            self.trans_model = construct_trans_model(
                self.encoder_model, self.args, temp_env.action_space)[0]
            self.trans_model = self.trans_model.to(self.args.device)
            self.trans_model.eval()
            temp_env.close()

            # Test transition model
            print("🔍 Testing transition model...")
            with torch.no_grad():
                # Prepare latent state for transition model
                if self.args.trans_model_type == 'continuous':
                    if hasattr(self.encoder_model, 'latent_dim'):
                        latent_for_trans = encoded.reshape(encoded.shape[0], self.encoder_model.latent_dim)
                    else:
                        latent_for_trans = encoded.reshape(encoded.shape[0], -1)
                else:
                    latent_for_trans = encoded

                self.debug_tensor(latent_for_trans, "Latent for transition model")

                # Test transition prediction
                dummy_action = torch.tensor([0], dtype=torch.long).to(self.args.device)  # Forward action
                trans_output = self.trans_model(latent_for_trans, dummy_action)

                if isinstance(trans_output, tuple):
                    next_latent = trans_output[0]
                else:
                    next_latent = trans_output

                self.debug_tensor(next_latent, "Predicted next latent")

                # Decode predicted latent
                decoded_next = self.encoder_model.decode(next_latent)
                self.debug_tensor(decoded_next, "Decoded predicted observation")

            print("✅ Model setup completed successfully!")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            raise

    def setup_environments(self):
        """Setup real environment."""
        self.real_env = make_env(self.args.env_name)
        print(f"✅ Environment created: {self.args.env_name}")

    def setup_gui(self):
        """Create the GUI interface."""
        self.root = tk.Tk()
        self.root.title(f"DEBUG: MiniGrid Model ({self.args.env_name})")
        self.root.geometry("800x600")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Title
        title = ttk.Label(main_frame, text="DEBUG: MiniGrid Model Prediction Analysis",
                          font=("Arial", 14, "bold"))
        title.pack(pady=(0, 10))

        # Images frame
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(pady=10)

        # Real environment
        real_frame = ttk.LabelFrame(images_frame, text="Real Environment", padding="5")
        real_frame.pack(side='left', padx=(0, 10))

        self.real_canvas = tk.Canvas(real_frame, width=200, height=200, bg="white")
        self.real_canvas.pack()

        # Model prediction
        model_frame = ttk.LabelFrame(images_frame, text="Model Prediction", padding="5")
        model_frame.pack(side='right', padx=(10, 0))

        self.model_canvas = tk.Canvas(model_frame, width=200, height=200, bg="white")
        self.model_canvas.pack()

        # Debug info frame
        debug_frame = ttk.LabelFrame(main_frame, text="Debug Info", padding="10")
        debug_frame.pack(pady=10, fill='x')

        self.debug_text = tk.Text(debug_frame, height=8, width=80)
        scrollbar = ttk.Scrollbar(debug_frame, orient="vertical", command=self.debug_text.yview)
        self.debug_text.configure(yscrollcommand=scrollbar.set)

        self.debug_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(pady=10)

        self.step_label = ttk.Label(info_frame, text="Steps: 0", font=("Arial", 12))
        self.step_label.pack()

        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.pack(pady=10, fill='x')

        controls_text = (
            "Arrow Keys: ↑=Forward, ←=Turn Left, →=Turn Right, ↓=Stay\n"
            "R: Reset  |  Q: Quit  |  T: Test Model"
        )
        ttk.Label(controls_frame, text=controls_text, justify='center').pack()

        # Action mapping for MiniGrid
        self.action_map = {
            'Up': 2,  # Move forward
            'Left': 0,  # Turn left
            'Right': 1,  # Turn right
            'Down': 6  # Done/Stay
        }

        # Bind keyboard events
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.focus_set()

    def log_debug(self, message):
        """Add debug message to the debug text widget."""
        self.debug_text.insert(tk.END, message + "\n")
        self.debug_text.see(tk.END)
        print(message)  # Also print to console

    def preprocess_obs(self, obs):
        """Preprocess observation for model input with debugging."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)  # Add batch dimension
        return obs

    def obs_to_display_image(self, obs, size=(200, 200), name=""):
        """Convert observation to displayable PIL Image with debugging."""
        try:
            if isinstance(obs, torch.Tensor):
                obs_np = obs.detach().cpu().numpy()
            else:
                obs_np = obs.copy()

            self.log_debug(f"🖼️ Converting {name} to image:")
            self.log_debug(f"  Input shape: {obs_np.shape}")
            self.log_debug(f"  Input range: [{obs_np.min():.3f}, {obs_np.max():.3f}]")

            # Handle different observation formats
            if len(obs_np.shape) == 4:  # Batch dimension
                obs_np = obs_np[0]
            if len(obs_np.shape) == 3 and obs_np.shape[0] <= 3:  # CHW format
                obs_np = obs_np.transpose(1, 2, 0)  # Convert to HWC

            self.log_debug(f"  After reshape: {obs_np.shape}")

            # Ensure values are in [0, 1] range
            if obs_np.max() > 1.0:
                obs_np = obs_np / 1
                self.log_debug(f"  Normalized by 1")

            obs_np = np.clip(obs_np, 0, 1)
            self.log_debug(f"  After clipping: [{obs_np.min():.3f}, {obs_np.max():.3f}]")

            # Convert to PIL Image
            if len(obs_np.shape) == 2:  # Grayscale
                obs_np = np.stack([obs_np] * 3, axis=-1)  # Convert to RGB
                self.log_debug(f"  Converted grayscale to RGB")

            # Check for all-zero or all-same values
            if np.allclose(obs_np, 0):
                self.log_debug(f"  ⚠️ WARNING: All pixels are zero!")
            elif np.allclose(obs_np, obs_np.flat[0]):
                self.log_debug(f"  ⚠️ WARNING: All pixels have same value: {obs_np.flat[0]:.3f}")

            img_array = (obs_np * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img = img.resize(size, Image.NEAREST)

            self.log_debug(f"  ✅ Image created successfully")
            return img

        except Exception as e:
            self.log_debug(f"  ❌ Error creating image: {e}")
            # Return a red error image
            error_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            error_img[:, :, 0] = 255  # Red
            return Image.fromarray(error_img)

    def reset_environments(self):
        """Reset both real and model environments."""
        self.log_debug("\n🔄 RESETTING ENVIRONMENTS")

        # Reset real environment
        reset_result = self.real_env.reset()
        if isinstance(reset_result, tuple):
            self.real_obs, _ = reset_result
        else:
            self.real_obs = reset_result

        self.debug_tensor(self.real_obs, "Reset real observation")

        # Reset model state to match real environment
        self.sync_model_with_real()

        self.step_count = 0
        self.update_display()
        self.log_debug("✅ Reset completed!")

    def sync_model_with_real(self):
        """Sync model state with real environment state."""
        self.log_debug("\n🔗 SYNCING MODEL WITH REAL STATE")

        try:
            with torch.no_grad():
                obs_tensor = self.preprocess_obs(self.real_obs).to(self.args.device)
                self.debug_tensor(obs_tensor, "Preprocessed observation")

                # Encode real observation to get model state
                self.model_state = self.encoder_model.encode(obs_tensor)
                self.debug_tensor(self.model_state, "Encoded model state")

                # Reshape for transition model if needed
                if self.args.trans_model_type == 'continuous':
                    if hasattr(self.encoder_model, 'latent_dim'):
                        self.model_state = self.model_state.reshape(self.model_state.shape[0],
                                                                    self.encoder_model.latent_dim)
                    else:
                        self.model_state = self.model_state.reshape(self.model_state.shape[0], -1)
                    self.debug_tensor(self.model_state, "Reshaped model state for transition model")

                # Decode to get model observation
                model_obs_tensor = self.encoder_model.decode(self.model_state)
                self.debug_tensor(model_obs_tensor, "Decoded model observation")

                self.model_obs = model_obs_tensor.cpu().numpy()[0]
                self.debug_tensor(self.model_obs, "Final model observation (numpy)")

                self.log_debug("✅ Model sync completed!")

        except Exception as e:
            self.log_debug(f"❌ Error syncing model: {e}")
            import traceback
            traceback.print_exc()

    def step_real_environment(self, action):
        """Take a step in the real environment."""
        step_result = self.real_env.step(action)

        if len(step_result) == 4:
            self.real_obs, reward, done, info = step_result
        else:
            self.real_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        return reward, done, info

    def step_model_prediction(self, action):
        """Get model's prediction for the next state."""
        self.log_debug(f"\n🎯 PREDICTING NEXT STATE (action={action})")

        if self.model_state is None:
            self.log_debug("❌ No model state available!")
            return

        try:
            with torch.no_grad():
                action_tensor = torch.tensor([action], dtype=torch.long).to(self.args.device)
                self.debug_tensor(action_tensor, "Action tensor")

                # Predict next state using transition model
                trans_output = self.trans_model(self.model_state, action_tensor)

                if isinstance(trans_output, tuple):
                    next_state_pred = trans_output[0]
                    if len(trans_output) > 1:
                        reward_pred = trans_output[1]
                        self.debug_tensor(reward_pred, "Predicted reward")
                else:
                    next_state_pred = trans_output

                self.debug_tensor(next_state_pred, "Predicted next state")

                # Decode predicted state to observation
                next_obs_pred = self.encoder_model.decode(next_state_pred)
                self.debug_tensor(next_obs_pred, "Decoded predicted observation")

                # Update model state and observation
                self.model_state = next_state_pred
                self.model_obs = next_obs_pred.cpu().numpy()[0]

                self.log_debug("✅ Model prediction completed!")

        except Exception as e:
            self.log_debug(f"❌ Error in model prediction: {e}")
            import traceback
            traceback.print_exc()

    def test_model_pipeline(self):
        """Test the entire model pipeline with current state."""
        self.log_debug("\n🧪 TESTING MODEL PIPELINE")

        if self.real_obs is None:
            self.log_debug("❌ No real observation available!")
            return

        try:
            # Test encoding -> decoding
            with torch.no_grad():
                obs_tensor = self.preprocess_obs(self.real_obs).to(self.args.device)
                encoded = self.encoder_model.encode(obs_tensor)
                decoded = self.encoder_model.decode(encoded)

                self.debug_tensor(obs_tensor, "Original observation")
                self.debug_tensor(encoded, "Encoded")
                self.debug_tensor(decoded, "Decoded")

                # Check reconstruction quality
                mse = torch.mean((obs_tensor - decoded) ** 2)
                self.log_debug(f"Reconstruction MSE: {mse.item():.6f}")

                # Test transition model
                if hasattr(self.encoder_model, 'latent_dim') and self.args.trans_model_type == 'continuous':
                    encoded_for_trans = encoded.reshape(encoded.shape[0], self.encoder_model.latent_dim)
                else:
                    encoded_for_trans = encoded

                action_tensor = torch.tensor([2], dtype=torch.long).to(self.args.device)  # Forward
                trans_output = self.trans_model(encoded_for_trans, action_tensor)

                if isinstance(trans_output, tuple):
                    next_encoded = trans_output[0]
                else:
                    next_encoded = trans_output

                next_decoded = self.encoder_model.decode(next_encoded)

                self.debug_tensor(next_encoded, "Transition output")
                self.debug_tensor(next_decoded, "Transition decoded")

                # Check if transition changed anything
                change = torch.mean(torch.abs(decoded - next_decoded))
                self.log_debug(f"Transition change magnitude: {change.item():.6f}")

                self.log_debug("✅ Pipeline test completed!")

        except Exception as e:
            self.log_debug(f"❌ Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()

    def update_display(self):
        """Update the GUI display with current observations."""
        try:
            # Update real environment image
            if self.real_obs is not None:
                real_img = self.obs_to_display_image(self.real_obs, name="real")
                self.real_photo = ImageTk.PhotoImage(real_img)
                self.real_canvas.delete("all")
                self.real_canvas.create_image(100, 100, image=self.real_photo)

            # Update model prediction image
            if self.model_obs is not None:
                model_img = self.obs_to_display_image(self.model_obs, name="model")
                self.model_photo = ImageTk.PhotoImage(model_img)
                self.model_canvas.delete("all")
                self.model_canvas.create_image(100, 100, image=self.model_photo)

            # Update step counter
            self.step_label.config(text=f"Steps: {self.step_count}")

        except Exception as e:
            self.log_debug(f"❌ Error updating display: {e}")

    def on_key_press(self, event):
        """Handle keyboard input."""
        key = event.keysym

        if key == 'q' or key == 'Q':
            self.quit_app()
        elif key == 'r' or key == 'R':
            self.reset_environments()
        elif key == 't' or key == 'T':
            self.test_model_pipeline()
        elif key in self.action_map:
            self.take_action(self.action_map[key])

    def take_action(self, action):
        """Execute an action in both environments."""
        self.log_debug(f"\n🎮 TAKING ACTION: {action}")

        try:
            # Step real environment
            reward, done, info = self.step_real_environment(action)
            self.debug_tensor(self.real_obs, "New real observation")

            # Step model prediction
            self.step_model_prediction(action)

            self.step_count += 1
            self.log_debug(f"Step {self.step_count} completed, Reward: {reward:.2f}")

            # Update display
            self.update_display()

            # Check if episode is done
            if done:
                self.log_debug(f"🏁 Episode finished! Steps taken: {self.step_count}")
                self.root.after(3000, self.reset_environments)

        except Exception as e:
            self.log_debug(f"❌ Error taking action: {e}")
            import traceback
            traceback.print_exc()

    def quit_app(self):
        """Clean up and quit the application."""
        try:
            self.real_env.close()
        except:
            pass
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the GUI application."""
        print("🎮 Starting DEBUG GUI...")
        print("Controls: Arrow keys, R=reset, Q=quit, T=test model")

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nQuitting...")
        finally:
            self.quit_app()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Debug MiniGrid Model GUI")
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-6x6-v0')
    parser.add_argument('--ae_model_type', type=str, default='ae')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    print(f"🚀 Launching DEBUG GUI:")
    print(f"  Environment: {args.env_name}")
    print(f"  Model: {args.ae_model_type}")
    print(f"  Latent Dim: {args.latent_dim}")
    print(f"  Device: {args.device}")

    try:
        app = DebugModelGUI(args.env_name, args.ae_model_type, args.latent_dim, args.device)
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()