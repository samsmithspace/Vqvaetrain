#!/usr/bin/env python3
"""
Simplified MiniGrid Model Debug Tool - WORKING VERSION
"""

import sys
import os
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_helpers import make_env
from model_construction import construct_ae_model, construct_trans_model
from training_helpers import make_argparser, process_args
from data_logging import init_experiment


class ModelDebugGUI:
    def __init__(self, args):
        self.args = args
        self.real_obs = None
        self.model_obs = None
        self.model_state = None
        self.step_count = 0

        print(f"Loading models for {args.env_name} ({args.ae_model_type} + {args.trans_model_type})")

        self.setup_models()
        self.setup_environment()
        self.setup_gui()
        self.reset_environment()

    def setup_models(self):
        """Load encoder and transition models."""
        # Get observation shape
        temp_env = make_env(self.args.env_name)
        obs = temp_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        temp_env.close()

        # Load models
        self.encoder = construct_ae_model(obs.shape, self.args)[0].to(self.args.device).eval()
        self.transition = construct_trans_model(self.encoder, self.args,
                                                make_env(self.args.env_name).action_space)[0].to(self.args.device).eval()

        print(f"✓ Models loaded: {type(self.encoder).__name__} + {type(self.transition).__name__}")

    def setup_environment(self):
        """Setup real environment."""
        self.env = make_env(self.args.env_name)

    def setup_gui(self):
        """Create GUI interface."""
        self.root = tk.Tk()
        self.root.title(f"Model Debug: {self.args.env_name}")
        self.root.geometry("600x500")

        # Main frame
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill='both', expand=True)

        # Title
        ttk.Label(main, text=f"Model Debug: {self.args.ae_model_type} + {self.args.trans_model_type}",
                  font=("Arial", 12, "bold")).pack(pady=(0, 10))

        # Images
        img_frame = ttk.Frame(main)
        img_frame.pack(pady=10)

        real_frame = ttk.LabelFrame(img_frame, text="Real", padding="5")
        real_frame.pack(side='left', padx=5)
        self.real_canvas = tk.Canvas(real_frame, width=150, height=150, bg="white")
        self.real_canvas.pack()

        pred_frame = ttk.LabelFrame(img_frame, text="Predicted", padding="5")
        pred_frame.pack(side='right', padx=5)
        self.pred_canvas = tk.Canvas(pred_frame, width=150, height=150, bg="white")
        self.pred_canvas.pack()

        # Info
        info_frame = ttk.Frame(main)
        info_frame.pack(pady=10)
        self.step_label = ttk.Label(info_frame, text="Steps: 0")
        self.step_label.pack()

        # Controls
        controls = ttk.LabelFrame(main, text="Controls", padding="10")
        controls.pack(pady=10, fill='x')
        ttk.Label(controls, text="↑=Forward, ←=Left, →=Right, R=Reset, Q=Quit").pack()

        # Key bindings
        self.action_map = {'Up': 2, 'Left': 0, 'Right': 1}
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.focus_set()

    def preprocess_obs(self, obs):
        """Convert observation to tensor."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs

    def obs_to_image(self, obs, size=(150, 150)):
        """Convert observation to PIL image."""
        try:
            if isinstance(obs, torch.Tensor):
                obs = obs.detach().cpu().numpy()

            if len(obs.shape) == 4:
                obs = obs[0]
            if len(obs.shape) == 3 and obs.shape[0] <= 3:
                obs = obs.transpose(1, 2, 0)

            obs = np.clip(obs, 0, 1)

            if len(obs.shape) == 2:
                obs = np.stack([obs] * 3, axis=-1)

            img_array = (obs * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            return img.resize(size, Image.NEAREST)

        except Exception as e:
            print(f"Image conversion error: {e}")
            # Return red error image
            error_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            error_img[:, :, 0] = 255
            return Image.fromarray(error_img)

    def convert_state_for_transition(self, encoded_state):
        """Convert encoded state for transition model input."""
        if self.args.trans_model_type in ['discrete', 'shared_vq', 'universal_vq']:
            # For discrete models: indices -> embeddings -> flatten
            if hasattr(self.encoder, 'quantizer') and hasattr(self.encoder.quantizer, 'embedding'):
                embeddings = self.encoder.quantizer.embedding(encoded_state.long())
                return embeddings.reshape(embeddings.shape[0], -1)
            else:
                # Fallback: just flatten the indices
                return encoded_state.reshape(encoded_state.shape[0], -1)

        # For continuous models: just reshape/flatten
        return encoded_state.reshape(encoded_state.shape[0], -1)

    def convert_state_for_decoding(self, trans_output):
        """Convert transition output back for decoding."""
        if self.args.trans_model_type in ['discrete', 'shared_vq', 'universal_vq']:
            # For discrete models: embeddings -> indices
            if hasattr(self.encoder, 'quantizer') and len(trans_output.shape) > 2:
                batch_size = trans_output.shape[0]
                embed_dim = self.encoder.quantizer.embedding.weight.shape[1]
                n_embeds = trans_output.shape[1] // embed_dim
                reshaped = trans_output.reshape(batch_size, n_embeds, embed_dim)

                if hasattr(self.encoder.quantizer, 'forward'):
                    _, indices, _ = self.encoder.quantizer(reshaped)
                    return indices
                return reshaped
        return trans_output

    def reset_environment(self):
        """Reset environment and sync model state."""
        print("Resetting environment...")

        # Reset real environment
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            self.real_obs, _ = reset_result
        else:
            self.real_obs = reset_result

        # Sync model state
        with torch.no_grad():
            obs_tensor = self.preprocess_obs(self.real_obs).to(self.args.device)
            self.model_state = self.encoder.encode(obs_tensor)
            model_obs_tensor = self.encoder.decode(self.model_state)
            self.model_obs = model_obs_tensor.cpu().numpy()[0]

        self.step_count = 0
        self.update_display()
        print("✓ Reset complete")

    def step_environment(self, action):
        """Take a step in both real and model environments."""
        try:
            # Step real environment
            step_result = self.env.step(action)
            if len(step_result) == 4:
                self.real_obs, reward, done, info = step_result
            else:
                self.real_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            # Step model prediction
            with torch.no_grad():
                # Convert current state for transition model
                current_state = self.convert_state_for_transition(self.model_state)

                # Create action tensor
                action_tensor = torch.tensor([action], dtype=torch.long).to(self.args.device)

                # Predict next state
                trans_output = self.transition(current_state, action_tensor)
                if isinstance(trans_output, tuple):
                    next_state = trans_output[0]
                else:
                    next_state = trans_output

                # Convert back for decoding and storage
                next_state_for_decode = self.convert_state_for_decoding(next_state)

                # Update model state and observation
                if self.args.trans_model_type in ['discrete', 'shared_vq', 'universal_vq']:
                    self.model_state = next_state_for_decode
                else:
                    self.model_state = next_state

                pred_obs_tensor = self.encoder.decode(next_state_for_decode)
                self.model_obs = pred_obs_tensor.cpu().numpy()[0]

            self.step_count += 1
            self.update_display()

            if done:
                print(f"Episode done after {self.step_count} steps")
                self.root.after(2000, self.reset_environment)

        except Exception as e:
            print(f"Step error: {e}")
            # Continue with real environment only
            self.step_count += 1
            self.update_display()

    def update_display(self):
        """Update GUI images."""
        try:
            # Update real image
            if self.real_obs is not None:
                real_img = self.obs_to_image(self.real_obs)
                self.real_photo = ImageTk.PhotoImage(real_img)
                self.real_canvas.delete("all")
                self.real_canvas.create_image(75, 75, image=self.real_photo)

            # Update predicted image
            if self.model_obs is not None:
                pred_img = self.obs_to_image(self.model_obs)
                self.pred_photo = ImageTk.PhotoImage(pred_img)
                self.pred_canvas.delete("all")
                self.pred_canvas.create_image(75, 75, image=self.pred_photo)

            # Update step counter
            self.step_label.config(text=f"Steps: {self.step_count}")

        except Exception as e:
            print(f"Display update error: {e}")

    def on_key_press(self, event):
        """Handle keyboard input."""
        key = event.keysym

        if key.lower() == 'q':
            self.quit()
        elif key.lower() == 'r':
            self.reset_environment()
        elif key in self.action_map:
            self.step_environment(self.action_map[key])

    def quit(self):
        """Clean up and quit."""
        try:
            self.env.close()
        except:
            pass
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the GUI."""
        print("Starting GUI... Use arrow keys to control, R to reset, Q to quit")
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit()


def main():
    # Parse arguments using the same system as other scripts
    parser = make_argparser()
    args = parser.parse_args()
    args = process_args(args)

    # Disable logging
    args.wandb = False
    args.comet_ml = False
    args = init_experiment('debug', args)

    print(f"Environment: {args.env_name}")
    print(f"Models: {args.ae_model_type} + {args.trans_model_type}")
    print(f"Device: {args.device}")

    try:
        app = ModelDebugGUI(args)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()