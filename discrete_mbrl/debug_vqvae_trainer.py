import torch
import torch.nn.functional as F
from shared.trainers import VQVAETrainer  # Import the original


class DebugVQVAETrainer(VQVAETrainer):
    """Modified VQ-VAE trainer that logs individual loss components"""

    def calculate_losses(self, batch_data):
        """Calculate VQ-VAE losses with detailed logging"""
        obs, _, next_obs = batch_data[:3]

        # Forward pass
        recon_obs, vq_loss, perplexity = self.model(obs)

        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(recon_obs, obs, reduction='mean')

        # 2. VQ Loss (usually contains commitment + codebook losses)
        # This comes from the VQ-VAE model's quantizer
        commitment_loss = vq_loss  # This might be combined, we'll separate if possible

        # 3. Total Loss
        total_loss = recon_loss + commitment_loss

        # Return detailed loss dictionary
        loss_dict = {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity if perplexity is not None else torch.tensor(0.0)
        }

        # Check for NaN values and log them
        for name, value in loss_dict.items():
            if torch.isnan(value):
                print(f"🚨 NaN detected in {name}!")
                print(f"Input stats - min: {obs.min():.4f}, max: {obs.max():.4f}, mean: {obs.mean():.4f}")
                print(
                    f"Recon stats - min: {recon_obs.min():.4f}, max: {recon_obs.max():.4f}, mean: {recon_obs.mean():.4f}")

        return loss_dict

    def train(self, batch_data):
        """Training step with detailed loss logging"""
        self.optimizer.zero_grad()

        loss_dict = self.calculate_losses(batch_data)
        loss = loss_dict['loss']

        # Check for NaN before backward pass
        if torch.isnan(loss):
            print("🚨 NaN loss detected! Skipping backward pass.")
            return loss_dict, {}

        loss.backward()

        # Log gradient norms
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=float('inf'))

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        # Add gradient info to loss dict
        loss_dict['grad_norm'] = grad_norm

        aux_data = {
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm
        }

        return loss_dict, aux_data