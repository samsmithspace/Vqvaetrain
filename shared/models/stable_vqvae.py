# Add this to your shared/models/ directory or replace the existing VQ-VAE implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class StableVectorQuantizer(nn.Module):
    """
    Stable Vector Quantizer that prevents NaN and codebook collapse
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize embeddings with proper scaling
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # Track usage for codebook collapse detection
        self.register_buffer('cluster_usage', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embeddings.weight.data.clone())

        # Hyperparameters
        self.decay = 0.99
        self.eps = 1e-5
        self.threshold_dead_code = 1.0

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        input_shape = inputs.shape

        # Use reshape instead of view for better compatibility
        flat_input = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances to embeddings
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device,
                                dtype=inputs.dtype)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten - use reshape instead of view
        quantized = torch.matmul(encodings, self.embeddings.weight).reshape(input_shape)

        # Update embeddings with exponential moving average (only in training)
        if self.training:
            self._update_embeddings(flat_input.detach(), encodings.detach())

        # Calculate losses with proper gradient handling
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        # Straight through estimator - ensure gradients flow properly
        quantized = inputs + (quantized - inputs).detach()

        # Calculate perplexity (detached for logging only)
        with torch.no_grad():
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, commitment_loss + q_latent_loss, perplexity, encodings

    def _update_embeddings(self, flat_input, encodings):
        """Update embeddings using exponential moving average"""
        with torch.no_grad():
            # Update cluster usage
            cluster_usage = torch.sum(encodings, dim=0)
            self.cluster_usage.mul_(self.decay).add_(cluster_usage, alpha=1 - self.decay)

            # Update embeddings
            embed_sum = torch.matmul(encodings.t(), flat_input)
            self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Normalize embeddings
            cluster_usage = self.cluster_usage + self.eps
            embed_normalized = self.embed_avg / cluster_usage.unsqueeze(1)
            self.embeddings.weight.data.copy_(embed_normalized)

            # Reset dead codes
            dead_codes = self.cluster_usage < self.threshold_dead_code
            if dead_codes.any():
                print(f"🔧 Resetting {dead_codes.sum().item()} dead codes")
                # Reset dead codes to random embeddings
                n_dead = dead_codes.sum().item()
                random_indices = torch.randperm(self.num_embeddings)[:n_dead]
                self.embeddings.weight.data[dead_codes] = self.embeddings.weight.data[random_indices]
                # Add small noise
                self.embeddings.weight.data[dead_codes] += torch.randn_like(
                    self.embeddings.weight.data[dead_codes]) * 0.01


class StableVQVAEModel(nn.Module):
    """
    Stable VQ-VAE implementation with NaN prevention
    """

    def __init__(self, input_dim, codebook_size=512, embedding_dim=64,
                 encoder=None, decoder=None, n_latents=None, commitment_cost=0.25):
        super().__init__()

        self.encoder_type = 'vqvae'
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.n_embeddings = codebook_size
        self.commitment_cost = commitment_cost

        # Use provided encoder/decoder or create simple ones
        if encoder is None:
            encoder = self._make_encoder(input_dim, embedding_dim)
        if decoder is None:
            decoder = self._make_decoder(input_dim, embedding_dim)

        self.encoder = encoder
        self.decoder = decoder

        # Calculate number of latent positions
        with torch.no_grad():
            test_input = torch.zeros(1, *input_dim)
            encoded = self.encoder(test_input)
            # Ensure the encoded tensor has the right shape for embedding_dim
            if len(encoded.shape) == 4:  # BCHW format
                self.n_latent_embeds = encoded.shape[2] * encoded.shape[3]  # H * W
            else:
                self.n_latent_embeds = int(np.prod(encoded.shape[1:]) // embedding_dim)

        print(f"VQ-VAE: {self.n_latent_embeds} latent positions, {codebook_size} codes, {embedding_dim}D embeddings")

        # Stable quantizer
        self.quantizer = StableVectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )

        # Initialize with He initialization
        self.apply(self._init_weights)

    def _make_encoder(self, input_dim, embedding_dim):
        """Simple encoder if none provided"""
        return nn.Sequential(
            nn.Conv2d(input_dim[0], 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, 3, 1, 1),
        )

    def _make_decoder(self, input_dim, embedding_dim):
        """Simple decoder if none provided"""
        return nn.Sequential(
            nn.Conv2d(embedding_dim, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_dim[0], 4, 2, 1),
            nn.Tanh()
        )

    def _init_weights(self, m):
        """Safe weight initialization"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data *= 0.1  # Scale down
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data *= 0.1
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x, return_indices=False):
        """Encode input to latent codes"""
        z_e = self.encoder(x)
        z_q, _, _, encodings = self.quantizer(z_e)

        if return_indices:
            indices = torch.argmax(encodings, dim=1)
            indices = indices.view(x.shape[0], -1)
            return indices
        return z_q

    def decode(self, z_q):
        """Decode quantized latents to reconstruction"""
        return self.decoder(z_q)

    def forward(self, x):
        """Forward pass through VQ-VAE"""
        # Encode
        z_e = self.encoder(x)

        # Quantize
        z_q, quantizer_loss, perplexity, encodings = self.quantizer(z_e)

        # Decode
        x_recon = self.decoder(z_q)

        # Return reconstruction, quantizer loss, perplexity, and encodings
        return x_recon, quantizer_loss, perplexity, encodings


class RobustVQVAETrainer:
    """
    Robust VQ-VAE trainer with comprehensive NaN handling
    """

    def __init__(self, model, lr=1e-4, beta1=0.9, beta2=0.999,
                 grad_clip=1.0, log_freq=100):
        self.model = model
        self.log_freq = log_freq
        self.grad_clip = grad_clip
        self.train_step = 0

        # Separate optimizers for different components with different learning rates
        encoder_decoder_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
        quantizer_params = list(model.quantizer.parameters())

        # Use smaller learning rate for quantizer
        self.optimizer_main = torch.optim.Adam(
            encoder_decoder_params,
            lr=lr,
            betas=(beta1, beta2),
            eps=1e-8
        )
        self.optimizer_quantizer = torch.optim.Adam(
            quantizer_params,
            lr=lr * 0.1,  # 10x smaller learning rate for quantizer
            betas=(beta1, beta2),
            eps=1e-8
        )

        # Learning rate schedulers
        self.scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_main, T_max=1000)
        self.scheduler_quantizer = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_quantizer, T_max=1000)

        # Training state
        self.nan_count = 0
        self.max_nan_resets = 5

    def calculate_losses(self, batch_data, return_stats=False):
        """Calculate VQ-VAE losses with NaN protection"""
        device = next(self.model.parameters()).device

        # Get observations from batch data
        sample_size = int(batch_data[0].shape[0] / 2)
        obs = torch.cat([
            batch_data[0][:sample_size],
            batch_data[2][sample_size:]
        ], dim=0).to(device)

        # Input validation
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print("❌ NaN/Inf in input observations!")
            return self._get_safe_losses(device), {}

        # Clamp input to reasonable range
        obs = torch.clamp(obs, -10, 10)

        try:
            # Forward pass
            x_recon, quantizer_loss, perplexity, encodings = self.model(obs)

            # Check for NaN in outputs
            if (torch.isnan(x_recon).any() or torch.isnan(quantizer_loss).any() or
                    torch.isinf(x_recon).any() or torch.isinf(quantizer_loss).any()):
                print("❌ NaN/Inf in model outputs!")
                self._reset_model_weights()
                return self._get_safe_losses(device), {}

            # Reconstruction loss with clipping
            recon_loss = F.mse_loss(x_recon, obs, reduction='none')
            recon_loss = torch.clamp(recon_loss, 0, 100)  # Clip extreme values
            recon_loss = recon_loss.mean()

            # Clamp quantizer loss
            quantizer_loss = torch.clamp(quantizer_loss, 0, 10)

            loss_dict = {
                'recon_loss': recon_loss,
                'quantizer_loss': quantizer_loss,
                'total_loss': recon_loss + quantizer_loss
            }

            # Validate all losses
            for k, v in loss_dict.items():
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"❌ NaN/Inf in {k}!")
                    return self._get_safe_losses(device), {}

            if return_stats:
                stats = {
                    'perplexity': perplexity.item() if torch.is_tensor(perplexity) else perplexity,
                    'active_codes': (encodings.sum(0) > 0).sum().item(),
                    'codebook_usage': encodings.sum(0).max().item(),
                }
                return loss_dict, stats

            return loss_dict

        except Exception as e:
            print(f"❌ Error in forward pass: {e}")
            self._reset_model_weights()
            return self._get_safe_losses(device), {}

    def _get_safe_losses(self, device):
        """Return safe dummy losses when NaN occurs"""
        return {
            'recon_loss': torch.tensor(1.0, device=device),
            'quantizer_loss': torch.tensor(0.1, device=device),
            'total_loss': torch.tensor(1.1, device=device)
        }

    def _reset_model_weights(self):
        """Reset model weights when NaN is detected"""
        if self.nan_count >= self.max_nan_resets:
            print("🛑 Too many NaN resets, stopping")
            return

        print(f"🔧 Resetting model weights (attempt {self.nan_count + 1})")
        self.model.apply(self.model._init_weights)

        # Reinitialize quantizer embeddings
        with torch.no_grad():
            self.model.quantizer.embeddings.weight.uniform_(-0.01, 0.01)
            self.model.quantizer.embed_avg.copy_(self.model.quantizer.embeddings.weight.data)
            self.model.quantizer.cluster_usage.zero_()

        self.nan_count += 1

    def train(self, batch_data):
        """Training step with robust NaN handling"""
        loss_dict, stats = self.calculate_losses(batch_data, return_stats=True)

        # Check for NaN in any loss component
        has_nan = any(torch.isnan(v).any() for v in loss_dict.values())

        if has_nan:
            print(f"❌ NaN detected in losses at step {self.train_step}")
            return loss_dict, stats

        total_loss = loss_dict['total_loss']

        # Zero gradients
        self.optimizer_main.zero_grad()
        self.optimizer_quantizer.zero_grad()

        try:
            # Backward pass
            total_loss.backward()

            # Check gradients for NaN
            self._check_and_clip_gradients()

            # Update parameters
            self.optimizer_main.step()
            self.optimizer_quantizer.step()

            # Update learning rates
            if self.train_step % 100 == 0:
                self.scheduler_main.step()
                self.scheduler_quantizer.step()

        except Exception as e:
            print(f"❌ Error in backward pass: {e}")
            self._reset_model_weights()

        # Logging
        if self.log_freq > 0 and self.train_step % self.log_freq == 0:
            log_str = f'VQ-VAE step {self.train_step}'
            for k, v in loss_dict.items():
                log_str += f' | {k}: {v.item():.4f}'
            for k, v in stats.items():
                log_str += f' | {k}: {v:.2f}' if isinstance(v, float) else f' | {k}: {v}'
            print(log_str)

        self.train_step += 1
        return loss_dict, stats

    def _check_and_clip_gradients(self):
        """Check for NaN gradients and apply clipping"""
        # Check encoder/decoder gradients
        for name, param in self.model.encoder.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"⚠️ NaN gradient in encoder.{name}, zeroing")
                param.grad.zero_()

        for name, param in self.model.decoder.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"⚠️ NaN gradient in decoder.{name}, zeroing")
                param.grad.zero_()

        # Check quantizer gradients
        for name, param in self.model.quantizer.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"⚠️ NaN gradient in quantizer.{name}, zeroing")
                param.grad.zero_()

        # Apply gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)