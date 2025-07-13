import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.models.stable_vqvae import StableVQVAEModel, RobustVQVAETrainer
from training_helpers import get_args


def test_stable_vqvae():
    """Test the stable VQ-VAE implementation"""
    print("🧪 Testing Stable VQ-VAE Implementation")

    # Set up test parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = (3, 48, 48)
    batch_size = 512  # Use your actual training batch size

    # Create model
    model = StableVQVAEModel(
        input_dim=input_dim,
        codebook_size=64,
        embedding_dim=64,
        commitment_cost=0.25
    ).to(device)

    # Create trainer
    trainer = RobustVQVAETrainer(
        model,
        lr=1e-4,
        grad_clip=1.0,
        log_freq=10
    )

    print(f"✅ Model created successfully")
    print(f"   Input dim: {input_dim}")
    print(f"   Codebook size: {model.n_embeddings}")
    print(f"   Embedding dim: {model.embedding_dim}")
    print(f"   Latent positions: {model.n_latent_embeds}")

    # Test forward pass with large batch
    print(f"\n🔍 Testing forward pass with batch size {batch_size}")

    test_input = torch.randn(batch_size, *input_dim).to(device)

    try:
        with torch.no_grad():
            model.eval()
            recon, q_loss, perplexity, encodings = model(test_input)

            print(f"✅ Forward pass successful!")
            print(f"   Input shape: {test_input.shape}")
            print(f"   Reconstruction shape: {recon.shape}")
            print(f"   Quantizer loss: {q_loss.item():.6f}")
            print(f"   Perplexity: {perplexity.item():.2f}")
            print(f"   Has NaN: {torch.isnan(recon).any().item()}")

            if torch.isnan(recon).any():
                print("❌ Model still produces NaN!")
                return False

        # Test training step
        print(f"\n🏋️ Testing training step")
        model.train()

        # Create dummy batch data
        batch_data = [
            test_input[:256],  # obs
            torch.randint(0, 4, (256,)).to(device),  # actions
            test_input[256:],  # next_obs
            torch.randn(256).to(device),  # rewards
            torch.zeros(256).to(device)  # dones
        ]

        for step in range(5):
            loss_dict, stats = trainer.train(batch_data)

            print(f"   Step {step}: Loss = {loss_dict['total_loss'].item():.4f}, "
                  f"Perplexity = {stats.get('perplexity', 0):.2f}, "
                  f"Active codes = {stats.get('active_codes', 0)}")

            # Check for NaN
            if any(torch.isnan(v).any() for v in loss_dict.values()):
                print(f"❌ NaN detected at step {step}")
                return False

        print("✅ Training test successful!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_stable_vqvae()
    if success:
        print("\n🎉 Stable VQ-VAE test passed! Ready for training.")
    else:
        print("\n💥 Stable VQ-VAE test failed!")