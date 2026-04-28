"""
Script to compare the sizes of classifier and VAE models.
"""
import torch
import torch.nn as nn
from vae_model import VAE, original_BVAE, SimpleVAE
from cnn_model import CNN, CNN_1D, MLP


def count_parameters(model):
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def format_params(num_params):
    """Format parameter count as a readable string."""
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return f"{num_params}"


# Classifier Models
print("\n" + "=" * 70)
print("CLASSIFIER MODELS")
print("=" * 70)

# CNN with default settings
cnn = CNN(in_channels=3, input_size=64, num_classes=5, pooling=True)
cnn_params = count_parameters(cnn)
print(f"\nCNN (input_size=64, pooling=True)")
print(f"  Parameters: {format_params(cnn_params)} ({cnn_params:,})")

# CNN_1D with default settings
cnn_1d = CNN_1D(in_channels=1, input_size=128, num_classes=5, pooling=True)
cnn_1d_params = count_parameters(cnn_1d)
print(f"\nCNN_1D (input_size=128, pooling=True)")
print(f"  Parameters: {format_params(cnn_1d_params)} ({cnn_1d_params:,})")

# MLP with default settings
mlp = MLP(input_size=128, num_classes=5)
mlp_params = count_parameters(mlp)
print(f"\nMLP (input_size=128)")
print(f"  Parameters: {format_params(mlp_params)} ({mlp_params:,})")

# VAE Models
print("\n" + "=" * 70)
print("VAE MODELS")
print("=" * 70)

# VAE with default settings (64x64 images, latent_dim=128)
vae = VAE(input_channels=3, latent_dim=128, img_size=64, 
          max_channels=128, min_channels=16, bottleneck_spatial=4)
vae_params = count_parameters(vae)
print(f"\nVAE (img_size=64x64, latent_dim=128)")
print(f"  Parameters: {format_params(vae_params)} ({vae_params:,})")

# original_BVAE with default settings (64x64 images, latent_dim=128)
original_bvae = original_BVAE(input_channels=3, latent_dim=128, img_size=64)
original_bvae_params = count_parameters(original_bvae)
print(f"\noriginal_BVAE (img_size=64x64, latent_dim=128)")
print(f"  Parameters: {format_params(original_bvae_params)} ({original_bvae_params:,})")

# SimpleVAE with default settings (64x64 images, latent_dim=128)
simple_vae = SimpleVAE(input_channels=3, latent_dim=64)
simple_vae_params = count_parameters(simple_vae)
print(f"\nSimpleVAE (img_size=64x64, latent_dim=128)")
print(f"  Parameters: {format_params(simple_vae_params)} ({simple_vae_params:,})")


