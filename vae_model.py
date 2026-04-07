import math
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class Encoder(nn.Module):
    """
    Encoder module for VAE.

    img -> net -> (mu, log_var)
    mu - mean of the latent distribution
    log_var - log variance of the latent distribution
    Args:
        latent_dim: Dimension of the latent space
        out_channels: Number of channels in the output image (e.g., 3 for RGB)
        max_channels: Maximum number of channels in the convolutional layers
        min_channels: Minimum number of channels in the convolutional layers
        img_size: Size of the input image (assumed to be square)
        bottleneck_spatial: Spatial size of the bottleneck feature map (e.g., 4 for 4x4)
    """

    def __init__(self, latent_dim: int = 128,
                input_channels: int = 3,
                max_channels: int = 512, 
                min_channels: int = 16,
                img_size: int = 256,
                bottleneck_spatial: int = 4):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.bottleneck_spatial = bottleneck_spatial
        self.img_size = img_size
        self.input_channels = input_channels

        # Calculate the number of downsampling layers needed to reach the bottleneck spatial size (eg 256 -> 4 requires 6 downsamples)
        num_downsamples = int(math.log2(img_size / bottleneck_spatial))
        channels = [] # List to store the number of channels for each convolutional layer
        ch = min_channels
        for _ in range(num_downsamples):
            channels.append(ch)
            ch = min(ch * 2, max_channels)
        self.channels = channels

        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels

        for h_dim in channels:
            conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2)
            ))
            in_channels = h_dim

        self.neural_net = nn.Sequential(*conv_layers)

        self.pool = nn.AdaptiveAvgPool2d((bottleneck_spatial, bottleneck_spatial)) #??

        self.flatten_size = channels[-1] * bottleneck_spatial * bottleneck_spatial

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x: Input image tensor of shape [batch_size, input_channels, img_size, img_size]

        Returns:
            mu: Mean of the latent distribution [batch_size, latent_dim]
            log_var: Log variance of the latent distribution [batch_size, latent_dim]
        """
        h: torch.Tensor = self.neural_net(x)  # Pass through convolutional layers
        h = self.pool(h)  # Adaptive average pooling to get fixed spatial size

        h = torch.flatten(h, start_dim=1)  # Flatten the feature map

        mu = self.fc_mu(h)  # Get mean of latent distribution
        log_var = self.fc_log_var(h)  # Get log variance of latent distribution

        return mu, log_var
    
    def print_architecture(self):
        """
        Print the architecture of the encoder.
        """
        
        print("\nEncoder:")
        spatial = self.img_size
        in_channels = self.input_channels
        print(f"Input: {in_channels} x {spatial} x {spatial}")
        for i, ch in enumerate(self.channels):
            spatial = spatial // 2
            print(f"conv{i + 1}: {in_channels} -> {ch} x {spatial} x {spatial}, activations: {ch*spatial*spatial}")
            in_channels = ch
        print(f"  avg_pool:   [{self.channels[-1]}, {self.bottleneck_spatial}, {self.bottleneck_spatial}]")
        print(f"  flatten:    [{self.flatten_size:,}]")
        print(f"  fc_mu:      [{self.flatten_size:,}] -> [{self.latent_dim}]")
        print(f"  fc_logvar:  [{self.flatten_size:,}] -> [{self.latent_dim}]")

class Decoder(nn.Module):
    """
    latent -> net -> recon_img
    """
    def __init__(self, latent_dim: int = 128,
                output_channels: int = 3,
                max_channels: int = 512, 
                min_channels: int = 16,
                img_size: int = 256,
                bottleneck_spatial: int = 4):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.bottleneck_spatial = bottleneck_spatial
        self.output_channels = output_channels

        num_upsamples = int(math.log2(img_size / bottleneck_spatial))

        channels = [] # Number of channels for each convolutional layer
        ch = min_channels
        for _ in range(num_upsamples):
            channels.append(ch)
            ch = min(ch * 2, max_channels)
        channels.reverse() # Reverse to go from bottleneck to output
        self.channels = channels

        # Fully connected layer to expand the latent vector to the size of the bottleneck feature map
        self.fc = nn.Linear(latent_dim, channels[0] * bottleneck_spatial * bottleneck_spatial)

        conv_layers = []
        for i in range(num_upsamples - 1):
            conv_layers.append(nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i + 1],kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.LeakyReLU(0.2)
            ))
        self.neural_net = nn.Sequential(*conv_layers)

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(channels[-1], output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            z: Latent vector tensor of shape [batch_size, latent_dim]

        Returns:
            recon_x: Reconstructed image tensor [batch_size, output_channels, img_size, img_size]
        """

        h = self.fc(z)
        h = h.view(-1, self.channels[0], self.bottleneck_spatial, self.bottleneck_spatial)  # Reshape to feature map

        h = self.neural_net(h)
        recon_x = self.output_layer(h)

        return recon_x
    
    def print_architecture(self):
        """
        Print the architecture of the decoder.
        """

        print("\nDecoder Architecture:")
        print(f"  fc: [{self.latent_dim}] -> [{self.channels[0] * self.bottleneck_spatial * self.bottleneck_spatial:,}]")
        spatial = self.bottleneck_spatial

        for i in range(len(self.channels) - 1):
            spatial = spatial * 2
            print(f" up{i+1}: {self.channels[i]} -> {self.channels[i + 1]} x {spatial} x {spatial}, activations: {self.channels[i + 1]*spatial*spatial}")
        spatial = spatial * 2
        print(f" output: {self.channels[-1]} -> {self.output_channels} x {spatial} x {spatial}")

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.
    """

    def __init__(self,
                 input_channels: int = 3,
                 latent_dim: int = 128,
                 img_size: int = 256,
                 max_channels: int = 512,
                 min_channels: int = 16,
                 bottleneck_spatial: int = 4):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.input_channels = input_channels
        self.bottleneck_spatial = bottleneck_spatial
        self.max_channels = max_channels
        self.min_channels = min_channels

        self.encoder = Encoder(latent_dim=latent_dim, input_channels=input_channels,
                               max_channels=max_channels, min_channels=min_channels,
                               img_size=img_size, bottleneck_spatial=bottleneck_spatial)
        
        self.decoder = Decoder(latent_dim=latent_dim, output_channels=input_channels,
                               max_channels=max_channels, min_channels=min_channels,
                               img_size=img_size, bottleneck_spatial=bottleneck_spatial)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution.
        z = mu + exp(0.5 * logvar) * eps
        where eps ~ N(0, I)

        Args:
            mu: Mean of the latent distribution [batch_size, latent_dim]
            log_var: Log variance of the latent distribution [batch_size, latent_dim]
        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """

        eps = torch.randn_like(log_var)  # Some random noise to sample from the distribution

        std = torch.exp(0.5 * log_var)  # Standard deviation from log variance
        z = mu + std * eps  # Reparameterization trick
        return z
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.

        Args:
            x: Input image tensor of shape [batch_size, input_channels, img_size, img_size]

        Returns:
            recon_x: Reconstructed image tensor [batch_size, input_channels, img_size, img_size]
            mu: Mean of the latent distribution [batch_size, latent_dim]
            log_var: Log variance of the latent distribution [batch_size, latent_dim]
        """

        mu, log_var = self.encoder(x)  # Get mean and log variance from encoder
        z = self.reparameterize(mu, log_var)  # Sample from the latent distribution
        recon_x = self.decoder(z)  # Reconstruct the image from the latent vector

        return recon_x, mu, log_var
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an input image into the latent space.

        Args:
            x: Input image tensor of shape [batch_size, input_channels, img_size, img_size]

        Returns:
            mu: Mean of the latent distribution [batch_size, latent_dim]
            log_var: Log variance of the latent distribution [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector into an image.

        Args:
            z: Latent vector tensor of shape [batch_size, latent_dim]

        Returns:
            recon_x: Reconstructed image tensor [batch_size, input_channels, img_size, img_size]
        """
        return self.decoder(z)
    
    def print_architecture(self):
        """
        Print the architecture of the model.
        """

        enc = self.encoder
        dec = self.decoder
        bs = enc.bottleneck_spatial

        print("\n" + "=" * 60)
        print("Model Architecture")

        # Encoder architecture
        enc.print_architecture()

        # Bottleneck
        print(f"\nBottleneck:")
        print(f"  latent_dim: {self.latent_dim}")
        
        # Decoder architecture
        dec.print_architecture()

        # Summary
        print("\nSummary:")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Encoder channels: {enc.channels}")
        print(f"  Decoder channels: {dec.channels}")
        print(f"  Bottleneck spatial: {bs}x{bs}")
        print(f"  Flatten size: {enc.flatten_size:,}")
        print(f"  Total parameters: {total_params:,}")
        print("=" * 60 + "\n")
    
def pixel_reconstruction_loss(x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
    """
    Compute the pixel-wise reconstruction loss (MSE) between the reconstructed image and the original image.

    Args:
        x: Original image tensor [batch_size, input_channels, img_size, img_size]
        x_recon: Reconstructed image tensor [batch_size, input_channels, img_size, img_size]

    Returns:
        loss: Pixel-wise reconstruction loss (MSE)
    """
    per_pixel_loss = F.mse_loss(x_recon, x, reduction='none')
    per_sample_loss = per_pixel_loss.reshape(per_pixel_loss.size(0), -1).sum(dim=1)  # Average over all pixels for each sample
    loss = per_sample_loss.mean()  # Average over the batch
    return loss


def kl_divergence_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Compute the KL divergence loss between the latent distribution and the standard normal distribution.

    KL(N(mu, sigma) || N(0, I)) = 0.5 * sum(sigma^2 + mu^2 - log(sigma^2) - 1)

    Args:
        mu: Mean of the latent distribution [batch_size, latent_dim]
        log_var: Log variance of the latent distribution [batch_size, latent_dim]

    Returns:
        kl_loss: KL divergence loss
    """
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)  # Sum over latent dimensions
    kl_loss = kl_loss.mean()  # Average over the batch
    return kl_loss

def vae_loss(x: torch.Tensor, x_recon: torch.Tensor, 
             mu: torch.Tensor, log_var: torch.Tensor,
             beta: float = 1.0,
             recon_weight: float = 1.0,
             ssim_weight: float = 0.0
             ) -> Dict[str, torch.Tensor]:
    """Compute the total VAE loss, which is a combination of the pixel reconstruction loss and the KL divergence loss.

    Args:
        x: Original image tensor [batch_size, input_channels, img_size, img_size]
        x_recon: Reconstructed image tensor [batch_size, input_channels, img_size, img_size]
        mu: Mean of the latent distribution [batch_size, latent_dim]
        log_var: Log variance of the latent distribution [batch_size, latent_dim]
        beta: Weight for the KL divergence loss (default=1.0)
        recon_weight: Weight for the pixel reconstruction loss (default=1.0)
        ssim_weight: Weight for the SSIM loss (default=0.0)
    Returns:
        loss_dict: Dictionary containing the total loss and individual components
    """
    recon_loss = pixel_reconstruction_loss(x, x_recon)
    
    kl_loss = kl_divergence_loss(mu, log_var)
    #ssim_loss = 0.0
    #sinkhorn_loss = 0.0

    total_loss = (recon_weight * recon_loss) + (beta * kl_loss)

    loss_dict = {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss
    }

    return loss_dict

if __name__ == "__main__":
    # Example usage
    vae = VAE(input_channels=3, latent_dim=128, img_size=256)
    vae.print_architecture()

    # Test forward pass with dummy data
    dummy_input = torch.randn(4, 3, 256, 256)  # Batch of 4 images
    recon_x, mu, log_var = vae(dummy_input)
    print(f"Reconstructed image shape: {recon_x.shape}")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent log variance shape: {log_var.shape}")

    # Test loss computation
    loss_dict = vae_loss(dummy_input, recon_x, mu, log_var)
    print(f"Loss dictionary: {loss_dict}")


