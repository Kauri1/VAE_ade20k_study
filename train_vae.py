import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os
import torchinfo

from vae_model import VAE, vae_loss
from ade20k_dataset import ADE20KDataset


class VaeTrainer:
    """
    Trainer class for Variational Autoencoder (VAE) model.

    The training loop:
        for each epoch:
            for each batch in dataloader:
                1. Forward pass: Compute the VAE output and loss.
                2. Reparameterization: z = Reparamaterize(mu, logvar)
                ...
                3. Backpropagation: Compute gradients and update model parameters.
    """

    def __init__(self,
                 model: VAE,
                 train_loader,
                 val_loader,
                 learning_rate: float = 1e-4,
                 device: str = 'cuda',
                 save_dir: str = './experiments',
                 experiment_name: str = 'beta_vae_experiment',
                 use_amp: bool = True,
                 use_channels_last: bool = True,
                 img_size: int = 256,
                 recon_weight: float = 1.0,
                 ssim_weight: float = 0.0,
                 beta: float = 1.0,
                 beta_start: float = 0.0,
                 beta_warmup_epochs: int = 10,
                 label_distance_loss_weight: float = 0.1,
                 n_common_labels: int = None,
                 exclude_concepts: list = None):
        """
        Args:
            model: VAE model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            learning_rate: Learning rate for optimizer
            device: Device to use for training ('cuda' or 'cpu')
            save_dir: Directory to save model checkpoints and logs
            use_amp: Whether to use Automatic Mixed Precision (AMP) for faster training
            use_channels_last: Whether to use channels-last memory format for better performance on GPUs
            img_size: Image size (images will be resized to img_size x img_size)
            recon_weight: Weight for reconstruction loss in total loss calculation
            ssim_weight: Weight for SSIM loss in total loss calculation
            beta_warmup_epochs: Number of epochs over which to linearly increase beta from beta_start to beta
            beta: Final weight for KL divergence term in total loss calculation
            beta_start: Initial weight for KL divergence term at the start of training (will be linearly increased to beta)
            label_distance_loss_weight: Weight for the label distance loss (requires labels in dataset and implementation in model)
        """
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.use_amp = use_amp and self.device_type == 'cuda'
        self.use_channels_last = use_channels_last and self.device_type == 'cuda'

        if self.device_type == 'cuda':
            cudnn.benchmark = True  # Enable cuDNN autotuning for better performance on GPU
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for matrix multiplications on Ampere+ GPUs
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for convolution operations
            torch.set_float32_matmul_precision('high')  # Set float32 matmul precision to high (enables TF32)

        self.model = model.to(self.device)
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.beta = beta
        self.beta_start = beta_start
        self.beta_warmup_epochs = beta_warmup_epochs
        self.recon_weight = recon_weight
        self.ssim_weight = ssim_weight
        self.img_size = img_size
        self.label_loss_weight = label_distance_loss_weight
        self.n_common_labels = n_common_labels
        self.exclude_concepts = exclude_concepts

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, fused=self.device_type == 'cuda')

        self.scaler = GradScaler(enabled=self.use_amp)

        # Setup save directory
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.save_dir / 'logs')

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.num_epochs_loss_not_improved = 0
        self.early_stopping_patience = 10 

        self.save_config(learning_rate)

    def save_config(self, learning_rate):
        config = {
            'latent_dim': self.model.latent_dim,
            'learning_rate': learning_rate,
            'batch_size': self.train_loader.batch_size,
            'device': self.device,
            'recon_weight': self.recon_weight,
            'ssim_weight': self.ssim_weight,
            'label_loss_weight': self.label_loss_weight,
            'beta': self.beta,
            'beta_start': self.beta_start,
            'beta_warmup_epochs': self.beta_warmup_epochs,
            'img_size': self.img_size,
            'max_channels': self.model.max_channels,
            'min_channels': self.model.min_channels,
            'bottleneck_spatial': self.model.bottleneck_spatial,
            'num_workers': self.train_loader.num_workers,
            'n_common_labels': self.n_common_labels,
            'exclude_concepts': self.exclude_concepts
        }

        config_path = self.save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Saved training configuration to {config_path}")

    def get_current_beta(self) -> float:
        if self.current_epoch >= self.beta_warmup_epochs:
            return self.beta
        else:
            # Linearly increase beta from beta_start to beta over beta_warmup_epochs
            return self.beta_start + (self.beta - self.beta_start) * (self.current_epoch / self.beta_warmup_epochs)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the VAE for one epoch.
        
        Returns:
            Dictionary with average losses for epoch: total_loss, recon_loss, kl_loss
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_label_loss = 0.0
        total_samples = 0
        beta = self.get_current_beta()

        loop = tqdm(self.train_loader, 
                    desc=f"Epoch {self.current_epoch}/{self.num_epochs} [Train]", 
                    unit="batch",
                    dynamic_ncols=True)

        for batch_idx, (images, labels) in enumerate(loop):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.use_channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            batch_size = images.size(0)

            # Forward pass
            with autocast(device_type=self.device_type, enabled=self.use_amp):
                recon_images, mu, logvar = self.model(images)

                losses = vae_loss(
                    x = images,
                    x_recon = recon_images,
                    mu = mu,
                    labels=labels,
                    log_var = logvar,
                    beta=beta,
                    recon_weight=self.recon_weight,
                    ssim_weight=self.ssim_weight,
                    lweight=self.label_loss_weight
                )

                loss = losses['total_loss']
                recon_loss = losses['recon_loss']
                kl_loss = losses['kl_loss']
                label_loss = losses['label_loss']

            # Backpropagation and optimization step
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
                self.optimizer.step()
            


            epoch_loss += loss.item() * batch_size  # Multiply by batch_size to get total loss for the epoch
            epoch_recon_loss += recon_loss.item() * batch_size  # Multiply by batch_size to get total recon loss for the epoch
            epoch_kl_loss += kl_loss.item() * batch_size  # Multiply by batch_size to get total KL loss for the epoch
            epoch_label_loss += label_loss.item() * batch_size  # Multiply by batch_size to get total label loss for the epoch
            total_samples += batch_size

            # Update progress bar with current losses and beta value
            loop.set_postfix(loss=loss.item(), recon_loss=recon_loss.item(), kl_loss=kl_loss.item(), label_loss=label_loss.item(), beta=beta)

            # Log training metrics to TensorBoard every 100 steps
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/total_loss', losses['total_loss'].item(), self.global_step)
                self.writer.add_scalar('train/recon_loss', losses['recon_loss'].item(), self.global_step)
                self.writer.add_scalar('train/kl_loss', losses['kl_loss'].item(), self.global_step)
                self.writer.add_scalar('train/label_loss', losses['label_loss'].item(), self.global_step)
                self.writer.add_scalar('train/beta', beta, self.global_step)

            self.global_step += 1

        # Calculate average losses for the epoch
        avg_epoch_loss = epoch_loss / total_samples
        avg_epoch_recon_loss = epoch_recon_loss / total_samples
        avg_epoch_kl_loss = epoch_kl_loss / total_samples
        avg_epoch_label_loss = epoch_label_loss / total_samples

        return {
            'total_loss': avg_epoch_loss,
            'recon_loss': avg_epoch_recon_loss,
            'kl_loss': avg_epoch_kl_loss,
            'label_loss': avg_epoch_label_loss,
            'beta': beta
        }
    
    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """
        Validate the VAE on the validation set.

        Returns:
            Dictionary with average losses for validation: total_loss, recon_loss, kl_loss
        """
        self.model.eval()

        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_label_loss = 0.0
        total_samples = 0
        current_beta = self.get_current_beta()

        loop = tqdm(self.val_loader, 
                    desc=f"Epoch {self.current_epoch}/{self.num_epochs} [Val]", 
                    unit="batch",
                    dynamic_ncols=True)

        
        for batch_idx, (images, labels) in enumerate(loop):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if self.use_channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            batch_size = images.size(0)

            # Forward pass
            with autocast(device_type=self.device_type, enabled=self.use_amp):
                recon_images, mu, logvar = self.model(images)

                losses = vae_loss(
                    x = images,
                    x_recon = recon_images,
                    mu = mu,
                    log_var = logvar,
                    labels=labels,
                    beta=current_beta,
                    recon_weight=self.recon_weight,
                    ssim_weight=self.ssim_weight,
                    lweight=self.label_loss_weight
                )

                loss = losses['total_loss']
                recon_loss = losses['recon_loss']
                kl_loss = losses['kl_loss']
                label_loss = losses['label_loss']

            val_loss += loss.item() * batch_size  # Multiply by batch_size to get total loss for the epoch
            val_recon_loss += recon_loss.item() * batch_size  # Multiply by batch_size to get total recon loss for the epoch
            val_kl_loss += kl_loss.item() * batch_size  # Multiply by batch_size to get total KL loss for the epoch
            val_label_loss += label_loss.item() * batch_size  # Multiply by batch_size to get total label loss for the epoch
            total_samples += batch_size

            # Update progress bar with current losses
            loop.set_postfix(loss=loss.item(), recon_loss=recon_loss.item(), kl_loss=kl_loss.item(), label_loss=label_loss.item())

        # Calculate average losses for the epoch
        avg_val_loss = val_loss / total_samples
        avg_val_recon_loss = val_recon_loss / total_samples
        avg_val_kl_loss = val_kl_loss / total_samples
        avg_val_label_loss = val_label_loss / total_samples

        return {
            'total_loss': avg_val_loss,
            'recon_loss': avg_val_recon_loss,
            'kl_loss': avg_val_kl_loss,
            'label_loss': avg_val_label_loss,
            'beta': current_beta
        }
    
    @torch.no_grad()
    def visualize_reconstructions(self, num_images: int = 8):
        """
        Visualize original and reconstructed images from the validation set.

        Args:
            num_images: Number of images to visualize
        """
        self.model.eval()

        # Get a batch of validation images
        batch = next(iter(self.val_loader))
        images = batch[0][:num_images].to(self.device)
        labels = batch[1][:num_images].to(self.device)

        # Reconstruct images
        recon_images, _, _ = self.model(images)
        
        # Move images to CPU and convert to numpy for visualization
        images = images.cpu().numpy()
        recon_images = recon_images.cpu().numpy()

        # Visualizations
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        for i in range(num_images):
            # Original image
            axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')

            # Reconstructed image
            axes[1, i].imshow(np.transpose(recon_images[i], (1, 2, 0)))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed')
        
        plt.tight_layout()

        save_path = self.save_dir / f'reconstructions_epoch_{self.current_epoch}.png'

        plt.savefig(save_path, dpi=150, bbox_inches='tight')

        self.writer.add_figure('Reconstruction', fig, self.current_epoch)
        plt.close()

        print(f"Saved reconstructions to {save_path}")

    def save_checkpoint(self):
        """
        Save model checkpoint

        Args:
            val_loss: Current validation loss to compare against best validation loss
        """
        checkpoint_path = self.save_dir / f'model_epoch_{self.current_epoch}.pth'
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded model checkpoint from {checkpoint_path} (epoch {self.current_epoch}, best val loss {self.best_val_loss})")
    
    def train(self, num_epochs: int = 20, visualize_every: int = 5):
        """
        Main training loop.

        Args:
            num_epochs: Total number of epochs to train
            visualize_every: Frequency (in epochs) to visualize reconstructions
        """

        print("=" * 50)
        print(f"Starting training for {num_epochs} epochs...")
        print("=" * 50)

        self.num_epochs = num_epochs + self.current_epoch  # Adjust total epochs if resuming from checkpoint

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_losses = self.train_epoch()

            # Validate after each epoch
            val_losses = self.validate()

            print(f"Epoch {epoch}/{self.num_epochs} - Train Loss: {train_losses['total_loss']:.4f}, Val Loss: {val_losses['total_loss']:.4f}, Beta: {train_losses['beta']:.4f}")
            print(f"Train Recon Loss: {train_losses['recon_loss']:.4f}, Train KL Loss: {train_losses['kl_loss']:.4f}, Train Label Loss: {train_losses['label_loss']:.4f}")

            # Log validation metrics to TensorBoard
            self.writer.add_scalar('val/total_loss', val_losses['total_loss'], epoch)
            self.writer.add_scalar('val/recon_loss', val_losses['recon_loss'], epoch)
            self.writer.add_scalar('val/kl_loss', val_losses['kl_loss'], epoch)
            self.writer.add_scalar('val/label_loss', val_losses['label_loss'], epoch)

            # Save checkpoint if validation loss improved
            if val_losses['total_loss'] < self.best_val_loss or self.current_epoch % visualize_every == 0 or self.current_epoch == self.num_epochs - 1:
                self.best_val_loss = val_losses['total_loss']
                print(f"Validation loss improved to {self.best_val_loss:.4f}. Saving checkpoint...")
                self.save_checkpoint()
                self.num_epochs_loss_not_improved = 0  # Reset the counter when loss improves

            else:
                self.num_epochs_loss_not_improved += 1  # Increment the counter when loss doesn't improve

            # Early stopping
            if self.num_epochs_loss_not_improved >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.current_epoch} epochs.")
                break


            # Visualize reconstructions every visualize_every epochs
            if (epoch) % visualize_every == 0:
                self.visualize_reconstructions()
            
        print("Training complete.")
        self.writer.close()
    

if __name__ == "__main__":
    # Example usage
    from torch.utils.data import DataLoader

    # Create datasets and dataloaders
    train_dataset = ADE20KDataset(root_dir='ade20k_data/ADEData2016', split='training', img_size=256)
    val_dataset = ADE20KDataset(root_dir='ade20k_data/ADEData2016', split='validation', img_size=256)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Create VAE model
    model = VAE(latent_dim=128)

    model.print_architecture()

    # Create trainer and start training
    trainer = VaeTrainer(model=model, train_loader=train_loader, val_loader=val_loader, learning_rate=1e-4, device='cuda', beta_start=1, label_distance_loss_weight=0.1)
    trainer.train(num_epochs=20, visualize_every=5)