import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from pathlib import Path

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from cnn_model import CNN, CNN_1D
from ade20k_dataset import ADE20KDataset, get_dataloaders


class CNNTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 device,
                 save_dir="./cnn_models",
                 model_save_name="cnn_model.pth"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.writer = SummaryWriter(f'{save_dir}/{model_save_name}')

        self.current_epoch = 0

        self.scaler = GradScaler()

        self.save_dir = Path(save_dir) / model_save_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float('inf')
        self.num_epochs_loss_not_improved = 0
        self.early_stopping_patience = 10 
    def train_epoch(self):
        self.model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        loop = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} Training", dynamic_ncols=True, leave=False)

        for batch_idx, (images, labels) in enumerate(loop):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Check if input is a 1D flat vector (e.g. shape is [batch_size, 256])
            if len(images.shape) == 2 and self.model.__class__.__name__ == "CNN":
                # Calculate the spatial dimension assuming a square feature map
                # e.g., 256 vector -> 1 channel, 16x16
                spatial_size = int(images.shape[1] ** 0.5)

                # Reshape to [batch_size, 1_channel, H, W]
                images = images.view(-1, 1, spatial_size, spatial_size)
                #print(images)
            
            if len(images.shape) == 2 and self.model.__class__.__name__ == "CNN_1D":
                # For CNN_1D [batch_size, 1_channel, sequence_length]
                images = images.unsqueeze(1)
                #print(images.shape)
                # Gaussian noise to prevent memorization
                noise_std = 0.05 
                images = images + torch.randn_like(images) * noise_std 

            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Forward
            with autocast(device_type=self.device.type):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            epoch_accuracy += (predicted == labels).sum().item()


            # Update progress bar
            loop.set_postfix(loss=loss.item(), accuracy=epoch_accuracy / ((batch_idx + 1) * batch_size))

        return {
            'loss': epoch_loss / len(self.train_loader),
            'accuracy': epoch_accuracy / len(self.train_loader.dataset)
        }

    def validate_epoch(self):
        self.model.eval()

        epoch_loss = 0.0
        epoch_accuracy = 0.0

        loop = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} Validation", dynamic_ncols=True, leave=False)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loop):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Check if input is a 1D flat vector (e.g. shape is [batch_size, 256])
                if len(images.shape) == 2 and self.model.__class__.__name__ == "CNN":
                    spatial_size = int(images.shape[1] ** 0.5)
                    images = images.view(-1, 1, spatial_size, spatial_size)
                
                if len(images.shape) == 2 and self.model.__class__.__name__ == "CNN_1D":
                    images = images.unsqueeze(1)

                batch_size = images.size(0)

                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                epoch_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                epoch_accuracy += (predicted == labels).sum().item()

                # Update progress bar
                loop.set_postfix(loss=loss.item(), accuracy=epoch_accuracy / ((batch_idx + 1) * batch_size))

        return {
            'loss': epoch_loss / len(self.val_loader),
            'accuracy': epoch_accuracy / len(self.val_loader.dataset)
        }
    
    def train(self, num_epochs):

        self.num_epochs = self.current_epoch + num_epochs

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch += 1

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            print(f"Epoch {self.current_epoch}/{self.num_epochs} - Train Loss: {train_metrics['loss']:.4f}, Train Accuracy: {train_metrics['accuracy']:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")

            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)

            # Save model checkpoint and Early Stopping tracking
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_model()
                self.num_epochs_loss_not_improved = 0  # Reset counter if improved
            else:
                self.num_epochs_loss_not_improved += 1 # Increment if not improved

            if self.num_epochs_loss_not_improved >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.num_epochs_loss_not_improved} epochs without improvement.")
                break

        self.writer.close()
    
    def save_model(self):
        torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'best_val_loss': self.best_val_loss
            }, self.save_dir / "cnn_model.pth")
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded model from {model_path} at epoch {self.current_epoch} with best validation loss {self.best_val_loss:.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_CNN = True
    train_CNN_1D = False

    root_dir = "ade20k_data/ADEData2016"
    batch_size = 32
    num_workers = 4

    if train_CNN == True:
        #img_size = 16
        img_size = 128
        

        train_loader, val_loader = get_dataloaders(root_dir=root_dir, 
                                                batch_size=batch_size, 
                                                img_size=img_size, 
                                                num_workers=num_workers,
                                                train_augmentation=True,
                                                #latent_dir="experiments/good_v2_top3/latents",
                                                n_common_labels=100,
                                                exclude_concepts=["misc"]
                                                )

        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")

        num_classes = len(train_loader.dataset.unique_classes)
        print(f"Number of classes: {num_classes}")

        #model = CNN(in_channels=1, num_classes=num_classes, input_size=img_size, pooling=False)
        model = CNN(in_channels=3, num_classes=num_classes, input_size=img_size, pooling=True)
        trainer = CNNTrainer(model=model, 
                            train_loader=train_loader, 
                            val_loader=val_loader, 
                            device=device,
                            save_dir="./cnn_models",
                            model_save_name="cnn_experiment_3")
    

    
        #trainer.load_model("./cnn_models/cnn_experiment_1/cnn_model.pth")

        #writer = SummaryWriter('runs/cnn_model_visualization')
        #writer.add_graph(model, torch.randn(1, 3, img_size, img_size).to(device))
        #writer.close()

        trainer.train(num_epochs=100)

    if train_CNN_1D == True:
        vec_size = 256

        train_loader, val_loader = get_dataloaders(root_dir=root_dir, 
                                                batch_size=batch_size,
                                                img_size=vec_size, 
                                                num_workers=num_workers,
                                                train_augmentation=False,
                                                latent_dir="experiments/good_v2_exclude_misc/latents",
                                                n_common_labels=3,
                                                exclude_concepts=["misc"]
                                                )
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")

        num_classes = len(train_loader.dataset.unique_classes)
        print(f"Number of classes: {num_classes}")

        batch = next(iter(train_loader))
        print(f"Batch shape: {batch[0].shape}, Labels shape: {batch[1].shape}")

        model_1D = CNN_1D(num_classes=num_classes, input_size=vec_size)
        trainer_1D = CNNTrainer(model=model_1D,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                device=device,
                                save_dir="./cnn_models",
                                model_save_name="cnn_1d_experiment_1")
        
        trainer_1D.train(num_epochs=100)