import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from pathlib import Path

from cnn_model import CNN
from train_cnn import CNNTrainer
from vae_model import VAE
from train_vae import VaeTrainer

from ade20k_dataset import ADE20KDataset, get_dataloaders

def main():
    

    train_loader, val_loader = get_dataloaders(batch_size=32, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer1 = CNNTrainer(
        model=CNN(num_classes=150),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=Path("./cnn_study_models"),
        model_save_name="cnn_study_experiment_1"
    )

    trainer2 = CNNTrainer(
        model=CNN(num_classes=150),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=Path("./cnn_study_models"),
        model_save_name="cnn_study_experiment_2"
    )

    vae_trainer = VaeTrainer(
        model=VAE(latent_dim=128),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=Path("./vae_study_models"),
        model_save_name="vae_study_experiment_1"
    )

    trainer1.load_model("./cnn_study_models/cnn_study_experiment_1/cnn_model.pth")
    trainer2.load_model("./cnn_study_models/cnn_study_experiment_2/cnn_model.pth")
    #vae_trainer.load_model("./vae_study_models/vae_study_experiment_1/vae_model.pth")

    # Evaluate both models on the validation set
    val_metrics1 = trainer1.evaluate()
    val_metrics2 = trainer2.evaluate()
    print(f"Experiment 1 - Validation Loss: {val_metrics1['loss']:.4f}, Validation Accuracy: {val_metrics1['accuracy']:.4f}")
    print(f"Experiment 2 - Validation Loss: {val_metrics2['loss']:.4f}, Validation Accuracy: {val_metrics2['accuracy']:.4f}")

if __name__ == "__main__":
    main()