import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from ade20k_dataset import get_dataloaders, ADE20KDataset
import torchvision.transforms as transforms

def main():
    n = 16

    img_size = 64
    # Load dataset without augmentations and with augmentations
    base_dataset = ADE20KDataset(root_dir="ade20k_data/ADEData2016", split='training', sub_split='train', transform=transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()]))
    
    pad_size = max(8, img_size // 6)
    pad_ratio = pad_size / img_size
    aug_transform = transforms.Compose([
        transforms.Pad(padding=pad_size, padding_mode="reflect"),
        transforms.RandomRotation(degrees=5),
        transforms.Resize((img_size+pad_size, img_size+pad_size)),
        transforms.CenterCrop(size=img_size),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAdjustSharpness(sharpness_factor=5),
        transforms.ToTensor(),
    ])
    aug_dataset = ADE20KDataset(root_dir="ade20k_data/ADEData2016", split='training', sub_split='train', transform=aug_transform)
    
    # Plot n images
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        # Same index to see how one image changes
        orig_img, _ = base_dataset[i]
        aug_img, _ = aug_dataset[i]
        
        # Convert tensors to numpy (C, H, W) -> (H, W, C)
        orig_img = np.transpose(orig_img.numpy(), (1, 2, 0))
        aug_img = np.transpose(aug_img.numpy(), (1, 2, 0))
        
        axes[i][0].imshow(orig_img)
        axes[i][0].axis('off')
        axes[i][0].set_title("Original")
        
        axes[i][1].imshow(aug_img)
        axes[i][1].axis('off')
        axes[i][1].set_title("Augmented")
        
    plt.tight_layout()
    plt.savefig("augmentations_comparison.png", bbox_inches='tight')
    print(f"Saved visualization to augmentations_comparison.png")
        

if __name__ == "__main__":
    main()
