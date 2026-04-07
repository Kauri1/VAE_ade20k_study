import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Callable, Tuple
from pathlib import Path

class ADE20KDataset(Dataset):
    """
    Custom Dataset for ADE20K semantic segmentation.
    """

    def __init__(self,
                 root_dir: str,
                 split: str = 'training',
                 img_size: int = 256,
                 transform: Optional[Callable] = None):
        """
        Args:
            root_dir (str): Root directory of the ADE20K dataset.
            split (str): Dataset split to use ('training', 'validation').
            img_size (int): Desired image size (images will be resized to img_size x img_size).
            transform (callable, optional): Optional transformation to apply to the images.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), # Converts to [0, 1]
            ])
        else:
            self.transform = transform
        
        # Load image and mask file paths
        self.images_dir = self.root_dir / f'images/{split}'
        if not self.images_dir.exists():
            raise ValueError(f"Directory {self.images_dir} does not exist.")

        self.image_files = self._get_image_files()

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        print(f"Found {len(self.image_files)} images in {self.images_dir}")

        # Load class labels
        self.labels_file = self.root_dir / 'sceneCategories.txt'
        if not self.labels_file.exists():
            print(f"Warning: Labels file {self.labels_file} not found. Class labels will be unavailable.")
        if split == 'training':
            pr = 'ADE_train_'  # Prefix for training images in labels file
        elif split == 'validation':
            pr = 'ADE_val_'  # Prefix for validation images in labels file
        else:
            pr = ''  # No prefix for other splits (if any)
        self.labels = self._get_labels(pr) if self.labels_file.exists() else {}

        print(f"Loaded {len(self.labels)} class labels from {self.labels_file}")
        print(f"Example label: {next(iter(self.labels.items())) if self.labels else 'No labels loaded'}")



    


    def _get_image_files(self):
        """
        Get list of image file paths in the specified split directory.
        """
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(f'*{ext}')))
            image_files.extend(list(self.images_dir.glob(f'*{ext.upper()}')))  # Also check for uppercase extensions

        # Remove duplicates and sort
        unique_files = {p.resolve() for p in image_files}

        return sorted(unique_files)
    
    def _get_labels(self, prefix: str = '') -> dict:
        """
        Load class labels from the sceneCategories.txt file.
        """
        labels = {}
        with open(self.labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    if parts[0].startswith(prefix):
                        picture_id = parts[0]
                        category_name = parts[1]
                        labels[picture_id] = category_name
                        
        return labels

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        get image at index
        """

        img_path = self.image_files[idx]

        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        label = self.get_class_label(idx)

        return image, label
    
    def get_class_label(self, idx):
        """
        Get class label for the image at the specified index.
        """
        img_path = self.image_files[idx]
        picture_id = img_path.stem  # Get filename without extension
        return self.labels.get(picture_id, "Unknown")
    
    def get_all_images_of_label(self, label):
        """
        Get all image paths for a specific class label.
        """
        return [self.image_files[i] for i in range(len(self.image_files)) if self.get_class_label(i) == label]

def get_dataloaders(root_dir: str, 
                batch_size: int = 32,
                img_size: int = 256,
                num_workers: int = 4,
                train_augmentation: bool = False,
                pin_memory: bool = True,
                persistent_workers: bool = True,
                prefetch_factor: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Utility function to create DataLoaders for ADE20K dataset.

    Args:
        root_dir: Root directory containing ADE20K data
        batch_size: Batch size for dataloaders, higher can speed up training but requires more memory
        img_size: Image size (images will be resized to img_size x img_size)
        num_workers: Number of worker processes for data loading, higher can speed up loading
        train_augmentation: If True, apply data augmentation to training set
        pin_memory: If True, pin CPU memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs (requires num_workers > 0)
        prefetch_factor: Number of batches prefetched by each worker
    """

    if train_augmentation:
        # Keep augmented training samples at the requested output size.
        pad_size = max(8, img_size // 6)
        train_transform = transforms.Compose([
            transforms.Pad(padding=pad_size, padding_mode="reflect"),
            transforms.RandomRotation(degrees=20),
            transforms.CenterCrop(size=img_size),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.6, 2.0),
                ratio=(0.8, 1.2)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = ADE20KDataset(root_dir=root_dir, 
                                split='training', 
                                img_size=img_size, 
                                transform=train_transform)

    val_dataset = ADE20KDataset(root_dir=root_dir, 
                                split='validation', 
                                img_size=img_size, 
                                transform=val_transform)
    
    use_persistent_workers = persistent_workers and num_workers > 0
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers, 
                            pin_memory=pin_memory,
                            persistent_workers=use_persistent_workers,
                            prefetch_factor=prefetch_factor if num_workers > 0 else None)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    return train_loader, val_loader

if __name__ == "__main__":
    root_dir = "ade20k_data/ADEData2016"
    batch_size = 16
    img_size = 256
    num_workers = 0

    train_loader, val_loader = get_dataloaders(root_dir=root_dir, 
                                            batch_size=batch_size, 
                                            img_size=img_size, 
                                            num_workers=num_workers,
                                            train_augmentation=True)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Iterate through one batch to verify everything works
    for images in train_loader:
        print(f"Batch shape: {images.shape}")
        break