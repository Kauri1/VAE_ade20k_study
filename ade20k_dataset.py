import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Callable, Tuple
from pathlib import Path
from torch.utils.data import Subset
import random

class ADE20KDataset(Dataset):
    """
    Custom Dataset for ADE20K semantic segmentation.
    """

    def __init__(self,
                 root_dir: str,
                 split: str = 'training',
                 img_size: int = 256,
                 transform: Optional[Callable] = None,
                 n_common_labels: Optional[InterruptedError] = None,
                 exclude_concepts: Optional[list] = None,
                 latent_dir: Optional[str] = None,
                 sub_split: Optional[str] = None,
                 split_seed: int = 42):
        """
        Args:
            root_dir (str): Root directory of the ADE20K dataset.
            split (str): Dataset split to use ('training', 'validation').
            img_size (int): Desired image size (images will be resized to img_size x img_size).
            transform (callable, optional): Optional transformation to apply to the images.
            n_common_labels (int): Number of most common labels to consider for filtering. If > 0, only images with these labels will be included in the dataset.
            exclude_concepts (list, optional): List of concepts to exclude from the dataset. If provided, any image whose label is in this list will be filtered out.
            latent_dir (str, optional): Directory for latent representations.
            sub_split (str, optional): Sub-split to use ('train', 'val').
            split_seed (int): Random seed for splitting the dataset.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.n_common_labels = n_common_labels
        self.exclude_concepts = exclude_concepts
        self.latent_dir = self.latent_dir = Path(latent_dir) if latent_dir else None
        self.return_paths = False # Set to True if you want __getitem__ to return image paths along with data and labels
        self.sub_split = sub_split
        self.split_seed = split_seed

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

        #self.labels = self._get_labels(pr) if self.labels_file.exists() else {}
        #self.unique_split_labels = sorted(list(set(self.labels.values()))) if self.labels else []
        #self.unique_split_labels.append("Unknown")  # Fallback for images without labels in this split

        # labels_dict[picture_id] = category_name 
        #self.split_labels_dict = self._get_labels(pr) if self.labels_file.exists() else {}
        self.all_labels_dict = self._get_labels() if self.labels_file.exists() else {}
        #print(all_labels_dict)

        # Filter out images with excluded concepts
        if self.exclude_concepts:
            filtered_image_files = []
            for img_path in self.image_files:
                picture_id = img_path.stem  # Get filename without extension
                label = self.all_labels_dict.get(picture_id, "Unknown")
                if label not in self.exclude_concepts:
                    filtered_image_files.append(img_path)
            
            self.image_files = filtered_image_files
            print(f"After excluding concepts {self.exclude_concepts}, {len(self.image_files)} images remain.")


        self.all_labels_dict = {k: v for k, v in self.all_labels_dict.items() if v not in (self.exclude_concepts or [])}

        #Top n labels
        n = self.n_common_labels if self.n_common_labels is not None else len(set(self.all_labels_dict.values()))
        label_counts = {}
        for img_path in self.image_files:
            label = self.all_labels_dict.get(img_path.stem, "Unknown")
            label_counts[label] = label_counts.get(label, 0) + 1

        top_labels_tuples = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:n]
        top_n_labels_set = {label for label, count in top_labels_tuples}

        print(f"Top {n} most common labels:")
        for label, count in top_labels_tuples:
            print(f"{label}: {count} images", end=", ")
        print()

        filtered_image_files = []
        for img_path in self.image_files:
            picture_id = img_path.stem  # Get filename without extension
            label = self.all_labels_dict.get(picture_id, "Unknown")
            if label in top_n_labels_set:
                filtered_image_files.append(img_path)
            
        self.image_files = filtered_image_files
        print(f"After filtering, {len(self.image_files)} images remain with the top {n} labels.")


        if sub_split in ['train', 'val']:
            rng = random.Random(self.split_seed)
            shuffled_files = self.image_files.copy()
            rng.shuffle(shuffled_files)

            split_idx = int(0.8 * len(shuffled_files))
            if sub_split == 'train':
                self.image_files = shuffled_files[:split_idx]
            else:
                self.image_files = shuffled_files[split_idx:]
            print(f"Selected sub-split '{sub_split}': {len(self.image_files)} images remain.")

        
        #self.unique_classes = sorted(list(set(self.all_labels_dict.values())))
        self.unique_classes = sorted(list(top_n_labels_set))
        #self.unique_classes.append("Unknown") # Fallback

        #print(f"Unique classes found: {self.unique_classes}")

        # class_to_idx[name] = index in unique_classes list
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.unique_classes)}
        #print(f"Class to index mapping: {self.class_to_idx}")

        self.label_indecies = []
        for img_path in self.image_files:
            picture_id = img_path.stem
            cat_name = self.all_labels_dict.get(picture_id, "Unknown")
            self.label_indecies.append(self.class_to_idx[cat_name])

        print(f"Loaded {len(self.all_labels_dict)} class labels from {self.labels_file}")
        print(f"Example label: {next(iter(self.all_labels_dict.values())) if self.all_labels_dict else 'No labels loaded'}")


    


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
        Args:
            prefix: Only load labels for images whose IDs start with this prefix (e.g., 'ADE_train_' or 'ADE_val_'). If empty, load all labels.
        Returns:
            A dictionary mapping image IDs to class labels.
            labels[picture_id] = category_name
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

        label = self.get_class_label(idx)

        if self.return_paths:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label, img_path.stem

        if self.latent_dir is not None:
            picture_id = img_path.stem
            latent_path = self.latent_dir / f"{picture_id}.pt"
            
            # Load the pre-computed latent tensor
            latent_data = torch.load(latent_path, weights_only=True)
            if isinstance(latent_data, dict) and 'mu' in latent_data and 'logvar' in latent_data:
                mu = latent_data['mu']
                
                # Only sample from the distribution during training
                if self.split == 'training' and self.sub_split == 'train':
                    logvar = latent_data['logvar']
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    latent = mu + std * eps
                else:
                    # Use deterministic mean for validation and testing
                    latent = mu
            else:
                latent = latent_data
            
            return latent, label

        image = Image.open(img_path).convert('RGB')

        image = self.transform(image)

        

        return image, label
    
    def get_class_label(self, idx):
        """
        Get class label for the image at the specified index.
        """
        #img_path = self.image_files[idx]
        #picture_id = img_path.stem  # Get filename without extension
        #cat_name = self.all_labels_dict.get(picture_id, "Unknown")
        #return torch.tensor(self.class_to_idx[cat_name], dtype=torch.long)
        return torch.tensor(self.label_indecies[idx], dtype=torch.long)
    
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
                prefetch_factor: int = 4,
                n_common_labels: Optional[int] = None,
                exclude_concepts: Optional[list] = None,
                latent_dir: Optional[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
        n_common_labels: Number of most common labels to consider for filtering (if > 0)
        latent_dir: Directory for latent representations. If provided, dataset will load pre-computed latent tensors instead of images.
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
                                  transform=train_transform,
                                  n_common_labels=n_common_labels,
                                  exclude_concepts=exclude_concepts,
                                  latent_dir=Path(latent_dir) / "train" if latent_dir else None,
                                  sub_split='train')


    val_dataset = ADE20KDataset(root_dir=root_dir, 
                                split='training', 
                                img_size=img_size, 
                                transform=val_transform,
                                n_common_labels=n_common_labels,
                                exclude_concepts=exclude_concepts,
                                latent_dir=Path(latent_dir) / "validation" if latent_dir else None,
                                sub_split='val')
    

    test_dataset = ADE20KDataset(root_dir=root_dir, 
                                split='validation', 
                                img_size=img_size, 
                                transform=val_transform,
                                n_common_labels=n_common_labels,
                                exclude_concepts=exclude_concepts,
                                latent_dir=Path(latent_dir) / "test" if latent_dir else None)
    

    # train_dataset = ADE20KDataset(root_dir=root_dir, 
    #                             split='training', 
    #                             img_size=img_size, 
    #                             transform=train_transform,
    #                             n_common_labels=n_common_labels,
    #                             exclude_concepts=exclude_concepts,
    #                             latent_dir=Path(latent_dir) / "train" if latent_dir else None)

    # val_dataset = ADE20KDataset(root_dir=root_dir, 
    #                             split='validation', 
    #                             img_size=img_size, 
    #                             transform=val_transform,
    #                             n_common_labels=n_common_labels,
    #                             exclude_concepts=exclude_concepts,
    #                             latent_dir=Path(latent_dir) / "validation" if latent_dir else None)
    
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    # train_loader = DataLoader(train_dataset, 
    #                         batch_size=batch_size, 
    #                         shuffle=True, 
    #                         num_workers=num_workers, 
    #                         pin_memory=pin_memory,
    #                         persistent_workers=use_persistent_workers,
    #                         prefetch_factor=prefetch_factor if num_workers > 0 else None)
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     drop_last=False,
    #     persistent_workers=use_persistent_workers,
    #     prefetch_factor=prefetch_factor if num_workers > 0 else None
    # )


    train_files = set(train_dataset.image_files)
    val_files = set(val_dataset.image_files)
    test_files = set(test_dataset.image_files)

    assert train_files.isdisjoint(val_files), "Overlap detected between training and validation sets!"
    assert train_files.isdisjoint(test_files), "Overlap detected between training and test sets!"
    assert val_files.isdisjoint(test_files), "Overlap detected between validation and test sets!"
    


    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    root_dir = "ade20k_data/ADEData2016"
    batch_size = 16
    img_size = 256
    num_workers = 0

    train_loader, val_loader, test_loader = get_dataloaders(root_dir=root_dir, 
                                            batch_size=batch_size, 
                                            img_size=img_size, 
                                            num_workers=num_workers,
                                            train_augmentation=True,
                                            n_common_labels=10,
                                            exclude_concepts=["misc"])

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

