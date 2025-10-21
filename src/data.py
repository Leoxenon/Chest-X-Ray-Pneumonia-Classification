"""
Data loading and preprocessing module for chest X-ray pneumonia classification.
"""

import os
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class ChestXRayDataset(Dataset):
    """
    Dataset class for loading chest X-ray images for pneumonia classification.
    
    Args:
        data_dir: Root directory containing the dataset
        split: One of 'train', 'val', or 'test'
        transform: Optional transform to be applied on images
        return_path: If True, returns image path along with image and label
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        return_path: bool = False
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.return_path = return_path
        
        # Define class names
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels."""
        samples = []
        split_dir = os.path.join(self.data_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_path:
            return image, label, img_path
        return image, label


def get_transforms(split: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    Get appropriate transforms for the given dataset split.
    
    Args:
        split: One of 'train', 'val', or 'test'
        img_size: Target image size
        
    Returns:
        Composed transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing the dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        img_size: Target image size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = ChestXRayDataset(
        data_dir=data_dir,
        split='train',
        transform=get_transforms('train', img_size)
    )
    
    val_dataset = ChestXRayDataset(
        data_dir=data_dir,
        split='val',
        transform=get_transforms('val', img_size)
    )
    
    test_dataset = ChestXRayDataset(
        data_dir=data_dir,
        split='test',
        transform=get_transforms('test', img_size)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
