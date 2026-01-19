# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ELPV Dataset Integration - FIXED VERSION
# 
# Key fixes:
# 1. Uses pre-generated splits from .npy files for reproducibility
# 2. Proper train/val/test separation
# 3. Correct include_lb_to_ulb handling
# 4. Seed-controlled randomness

import os
import numpy as np
from PIL import Image
from torchvision import transforms
from .datasetbase import BasicDataset


class ELPV_Dataset(BasicDataset):
    """
    ELPV Solar Cell Defect Detection Dataset

    Dataset of 2,624 grayscale images (300x300) of solar cells from electroluminescence images.
    Binary classification: functional vs defective cells based on probability threshold.
    
    This version loads data using pre-generated index files for reproducibility.
    """

    def __init__(self,
                 alg,
                 data,
                 targets,
                 num_classes=2,
                 transform=None,
                 is_ulb=False,
                 strong_transform=None,
                 *args,
                 **kwargs):
        """
        Args:
            alg: Algorithm name
            data: Image data array
            targets: Label array
            num_classes: Number of classes (2 for ELPV)
            transform: Transform to apply
            is_ulb: Whether this is unlabeled data
            strong_transform: Strong augmentation for SSL
        """
        super(ELPV_Dataset, self).__init__(
            alg=alg, 
            data=data, 
            targets=targets,
            num_classes=num_classes,
            transform=transform,
            is_ulb=is_ulb,
            strong_transform=strong_transform,
            *args, 
            **kwargs
        )


def load_elpv_data():
    """
    Load the full ELPV dataset.
    
    Returns:
        images: numpy array of shape (2624, 300, 300, 3) - RGB images
        labels: numpy array of shape (2624,) - binary labels
    """
    from elpv_dataset.utils import load_dataset
    
    images, proba, types = load_dataset()
    
    # Convert probability to binary labels (threshold=0.5)
    # 0: functional, 1: defective
    labels = (proba >= 0.5).astype(np.int64)
    
    # Convert grayscale to 3-channel RGB for compatibility with pretrained models
    images_rgb = np.stack([images, images, images], axis=-1)
    
    return images_rgb, labels


def get_elpv(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    """
    Get ELPV dataset for SSL with proper train/val/test splits.
    
    This function loads PRE-GENERATED splits from .npy files to ensure
    reproducibility across all experiments.

    Args:
        args: Argument object (must contain: seed, img_size, crop_ratio)
        alg: SSL algorithm name
        name: Dataset name ('elpv')
        num_labels: Number of labeled samples
        num_classes: Number of classes (2 for ELPV)
        data_dir: Data directory
        include_lb_to_ulb: Include labeled data in unlabeled pool

    Returns:
        lb_dset: Labeled dataset
        ulb_dset: Unlabeled dataset
        val_dset: Validation dataset
        test_dset: Test dataset
    """
    seed = args.seed
    img_size = args.img_size
    crop_ratio = args.crop_ratio
    
    # Path to pre-generated splits
    split_dir = os.path.join(data_dir, 'elpv', 'splits')
    
    # Check if pre-generated splits exist
    val_idx_path = os.path.join(split_dir, 'val_indices.npy')
    test_idx_path = os.path.join(split_dir, 'test_indices.npy')
    lb_idx_path = os.path.join(split_dir, 'train', f'seed{seed}', f'lb_{num_labels}.npy')
    ulb_idx_path = os.path.join(split_dir, 'train', f'seed{seed}', f'ulb_{num_labels}.npy')
    
    if not all(os.path.exists(p) for p in [val_idx_path, test_idx_path, lb_idx_path, ulb_idx_path]):
        raise FileNotFoundError(
            f"Pre-generated splits not found. Please run scripts/elpv_benchmark/01_prepare_data.py first.\n"
            f"Expected files:\n"
            f"  - {val_idx_path}\n"
            f"  - {test_idx_path}\n"
            f"  - {lb_idx_path}\n"
            f"  - {ulb_idx_path}"
        )
    
    # Load full dataset
    all_images, all_labels = load_elpv_data()
    
    # Load pre-generated indices
    val_indices = np.load(val_idx_path)
    test_indices = np.load(test_idx_path)
    lb_indices = np.load(lb_idx_path)
    ulb_indices = np.load(ulb_idx_path)
    
    # Handle include_lb_to_ulb
    if include_lb_to_ulb:
        ulb_indices = np.concatenate([lb_indices, ulb_indices])
    
    # Extract data for each split
    lb_data = all_images[lb_indices]
    lb_targets = all_labels[lb_indices]
    
    ulb_data = all_images[ulb_indices]
    ulb_targets = all_labels[ulb_indices]
    
    val_data = all_images[val_indices]
    val_targets = all_labels[val_indices]
    
    test_data = all_images[test_indices]
    test_targets = all_labels[test_indices]
    
    # Define transforms
    # Weak augmentation for labeled and unlabeled data
    transform_weak = transforms.Compose([
        transforms.Resize((int(np.ceil(img_size / crop_ratio)), int(np.ceil(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Strong augmentation for unlabeled data (RandAugment-style)
    transform_strong = transforms.Compose([
        transforms.Resize((int(np.ceil(img_size / crop_ratio)), int(np.ceil(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Evaluation transform (no augmentation)
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    lb_dset = ELPV_Dataset(
        alg=alg,
        data=lb_data,
        targets=lb_targets,
        num_classes=num_classes,
        transform=transform_weak,
        is_ulb=False
    )
    
    ulb_dset = ELPV_Dataset(
        alg=alg,
        data=ulb_data,
        targets=ulb_targets,
        num_classes=num_classes,
        transform=transform_weak,
        is_ulb=True,
        strong_transform=transform_strong
    )
    
    val_dset = ELPV_Dataset(
        alg=alg,
        data=val_data,
        targets=val_targets,
        num_classes=num_classes,
        transform=transform_val,
        is_ulb=False
    )
    
    test_dset = ELPV_Dataset(
        alg=alg,
        data=test_data,
        targets=test_targets,
        num_classes=num_classes,
        transform=transform_val,
        is_ulb=False
    )
    
    # Print dataset info
    print(f"\n{'='*60}")
    print(f"ELPV Dataset Loaded (seed={seed}, num_labels={num_labels})")
    print(f"{'='*60}")
    print(f"  Labeled:   {len(lb_dset):5d} samples (Class 0: {np.sum(lb_targets==0)}, Class 1: {np.sum(lb_targets==1)})")
    print(f"  Unlabeled: {len(ulb_dset):5d} samples (includes labeled: {include_lb_to_ulb})")
    print(f"  Validation:{len(val_dset):5d} samples (Class 0: {np.sum(val_targets==0)}, Class 1: {np.sum(val_targets==1)})")
    print(f"  Test:      {len(test_dset):5d} samples (Class 0: {np.sum(test_targets==0)}, Class 1: {np.sum(test_targets==1)})")
    print(f"{'='*60}\n")
    
    return lb_dset, ulb_dset, val_dset, test_dset
