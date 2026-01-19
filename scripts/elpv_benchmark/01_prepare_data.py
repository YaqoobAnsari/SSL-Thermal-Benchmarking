#!/usr/bin/env python
"""
01_prepare_data.py - Prepare ELPV dataset splits for SSL benchmark

This script creates FIXED, REPRODUCIBLE data splits:
1. Train/Val/Test split (70/15/15)
2. For each seed (0, 1, 2):
   - Sample labeled indices for each label count (10, 25, 50, 100, 200, 400, 800)
   - Remaining train data becomes unlabeled

The splits are saved as .npy files and MUST be generated before running experiments.
This ensures ALL experiments use the SAME data splits for fair comparison.

Usage:
    python 01_prepare_data.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# Configuration
SEEDS = [0, 1, 2]
LABEL_AMOUNTS = [10, 50, 200, 800]  # Reduced to 4 key points
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_CLASSES = 2


def load_elpv_data():
    """Load the full ELPV dataset."""
    from elpv_dataset.utils import load_dataset
    
    print("Loading ELPV dataset...")
    images, proba, types = load_dataset()
    
    # Convert probability to binary labels (threshold=0.5)
    labels = (proba >= 0.5).astype(np.int64)
    
    print(f"  Total samples: {len(images)}")
    print(f"  Class 0 (functional): {np.sum(labels == 0)}")
    print(f"  Class 1 (defective): {np.sum(labels == 1)}")
    
    return images, labels


def create_stratified_split(labels, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Create stratified train/val/test split.
    
    Returns indices for each split, maintaining class proportions.
    """
    np.random.seed(seed)
    
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Split each class separately to maintain stratification
    for c in range(NUM_CLASSES):
        class_indices = indices[labels == c]
        np.random.shuffle(class_indices)
        
        n_class = len(class_indices)
        n_train = int(n_class * train_ratio)
        n_val = int(n_class * val_ratio)
        
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])
    
    # Shuffle final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def sample_labeled_indices(train_indices, labels, num_labels, seed):
    """
    Sample labeled indices from training set (stratified).
    
    Returns:
        lb_indices: Indices for labeled data
        ulb_indices: Indices for unlabeled data (remaining train data)
    """
    np.random.seed(seed)
    
    train_labels = labels[train_indices]
    
    lb_indices = []
    ulb_indices = []
    
    # Sample equally from each class
    samples_per_class = num_labels // NUM_CLASSES
    
    for c in range(NUM_CLASSES):
        # Get indices within train_indices that belong to this class
        class_mask = train_labels == c
        class_positions = np.where(class_mask)[0]
        np.random.shuffle(class_positions)
        
        # Take samples_per_class for labeled
        lb_positions = class_positions[:samples_per_class]
        ulb_positions = class_positions[samples_per_class:]
        
        lb_indices.extend(train_indices[lb_positions])
        ulb_indices.extend(train_indices[ulb_positions])
    
    # Shuffle
    lb_indices = np.array(lb_indices)
    ulb_indices = np.array(ulb_indices)
    np.random.shuffle(lb_indices)
    np.random.shuffle(ulb_indices)
    
    return lb_indices, ulb_indices


def verify_no_overlap(train_idx, val_idx, test_idx):
    """Verify no overlap between splits."""
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    assert len(train_set & val_set) == 0, "Overlap between train and val!"
    assert len(train_set & test_set) == 0, "Overlap between train and test!"
    assert len(val_set & test_set) == 0, "Overlap between val and test!"
    
    return True


def verify_class_balance(indices, labels, split_name):
    """Verify class balance in split."""
    split_labels = labels[indices]
    class_counts = [np.sum(split_labels == c) for c in range(NUM_CLASSES)]
    
    print(f"  {split_name}: {len(indices)} samples")
    for c in range(NUM_CLASSES):
        pct = 100 * class_counts[c] / len(indices)
        print(f"    Class {c}: {class_counts[c]} ({pct:.1f}%)")
    
    return class_counts


def main():
    print("="*70)
    print("ELPV SSL BENCHMARK - DATA PREPARATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds: {SEEDS}")
    print(f"Label amounts: {LABEL_AMOUNTS}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print("="*70 + "\n")
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    split_dir = os.path.join(base_dir, 'data', 'elpv', 'splits')
    
    # Load data
    images, labels = load_elpv_data()
    n_total = len(labels)
    
    # Create FIXED train/val/test split (using seed 42 for reproducibility)
    print("\n" + "-"*50)
    print("Creating train/val/test split...")
    print("-"*50)
    
    train_indices, val_indices, test_indices = create_stratified_split(
        labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, seed=42
    )
    
    # Verify no overlap
    verify_no_overlap(train_indices, val_indices, test_indices)
    print("✅ No overlap between splits")
    
    # Verify class balance
    verify_class_balance(train_indices, labels, "Train")
    verify_class_balance(val_indices, labels, "Val")
    verify_class_balance(test_indices, labels, "Test")
    
    # Save val and test indices (FIXED for all experiments)
    os.makedirs(split_dir, exist_ok=True)
    
    val_path = os.path.join(split_dir, 'val_indices.npy')
    test_path = os.path.join(split_dir, 'test_indices.npy')
    
    np.save(val_path, val_indices)
    np.save(test_path, test_indices)
    
    print(f"\n✅ Saved: {val_path}")
    print(f"✅ Saved: {test_path}")
    
    # Create labeled/unlabeled splits for each seed and label amount
    print("\n" + "-"*50)
    print("Creating labeled/unlabeled splits for each seed...")
    print("-"*50)
    
    split_info = {
        'created': datetime.now().isoformat(),
        'total_samples': n_total,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_indices),
        'seeds': SEEDS,
        'label_amounts': LABEL_AMOUNTS,
        'splits': {}
    }
    
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        seed_dir = os.path.join(split_dir, 'train', f'seed{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        
        split_info['splits'][f'seed{seed}'] = {}
        
        for num_labels in LABEL_AMOUNTS:
            lb_indices, ulb_indices = sample_labeled_indices(
                train_indices, labels, num_labels, seed
            )
            
            # Verify labeled indices are from train set
            assert all(idx in train_indices for idx in lb_indices), "Labeled indices not in train!"
            
            # Verify class balance in labeled set
            lb_labels = labels[lb_indices]
            class_counts = [np.sum(lb_labels == c) for c in range(NUM_CLASSES)]
            
            print(f"  {num_labels:4d} labels: Class 0={class_counts[0]}, Class 1={class_counts[1]}, Unlabeled={len(ulb_indices)}")
            
            # Save
            lb_path = os.path.join(seed_dir, f'lb_{num_labels}.npy')
            ulb_path = os.path.join(seed_dir, f'ulb_{num_labels}.npy')
            
            np.save(lb_path, lb_indices)
            np.save(ulb_path, ulb_indices)
            
            split_info['splits'][f'seed{seed}'][f'{num_labels}'] = {
                'labeled': len(lb_indices),
                'unlabeled': len(ulb_indices),
                'class_0_labeled': int(class_counts[0]),
                'class_1_labeled': int(class_counts[1]),
            }
    
    # Save split info
    info_path = os.path.join(split_dir, 'split_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"\n✅ Saved: {info_path}")
    
    # Final verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Verify all files exist
    all_ok = True
    
    for path in [val_path, test_path]:
        if os.path.exists(path):
            print(f"✅ {os.path.basename(path)}")
        else:
            print(f"❌ {os.path.basename(path)} - MISSING")
            all_ok = False
    
    for seed in SEEDS:
        for num_labels in LABEL_AMOUNTS:
            lb_path = os.path.join(split_dir, 'train', f'seed{seed}', f'lb_{num_labels}.npy')
            ulb_path = os.path.join(split_dir, 'train', f'seed{seed}', f'ulb_{num_labels}.npy')
            
            for path in [lb_path, ulb_path]:
                if os.path.exists(path):
                    loaded = np.load(path)
                    # Quick sanity check
                    if 'lb_' in path:
                        # Actual labeled count may be slightly less due to integer division
                        expected_min = (num_labels // NUM_CLASSES) * NUM_CLASSES
                        expected_max = num_labels
                        if not (expected_min <= len(loaded) <= expected_max):
                            print(f"⚠️  {path}: expected {expected_min}-{expected_max}, got {len(loaded)}")
                    # ulb files contain remaining train data, no strict check needed
                else:
                    print(f"❌ {path} - MISSING")
                    all_ok = False
    
    if all_ok:
        print(f"\n✅ All {len(SEEDS) * len(LABEL_AMOUNTS) * 2 + 2} split files created successfully!")
    else:
        print(f"\n❌ Some files missing - check errors above")
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70 + "\n")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
