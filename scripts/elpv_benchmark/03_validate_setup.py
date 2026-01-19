#!/usr/bin/env python
"""
03_validate_setup.py - Validate the complete setup before running experiments

Performs comprehensive validation:
1. Check all data splits exist
2. Check all config files exist
3. Verify config contents are correct
4. Test loading a single experiment
5. Verify output directories are writable

Run this AFTER 01_prepare_data.py and 02_generate_configs.py

Usage:
    python 03_validate_setup.py
"""

import os
import sys
import yaml
import json
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


SEEDS = [0, 1, 2]
LABEL_AMOUNTS = [10, 50, 200, 800]  # Reduced to 4
BACKBONES = ['wrn_28_2', 'vit_b_16']  # Reduced to 2: 1 CNN + 1 Transformer
SSL_ALGORITHMS = ['fixmatch', 'flexmatch', 'freematch', 'softmatch', 'meanteacher', 'abc', 'darp', 'daso']  # 8 algorithms


def validate_data_splits(base_dir):
    """Validate all data splits exist and have correct sizes."""
    print("\n[1/5] Validating data splits...")
    
    split_dir = os.path.join(base_dir, 'data', 'elpv', 'splits')
    
    errors = []
    
    # Check val and test indices
    for name in ['val_indices.npy', 'test_indices.npy']:
        path = os.path.join(split_dir, name)
        if os.path.exists(path):
            data = np.load(path)
            print(f"  ✅ {name}: {len(data)} samples")
        else:
            errors.append(f"Missing: {path}")
            print(f"  ❌ {name}: MISSING")
    
    # Check train splits for each seed and label amount
    for seed in SEEDS:
        for num_labels in LABEL_AMOUNTS:
            lb_path = os.path.join(split_dir, 'train', f'seed{seed}', f'lb_{num_labels}.npy')
            ulb_path = os.path.join(split_dir, 'train', f'seed{seed}', f'ulb_{num_labels}.npy')
            
            if os.path.exists(lb_path):
                lb_data = np.load(lb_path)
                # Allow for rounding due to balanced class sampling (num_labels // num_classes * num_classes)
                expected_min = (num_labels // 2) * 2  # For 2 classes
                if len(lb_data) < expected_min:
                    errors.append(f"Wrong size: {lb_path} (expected >= {expected_min}, got {len(lb_data)})")
            else:
                errors.append(f"Missing: {lb_path}")
            
            if not os.path.exists(ulb_path):
                errors.append(f"Missing: {ulb_path}")
    
    if errors:
        print(f"  ❌ {len(errors)} errors found")
        for err in errors[:5]:
            print(f"     - {err}")
        if len(errors) > 5:
            print(f"     ... and {len(errors) - 5} more")
        return False
    else:
        print(f"  ✅ All {len(SEEDS) * len(LABEL_AMOUNTS) * 2 + 2} split files validated")
        return True


def validate_configs(base_dir):
    """Validate all config files exist and have correct structure."""
    print("\n[2/5] Validating config files...")
    
    config_dir = os.path.join(base_dir, 'configs', 'elpv_benchmark', 'experiments')
    
    errors = []
    configs_found = 0
    
    # Check SSL configs
    for algo in SSL_ALGORITHMS:
        for backbone in BACKBONES:
            for num_labels in LABEL_AMOUNTS:
                for seed in SEEDS:
                    config_name = f"{algo}_{backbone}_{num_labels}_seed{seed}.yaml"
                    config_path = os.path.join(config_dir, config_name)
                    
                    if os.path.exists(config_path):
                        configs_found += 1
                        
                        # Validate contents
                        try:
                            with open(config_path, 'r') as f:
                                config = yaml.safe_load(f)
                            
                            # Check required fields
                            required = ['algorithm', 'num_labels', 'seed', 'save_dir', 'save_name']
                            for field in required:
                                if field not in config:
                                    errors.append(f"Missing field '{field}' in {config_name}")
                            
                            # Check values match filename
                            if config.get('num_labels') != num_labels:
                                errors.append(f"num_labels mismatch in {config_name}")
                            if config.get('seed') != seed:
                                errors.append(f"seed mismatch in {config_name}")
                                
                        except Exception as e:
                            errors.append(f"Error reading {config_name}: {e}")
                    else:
                        errors.append(f"Missing: {config_name}")
    
    # Check supervised configs
    for backbone in BACKBONES:
        for num_labels in LABEL_AMOUNTS:
            for seed in SEEDS:
                config_name = f"fullysupervised_{backbone}_{num_labels}_seed{seed}.yaml"
                config_path = os.path.join(config_dir, config_name)
                
                if os.path.exists(config_path):
                    configs_found += 1
                else:
                    errors.append(f"Missing: {config_name}")
    
    expected = len(SSL_ALGORITHMS) * len(BACKBONES) * len(LABEL_AMOUNTS) * len(SEEDS) + \
               len(BACKBONES) * len(LABEL_AMOUNTS) * len(SEEDS)
    
    if errors:
        print(f"  ❌ {len(errors)} errors found")
        for err in errors[:5]:
            print(f"     - {err}")
        if len(errors) > 5:
            print(f"     ... and {len(errors) - 5} more")
        return False
    else:
        print(f"  ✅ All {configs_found}/{expected} config files validated")
        return True


def validate_config_lists(base_dir):
    """Validate config list files for SLURM."""
    print("\n[3/5] Validating config lists...")
    
    config_dir = os.path.join(base_dir, 'configs', 'elpv_benchmark')
    
    errors = []
    
    for backbone in BACKBONES:
        list_file = os.path.join(config_dir, f'config_list_{backbone}.txt')
        
        if os.path.exists(list_file):
            with open(list_file, 'r') as f:
                lines = f.readlines()
            
            n_configs = len([l for l in lines if l.strip()])
            print(f"  ✅ config_list_{backbone}.txt: {n_configs} configs")
            
            # Check first and last config exist
            if lines:
                first_config = lines[0].strip()
                if not os.path.exists(first_config):
                    errors.append(f"First config missing: {first_config}")
        else:
            errors.append(f"Missing: {list_file}")
            print(f"  ❌ config_list_{backbone}.txt: MISSING")
    
    # Check master list
    master_list = os.path.join(config_dir, 'config_list_all.txt')
    if os.path.exists(master_list):
        with open(master_list, 'r') as f:
            n_total = len([l for l in f.readlines() if l.strip()])
        print(f"  ✅ config_list_all.txt: {n_total} configs")
    else:
        errors.append(f"Missing: {master_list}")
        print(f"  ❌ config_list_all.txt: MISSING")
    
    return len(errors) == 0


def validate_semilearn_imports():
    """Validate semilearn library can load ELPV dataset."""
    print("\n[4/5] Validating semilearn imports...")
    
    try:
        from semilearn.datasets.cv_datasets import get_elpv
        print("  ✅ get_elpv imported successfully")
        
        from semilearn.core.utils import get_dataset
        print("  ✅ get_dataset imported successfully")
        
        from semilearn.algorithms import get_algorithm
        print("  ✅ get_algorithm imported successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_output_directories(base_dir):
    """Validate output directories exist and are writable."""
    print("\n[5/5] Validating output directories...")
    
    dirs_to_check = [
        'results/elpv_benchmark/experiments',
        'results/elpv_benchmark/aggregated',
        'results/elpv_benchmark/figures',
        'logs',
    ]
    
    errors = []
    
    for dir_path in dirs_to_check:
        full_path = os.path.join(base_dir, dir_path)
        
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path, exist_ok=True)
                print(f"  ✅ Created: {dir_path}")
            except Exception as e:
                errors.append(f"Cannot create {dir_path}: {e}")
                print(f"  ❌ Cannot create: {dir_path}")
        else:
            # Check writable
            test_file = os.path.join(full_path, '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"  ✅ Writable: {dir_path}")
            except Exception as e:
                errors.append(f"Not writable {dir_path}: {e}")
                print(f"  ❌ Not writable: {dir_path}")
    
    return len(errors) == 0


def main():
    print("="*70)
    print("ELPV SSL BENCHMARK - SETUP VALIDATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    results = {
        'Data Splits': validate_data_splits(base_dir),
        'Config Files': validate_configs(base_dir),
        'Config Lists': validate_config_lists(base_dir),
        'Semilearn Imports': validate_semilearn_imports(),
        'Output Directories': validate_output_directories(base_dir),
    }
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("\nYou can now run experiments with:")
        print("  python scripts/elpv_benchmark/04_run_single.py --config <config.yaml>")
        print("\nOr submit SLURM jobs with:")
        print("  cd scripts/elpv_benchmark/slurm && ./submit_all.sh")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("\nPlease fix the issues above before running experiments.")
        print("You may need to run:")
        print("  python scripts/elpv_benchmark/01_prepare_data.py")
        print("  python scripts/elpv_benchmark/02_generate_configs.py")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
