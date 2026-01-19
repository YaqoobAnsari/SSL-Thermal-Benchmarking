#!/usr/bin/env python
"""
02_generate_configs.py - Generate experiment configs for ELPV SSL benchmark

REVISED LEAN DESIGN:
- 2 backbones: WRN-28-2 (CNN) + ViT-B/16 (Transformer)
- 8 SSL algorithms + 1 supervised baseline
- 4 label amounts: 10, 50, 200, 800
- 3 seeds: 0, 1, 2
- Total: 9 × 2 × 4 × 3 = 216 experiments

Usage:
    python 02_generate_configs.py
"""

import os
import sys
import yaml
from datetime import datetime
from itertools import product

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# LOCKED CONFIGURATION - DO NOT MODIFY DURING EXPERIMENTS
# ============================================================================

SEEDS = [0, 1, 2]
LABEL_AMOUNTS = [10, 50, 200, 800]  # Reduced from 7 to 4

# SOTA SSL algorithms
SOTA_ALGORITHMS = ['fixmatch', 'flexmatch', 'freematch', 'softmatch']

# Imbalance-aware algorithms (use FixMatch as base)
IMB_ALGORITHMS = ['abc', 'darp', 'daso']

# Classic algorithm
CLASSIC_ALGORITHMS = ['meanteacher']

# All SSL algorithms
ALL_SSL_ALGORITHMS = SOTA_ALGORITHMS + IMB_ALGORITHMS + CLASSIC_ALGORITHMS

# Backbones: 1 CNN + 1 Transformer
BACKBONES = ['wrn_28_2', 'vit_b_16']

# Backbone-specific hyperparameters
BACKBONE_CONFIGS = {
    'wrn_28_2': {
        'net': 'wrn_28_2',
        'net_from_name': False,
        'img_size': 96,
        'batch_size': 64,
        'eval_batch_size': 128,
        'lr': 0.03,
        'optim': 'SGD',
        'weight_decay': 5e-4,
        'use_pretrain': False,
        'num_train_iter': 16384,  # 2^14
        'num_eval_iter': 512,
        'num_warmup_iter': 512,
    },
    'vit_b_16': {
        'net': 'vit_base_patch16_224',
        'net_from_name': False,
        'img_size': 224,
        'batch_size': 16,  # Smaller batch for ViT
        'eval_batch_size': 32,
        'lr': 5e-5,
        'optim': 'AdamW',
        'weight_decay': 0.01,
        'use_pretrain': True,
        'layer_decay': 0.65,  # Layer-wise LR decay for ViT
        'num_train_iter': 8192,  # Fewer iterations for pretrained
        'num_eval_iter': 256,
        'num_warmup_iter': 256,
    },
}

# Algorithm-specific hyperparameters
ALGORITHM_CONFIGS = {
    'fixmatch': {
        'hard_label': True,
        'p_cutoff': 0.95,
        'T': 0.5,
    },
    'flexmatch': {
        'hard_label': True,
        'p_cutoff': 0.95,
        'T': 0.5,
        'thresh_warmup': True,
    },
    'freematch': {
        'hard_label': True,
        'T': 0.5,
        'ema_p': 0.999,
        'ent_loss_ratio': 0.001,
    },
    'softmatch': {
        'hard_label': False,
        'T': 0.5,
        'dist_align': True,
        'dist_uniform': True,
        'ema_p': 0.999,
    },
    'meanteacher': {
        'ulb_loss_ratio': 50.0,
    },
    'abc': {
        'hard_label': True,
        'T': 0.5,
        'p_cutoff': 0.95,
        'abc_p_cutoff': 0.95,
    },
    'darp': {
        'hard_label': True,
        'T': 0.5,
        'p_cutoff': 0.95,
        'darp_warmup_epochs': 200,
    },
    'daso': {
        'hard_label': True,
        'T': 0.5,
        'p_cutoff': 0.95,
        'daso_queue_len': 128,
    },
    'fullysupervised': {
        # No SSL-specific params
    },
}

# Common training parameters
COMMON_CONFIG = {
    'dataset': 'elpv',
    'num_classes': 2,
    'crop_ratio': 0.875,
    'epoch': 1,
    
    # SSL parameters
    'uratio': 7,
    'ulb_loss_ratio': 1.0,
    'ema_m': 0.999,
    
    # Data
    'include_lb_to_ulb': True,
    'lb_imb_ratio': 1,
    'ulb_imb_ratio': 1,
    'ulb_num_labels': None,
    'num_workers': 4,
    
    # Other
    'momentum': 0.9,
    'amp': True,  # Enable AMP for faster training
    'clip_grad': 0.0,
    'layer_decay': 1.0,
    'num_log_iter': 256,
    
    # Logging
    'overwrite': True,
    'use_wandb': False,
    'use_aim': False,
    'use_tensorboard': False,
}


def generate_config(algorithm, backbone, num_labels, seed, base_dir):
    """Generate a single experiment config."""
    
    config = COMMON_CONFIG.copy()
    
    # Add backbone-specific config
    backbone_config = BACKBONE_CONFIGS[backbone].copy()
    config.update(backbone_config)
    
    # Set algorithm
    if algorithm in SOTA_ALGORITHMS or algorithm in CLASSIC_ALGORITHMS:
        config['algorithm'] = algorithm
        config['imb_algorithm'] = None
    elif algorithm in IMB_ALGORITHMS:
        config['algorithm'] = 'fixmatch'  # Base algorithm for imbalanced methods
        config['imb_algorithm'] = algorithm
    else:
        config['algorithm'] = 'fullysupervised'
        config['imb_algorithm'] = None
        config['uratio'] = 1
        config['ulb_loss_ratio'] = 0.0
    
    # Add algorithm-specific config
    algo_key = algorithm if algorithm != 'supervised' else 'fullysupervised'
    if algo_key in ALGORITHM_CONFIGS:
        config.update(ALGORITHM_CONFIGS[algo_key])
    
    # Set experiment-specific params
    config['num_labels'] = num_labels
    config['seed'] = seed
    
    # Set paths (ABSOLUTE paths for reliability)
    config['data_dir'] = os.path.join(base_dir, 'data')
    config['save_dir'] = os.path.join(base_dir, 'results', 'elpv_benchmark', 'experiments')
    config['save_name'] = f"{algorithm}_{backbone}_{num_labels}_seed{seed}"
    
    return config


def save_config(config, config_dir):
    """Save config to YAML file."""
    filename = f"{config['save_name']}.yaml"
    filepath = os.path.join(config_dir, filename)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return filepath


def generate_base_configs(base_dir, config_dir):
    """Generate base config templates for each backbone."""
    base_config_dir = os.path.join(config_dir, 'base')
    os.makedirs(base_config_dir, exist_ok=True)
    
    for backbone in BACKBONES:
        config = COMMON_CONFIG.copy()
        config.update(BACKBONE_CONFIGS[backbone])
        
        filepath = os.path.join(base_config_dir, f'{backbone}.yaml')
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"  Created: {filepath}")


def main():
    print("="*70)
    print("ELPV SSL BENCHMARK - CONFIG GENERATION (LEAN DESIGN)")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  Backbones: {BACKBONES}")
    print(f"  SOTA SSL: {SOTA_ALGORITHMS}")
    print(f"  Imbalance-aware: {IMB_ALGORITHMS}")
    print(f"  Classic: {CLASSIC_ALGORITHMS}")
    print(f"  Label amounts: {LABEL_AMOUNTS}")
    print(f"  Seeds: {SEEDS}")
    
    n_ssl = len(ALL_SSL_ALGORITHMS) * len(BACKBONES) * len(LABEL_AMOUNTS) * len(SEEDS)
    n_supervised = len(BACKBONES) * len(LABEL_AMOUNTS) * len(SEEDS)
    print(f"\nExperiment counts:")
    print(f"  SSL experiments: {n_ssl}")
    print(f"  Supervised baselines: {n_supervised}")
    print(f"  Grand total: {n_ssl + n_supervised}")
    print("="*70 + "\n")
    
    # Get directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = os.path.join(base_dir, 'configs', 'elpv_benchmark')
    experiments_dir = os.path.join(config_dir, 'experiments')
    
    # Clean existing configs
    if os.path.exists(experiments_dir):
        import shutil
        shutil.rmtree(experiments_dir)
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Generate base configs
    print("Generating base configs...")
    generate_base_configs(base_dir, config_dir)
    
    # Generate experiment configs
    print("\nGenerating experiment configs...")
    
    generated_files = []
    config_lists = {backbone: [] for backbone in BACKBONES}
    
    # SSL experiments
    for algorithm, backbone, num_labels, seed in product(
        ALL_SSL_ALGORITHMS, BACKBONES, LABEL_AMOUNTS, SEEDS
    ):
        config = generate_config(algorithm, backbone, num_labels, seed, base_dir)
        filepath = save_config(config, experiments_dir)
        generated_files.append(filepath)
        config_lists[backbone].append(filepath)
    
    # Supervised baselines
    for backbone, num_labels, seed in product(BACKBONES, LABEL_AMOUNTS, SEEDS):
        config = generate_config('fullysupervised', backbone, num_labels, seed, base_dir)
        filepath = save_config(config, experiments_dir)
        generated_files.append(filepath)
        config_lists[backbone].append(filepath)
    
    print(f"  Generated {len(generated_files)} config files")
    
    # Save config lists for SLURM job arrays
    print("\nGenerating config lists for SLURM...")
    for backbone, configs in config_lists.items():
        list_file = os.path.join(config_dir, f'config_list_{backbone}.txt')
        with open(list_file, 'w') as f:
            for config_path in configs:
                f.write(f"{config_path}\n")
        print(f"  Created: {list_file} ({len(configs)} configs)")
    
    # Save master config list
    master_list_file = os.path.join(config_dir, 'config_list_all.txt')
    with open(master_list_file, 'w') as f:
        for filepath in generated_files:
            f.write(f"{filepath}\n")
    print(f"  Created: {master_list_file} ({len(generated_files)} configs)")
    
    # Save master config (metadata)
    master_config = {
        'created': datetime.now().isoformat(),
        'design': 'LEAN - IEEE TII Submission',
        'seeds': SEEDS,
        'label_amounts': LABEL_AMOUNTS,
        'ssl_algorithms': ALL_SSL_ALGORITHMS,
        'backbones': BACKBONES,
        'total_experiments': len(generated_files),
        'estimated_time_hours': {
            'wrn_28_2': len([c for c in config_lists['wrn_28_2']]) * 1.0,
            'vit_b_16': len([c for c in config_lists['vit_b_16']]) * 2.5,
        },
        'common_config': COMMON_CONFIG,
        'backbone_configs': BACKBONE_CONFIGS,
        'algorithm_configs': ALGORITHM_CONFIGS,
    }
    
    master_config_file = os.path.join(config_dir, 'master_config.yaml')
    with open(master_config_file, 'w') as f:
        yaml.dump(master_config, f, default_flow_style=False, sort_keys=False)
    print(f"  Created: {master_config_file}")
    
    # Summary
    print("\n" + "="*70)
    print("CONFIG GENERATION COMPLETE")
    print("="*70)
    print(f"Total configs generated: {len(generated_files)}")
    print(f"\nEstimated runtime (4 parallel jobs):")
    print(f"  WRN-28-2: {len(config_lists['wrn_28_2'])} × 1h / 4 = ~{len(config_lists['wrn_28_2'])//4}h")
    print(f"  ViT-B/16: {len(config_lists['vit_b_16'])} × 2.5h / 4 = ~{int(len(config_lists['vit_b_16'])*2.5/4)}h")
    print(f"  Total: ~{len(config_lists['wrn_28_2'])//4 + int(len(config_lists['vit_b_16'])*2.5/4)}h = ~{(len(config_lists['wrn_28_2'])//4 + int(len(config_lists['vit_b_16'])*2.5/4))//24} days")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
