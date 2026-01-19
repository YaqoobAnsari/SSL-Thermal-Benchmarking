#!/usr/bin/env python
"""
04_run_single.py - Run a single SSL experiment

This script runs a single experiment given a config file.
Used by SLURM job arrays to run experiments in parallel.

Usage:
    python 04_run_single.py --config path/to/config.yaml
    python 04_run_single.py --config path/to/config.yaml --gpu 0
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_experiment(config_path, gpu=None):
    """Run a single experiment."""
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_script = os.path.join(base_dir, 'train.py')
    
    # Check config exists
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    # Extract experiment name from config path
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    
    print("="*70)
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print("="*70)
    print(f"Config: {config_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Build command
    cmd = [sys.executable, train_script, '--c', config_path]
    
    # Set GPU if specified
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        print(f"Using GPU: {gpu}")
    
    # Run training
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')
        
        return_code = process.wait()
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*70)
        if return_code == 0:
            print(f"EXPERIMENT COMPLETED SUCCESSFULLY")
        else:
            print(f"EXPERIMENT FAILED (return code: {return_code})")
        print(f"Experiment: {exp_name}")
        print(f"Time: {elapsed_time/60:.2f} minutes")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        return return_code
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        process.terminate()
        return 130
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Run a single SSL experiment')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--gpu', '-g', type=int, default=None,
                        help='GPU ID to use (default: use CUDA_VISIBLE_DEVICES)')
    
    args = parser.parse_args()
    
    return run_experiment(args.config, args.gpu)


if __name__ == '__main__':
    sys.exit(main())
