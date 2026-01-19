#!/usr/bin/env python
"""
05_aggregate_results.py - Aggregate results from all experiments

Collects metrics.json from all experiments and creates:
1. master_results.csv - All experiments in one table
2. summary_by_algorithm.csv - Grouped by SSL algorithm
3. summary_by_backbone.csv - Grouped by backbone
4. summary_by_labels.csv - Grouped by label count

Usage:
    python 05_aggregate_results.py
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_experiment_results(results_dir):
    """Load all metrics.json files from experiments directory."""
    
    pattern = os.path.join(results_dir, '*', 'metrics.json')
    metrics_files = glob.glob(pattern)
    
    print(f"Found {len(metrics_files)} experiment results")
    
    results = []
    
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics
            result = {
                'experiment_id': data.get('experiment_id', ''),
                'algorithm': data['config']['algorithm'],
                'backbone': data['config']['backbone'],
                'num_labels': data['config']['num_labels'],
                'seed': data['config']['seed'],
                'batch_size': data['config']['batch_size'],
                'lr': data['config']['lr'],
                
                # Validation metrics
                'val_best_accuracy': data['validation_metrics'].get('best_accuracy', None),
                'val_best_iteration': data['validation_metrics'].get('best_iteration', None),
                
                # Test metrics
                'test_accuracy': data['test_metrics'].get('accuracy', None),
                'test_balanced_accuracy': data['test_metrics'].get('balanced_accuracy', None),
                'test_precision': data['test_metrics'].get('precision', None),
                'test_recall': data['test_metrics'].get('recall', None),
                'test_f1': data['test_metrics'].get('f1', None),
                'test_loss': data['test_metrics'].get('loss', None),
                
                # Training dynamics
                'convergence_iteration': data['training_dynamics'].get('convergence_iteration', None),
                'avg_utilization_ratio': data['training_dynamics'].get('avg_utilization_ratio', None),
                
                # Runtime
                'training_time_seconds': data['runtime'].get('total_time_seconds', None),
                'training_time_hours': data['runtime'].get('total_time_seconds', 0) / 3600,
            }
            
            # Handle imbalanced algorithms
            if 'imb_algorithm' in data.get('hyperparameters', {}):
                imb_alg = data['hyperparameters'].get('imb_algorithm')
                if imb_alg:
                    result['algorithm'] = imb_alg
            
            results.append(result)
            
        except Exception as e:
            print(f"  Warning: Error loading {metrics_file}: {e}")
    
    return pd.DataFrame(results)


def compute_summary_statistics(df, group_cols):
    """Compute mean and std for metrics grouped by specified columns."""
    
    metrics = ['test_accuracy', 'test_balanced_accuracy', 'test_precision', 
               'test_recall', 'test_f1', 'test_loss', 'training_time_hours']
    
    # Group and compute statistics
    grouped = df.groupby(group_cols)
    
    summary = grouped[metrics].agg(['mean', 'std', 'count'])
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary.reset_index()


def main():
    print("="*70)
    print("ELPV SSL BENCHMARK - RESULTS AGGREGATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Get directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, 'results', 'elpv_benchmark', 'experiments')
    output_dir = os.path.join(base_dir, 'results', 'elpv_benchmark', 'aggregated')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    print("Loading experiment results...")
    df = load_experiment_results(results_dir)
    
    if len(df) == 0:
        print("ERROR: No results found!")
        return 1
    
    print(f"Loaded {len(df)} experiments")
    print(f"  Algorithms: {df['algorithm'].nunique()}")
    print(f"  Backbones: {df['backbone'].nunique()}")
    print(f"  Label amounts: {df['num_labels'].nunique()}")
    print(f"  Seeds: {df['seed'].nunique()}")
    
    # Save master results
    print("\nSaving master results...")
    master_file = os.path.join(output_dir, 'master_results.csv')
    df.to_csv(master_file, index=False)
    print(f"  Saved: {master_file}")
    
    # Summary by algorithm
    print("\nComputing summary by algorithm...")
    summary_algo = compute_summary_statistics(df, ['algorithm'])
    summary_algo_file = os.path.join(output_dir, 'summary_by_algorithm.csv')
    summary_algo.to_csv(summary_algo_file, index=False)
    print(f"  Saved: {summary_algo_file}")
    
    # Summary by backbone
    print("\nComputing summary by backbone...")
    summary_backbone = compute_summary_statistics(df, ['backbone'])
    summary_backbone_file = os.path.join(output_dir, 'summary_by_backbone.csv')
    summary_backbone.to_csv(summary_backbone_file, index=False)
    print(f"  Saved: {summary_backbone_file}")
    
    # Summary by labels
    print("\nComputing summary by label count...")
    summary_labels = compute_summary_statistics(df, ['num_labels'])
    summary_labels_file = os.path.join(output_dir, 'summary_by_labels.csv')
    summary_labels.to_csv(summary_labels_file, index=False)
    print(f"  Saved: {summary_labels_file}")
    
    # Summary by algorithm and backbone
    print("\nComputing summary by algorithm and backbone...")
    summary_algo_backbone = compute_summary_statistics(df, ['algorithm', 'backbone'])
    summary_algo_backbone_file = os.path.join(output_dir, 'summary_by_algorithm_backbone.csv')
    summary_algo_backbone.to_csv(summary_algo_backbone_file, index=False)
    print(f"  Saved: {summary_algo_backbone_file}")
    
    # Summary by algorithm and labels
    print("\nComputing summary by algorithm and labels...")
    summary_algo_labels = compute_summary_statistics(df, ['algorithm', 'num_labels'])
    summary_algo_labels_file = os.path.join(output_dir, 'summary_by_algorithm_labels.csv')
    summary_algo_labels.to_csv(summary_algo_labels_file, index=False)
    print(f"  Saved: {summary_algo_labels_file}")
    
    # Print quick summary
    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)
    
    print("\nBest algorithms by mean test accuracy:")
    algo_acc = df.groupby('algorithm')['test_accuracy'].mean().sort_values(ascending=False)
    for algo, acc in algo_acc.head(5).items():
        print(f"  {algo}: {acc:.4f}")
    
    print("\nBest backbones by mean test accuracy:")
    backbone_acc = df.groupby('backbone')['test_accuracy'].mean().sort_values(ascending=False)
    for backbone, acc in backbone_acc.items():
        print(f"  {backbone}: {acc:.4f}")
    
    print("\nAccuracy by label count:")
    label_acc = df.groupby('num_labels')['test_accuracy'].mean()
    for labels, acc in label_acc.items():
        print(f"  {labels:4d} labels: {acc:.4f}")
    
    print("\n" + "="*70)
    print("AGGREGATION COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
