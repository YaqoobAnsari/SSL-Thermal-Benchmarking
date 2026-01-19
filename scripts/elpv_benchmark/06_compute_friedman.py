#!/usr/bin/env python
"""
06_compute_friedman.py - Compute Friedman rankings for SSL algorithms

Computes:
1. Friedman rank for each SSL algorithm
2. Statistical significance tests
3. Post-hoc Nemenyi test for pairwise comparisons

Usage:
    python 06_compute_friedman.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import friedmanchisquare, rankdata

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def compute_friedman_ranks(df, metric='test_accuracy'):
    """
    Compute Friedman ranks for SSL algorithms.
    
    For each (backbone, num_labels) combination:
    - Rank algorithms 1-N based on metric (1 = best)
    - Average ranks across all combinations
    """
    
    # Get unique values
    algorithms = df['algorithm'].unique()
    backbones = df['backbone'].unique()
    label_amounts = df['num_labels'].unique()
    
    print(f"Computing Friedman ranks for {len(algorithms)} algorithms")
    print(f"  Backbones: {len(backbones)}")
    print(f"  Label amounts: {len(label_amounts)}")
    
    rankings = []
    
    # For each (backbone, labels) combination
    for backbone in backbones:
        for num_labels in label_amounts:
            # Get mean metric for each algorithm (averaged over seeds)
            subset = df[(df['backbone'] == backbone) & (df['num_labels'] == num_labels)]
            
            if len(subset) == 0:
                continue
            
            algo_means = subset.groupby('algorithm')[metric].mean()
            
            # Rank (1 = best, higher metric is better for accuracy)
            ranks = rankdata(-algo_means.values)  # Negative because higher is better
            
            for algo, rank, value in zip(algo_means.index, ranks, algo_means.values):
                rankings.append({
                    'algorithm': algo,
                    'backbone': backbone,
                    'num_labels': num_labels,
                    'rank': rank,
                    metric: value
                })
    
    rankings_df = pd.DataFrame(rankings)
    
    return rankings_df


def friedman_test(rankings_df, metric='test_accuracy'):
    """
    Perform Friedman test for statistical significance.
    
    H0: All algorithms perform equally
    H1: At least one algorithm performs differently
    """
    
    # Create pivot table: rows = (backbone, num_labels), columns = algorithms
    pivot = rankings_df.pivot_table(
        values=metric,
        index=['backbone', 'num_labels'],
        columns='algorithm'
    )
    
    # Remove rows with missing values
    pivot = pivot.dropna()
    
    if len(pivot) < 2:
        print("Warning: Not enough data for Friedman test")
        return None, None
    
    # Perform Friedman test
    try:
        stat, p_value = friedmanchisquare(*[pivot[col].values for col in pivot.columns])
        return stat, p_value
    except Exception as e:
        print(f"Warning: Friedman test failed: {e}")
        return None, None


def nemenyi_critical_difference(n_algorithms, n_datasets, alpha=0.05):
    """
    Compute Nemenyi critical difference for post-hoc test.
    
    CD = q_alpha * sqrt(k(k+1)/(6N))
    where k = number of algorithms, N = number of datasets
    """
    # Critical values for Nemenyi test (alpha=0.05)
    # Source: Demsar (2006) "Statistical Comparisons of Classifiers over Multiple Data Sets"
    q_alpha = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
        14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458,
        18: 3.489, 19: 3.517, 20: 3.544
    }
    
    if n_algorithms not in q_alpha:
        # Approximate for larger values
        q = 2.576  # z_0.005 for large k
    else:
        q = q_alpha[n_algorithms]
    
    cd = q * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_datasets))
    
    return cd


def main():
    print("="*70)
    print("ELPV SSL BENCHMARK - FRIEDMAN RANKING")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Get directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    aggregated_dir = os.path.join(base_dir, 'results', 'elpv_benchmark', 'aggregated')
    
    # Load master results
    master_file = os.path.join(aggregated_dir, 'master_results.csv')
    
    if not os.path.exists(master_file):
        print(f"ERROR: Master results not found: {master_file}")
        print("Please run 05_aggregate_results.py first")
        return 1
    
    df = pd.read_csv(master_file)
    print(f"Loaded {len(df)} experiment results")
    
    # Filter out supervised baseline for SSL comparison
    df_ssl = df[df['algorithm'] != 'fullysupervised']
    
    # Compute Friedman ranks
    print("\n" + "-"*50)
    print("Computing Friedman ranks...")
    print("-"*50)
    
    rankings_df = compute_friedman_ranks(df_ssl, metric='test_accuracy')
    
    # Average rank per algorithm
    avg_ranks = rankings_df.groupby('algorithm')['rank'].agg(['mean', 'std', 'count'])
    avg_ranks = avg_ranks.sort_values('mean')
    avg_ranks.columns = ['avg_rank', 'std_rank', 'n_comparisons']
    
    print("\nAverage Friedman Ranks (lower is better):")
    print("-"*50)
    for algo, row in avg_ranks.iterrows():
        print(f"  {algo:15s}: {row['avg_rank']:.2f} ± {row['std_rank']:.2f} (n={int(row['n_comparisons'])})")
    
    # Friedman statistical test
    print("\n" + "-"*50)
    print("Friedman Statistical Test")
    print("-"*50)
    
    stat, p_value = friedman_test(rankings_df, metric='test_accuracy')
    
    if stat is not None:
        print(f"  Friedman statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("  Result: SIGNIFICANT (p < 0.05)")
            print("  Interpretation: At least one algorithm performs significantly differently")
        else:
            print("  Result: NOT SIGNIFICANT (p >= 0.05)")
            print("  Interpretation: No significant difference between algorithms")
    
    # Nemenyi critical difference
    n_algorithms = len(df_ssl['algorithm'].unique())
    n_datasets = len(rankings_df.groupby(['backbone', 'num_labels']))
    
    cd = nemenyi_critical_difference(n_algorithms, n_datasets)
    
    print(f"\n  Nemenyi Critical Difference (CD): {cd:.3f}")
    print(f"  Algorithms with rank difference > {cd:.3f} are significantly different")
    
    # Compute pairwise differences
    print("\n" + "-"*50)
    print("Pairwise Rank Differences")
    print("-"*50)
    
    algorithms = avg_ranks.index.tolist()
    pairwise = pd.DataFrame(index=algorithms, columns=algorithms, dtype=float)
    
    for algo1 in algorithms:
        for algo2 in algorithms:
            diff = abs(avg_ranks.loc[algo1, 'avg_rank'] - avg_ranks.loc[algo2, 'avg_rank'])
            pairwise.loc[algo1, algo2] = diff
    
    # Save results
    print("\nSaving results...")
    
    # Save detailed rankings
    rankings_file = os.path.join(aggregated_dir, 'friedman_detailed_rankings.csv')
    rankings_df.to_csv(rankings_file, index=False)
    print(f"  Saved: {rankings_file}")
    
    # Save average ranks
    avg_ranks_file = os.path.join(aggregated_dir, 'friedman_ranks.csv')
    avg_ranks.to_csv(avg_ranks_file)
    print(f"  Saved: {avg_ranks_file}")
    
    # Save pairwise differences
    pairwise_file = os.path.join(aggregated_dir, 'friedman_pairwise_diff.csv')
    pairwise.to_csv(pairwise_file)
    print(f"  Saved: {pairwise_file}")
    
    # Save statistical test results
    stats_results = {
        'friedman_statistic': float(stat) if stat else None,
        'friedman_p_value': float(p_value) if p_value else None,
        'n_algorithms': n_algorithms,
        'n_datasets': n_datasets,
        'nemenyi_critical_difference': float(cd),
        'alpha': 0.05,
        'significant': bool(p_value < 0.05) if p_value else None,
        'average_ranks': avg_ranks.to_dict(),
    }
    
    stats_file = os.path.join(aggregated_dir, 'statistical_tests.json')
    with open(stats_file, 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"  Saved: {stats_file}")
    
    # Also compute ranks by label amount
    print("\n" + "-"*50)
    print("Ranks by Label Amount")
    print("-"*50)
    
    for num_labels in sorted(df_ssl['num_labels'].unique()):
        subset = rankings_df[rankings_df['num_labels'] == num_labels]
        label_ranks = subset.groupby('algorithm')['rank'].mean().sort_values()
        
        print(f"\n  {num_labels} labels:")
        for algo, rank in label_ranks.head(3).items():
            print(f"    {algo}: {rank:.2f}")
    
    print("\n" + "="*70)
    print("FRIEDMAN RANKING COMPLETE")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
