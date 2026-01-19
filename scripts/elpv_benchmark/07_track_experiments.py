#!/usr/bin/env python
"""
07_track_experiments.py - Track experiment status

Scans all experiment directories and reports:
- COMPLETED: Finished successfully
- RUNNING: Currently in progress
- INTERRUPTED: Killed mid-way (can resume)
- FAILED: Error occurred
- PENDING: Not yet started

Usage:
    python 07_track_experiments.py
    python 07_track_experiments.py --summary
    python 07_track_experiments.py --pending-only
    python 07_track_experiments.py --failed-only
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_expected_experiments():
    """Get list of all expected experiments from config files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_list = os.path.join(base_dir, 'configs', 'elpv_benchmark', 'config_list_all.txt')
    
    if not os.path.exists(config_list):
        print(f"Error: Config list not found: {config_list}")
        return []
    
    with open(config_list, 'r') as f:
        configs = [line.strip() for line in f if line.strip()]
    
    # Extract experiment names from config paths
    experiments = []
    for config_path in configs:
        exp_name = os.path.splitext(os.path.basename(config_path))[0]
        experiments.append(exp_name)
    
    return experiments


def get_experiment_status(exp_dir):
    """Get status of a single experiment from its directory."""
    status_file = os.path.join(exp_dir, 'status.json')
    metrics_file = os.path.join(exp_dir, 'metrics.json')
    checkpoint_file = os.path.join(exp_dir, 'metrics_checkpoint.json')
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        return status_data
    
    # Infer status from files
    if os.path.exists(metrics_file):
        # Has final metrics, assume completed
        return {'status': 'COMPLETED', 'inferred': True}
    
    if os.path.exists(checkpoint_file):
        # Has checkpoint but no final metrics
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        return {
            'status': 'INTERRUPTED',
            'inferred': True,
            'progress_percent': checkpoint.get('runtime', {}).get('progress_percent', 0)
        }
    
    # Directory exists but no status info
    return {'status': 'UNKNOWN', 'inferred': True}


def scan_experiments(results_dir):
    """Scan all experiment directories and get status."""
    experiments = {}
    
    if not os.path.exists(results_dir):
        return experiments
    
    for exp_name in os.listdir(results_dir):
        exp_dir = os.path.join(results_dir, exp_name)
        if os.path.isdir(exp_dir):
            experiments[exp_name] = get_experiment_status(exp_dir)
            experiments[exp_name]['dir'] = exp_dir
    
    return experiments


def generate_report(expected, actual):
    """Generate status report."""
    report = {
        'COMPLETED': [],
        'RUNNING': [],
        'INTERRUPTED': [],
        'FAILED': [],
        'PENDING': [],
        'UNKNOWN': [],
    }
    
    for exp_name in expected:
        if exp_name in actual:
            status = actual[exp_name].get('status', 'UNKNOWN')
            report[status].append({
                'name': exp_name,
                **actual[exp_name]
            })
        else:
            report['PENDING'].append({'name': exp_name})
    
    return report


def print_report(report, args):
    """Print formatted report."""
    print("\n" + "="*80)
    print("ELPV SSL BENCHMARK - EXPERIMENT STATUS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Summary
    total = sum(len(v) for v in report.values())
    print(f"\nSUMMARY:")
    print(f"  Total experiments: {total}")
    for status, experiments in report.items():
        if experiments:
            pct = 100 * len(experiments) / total if total > 0 else 0
            emoji = {
                'COMPLETED': '✅',
                'RUNNING': '🔄',
                'INTERRUPTED': '⚠️',
                'FAILED': '❌',
                'PENDING': '⏳',
                'UNKNOWN': '❓'
            }.get(status, '?')
            print(f"  {emoji} {status}: {len(experiments)} ({pct:.1f}%)")
    
    if args.summary:
        return
    
    # Details
    if not args.pending_only and not args.failed_only:
        for status in ['COMPLETED', 'RUNNING', 'INTERRUPTED', 'FAILED', 'PENDING']:
            if report[status]:
                print(f"\n{'-'*40}")
                print(f"{status} ({len(report[status])} experiments):")
                print(f"{'-'*40}")
                for exp in report[status][:10]:  # Show first 10
                    print(f"  - {exp['name']}")
                    if 'test_accuracy' in exp and exp['test_accuracy']:
                        print(f"    Test Acc: {exp['test_accuracy']:.4f}")
                    if 'progress_percent' in exp:
                        print(f"    Progress: {exp['progress_percent']:.1f}%")
                if len(report[status]) > 10:
                    print(f"  ... and {len(report[status]) - 10} more")
    
    if args.pending_only:
        print(f"\nPENDING EXPERIMENTS ({len(report['PENDING'])}):")
        for exp in report['PENDING']:
            print(f"  {exp['name']}")
    
    if args.failed_only:
        print(f"\nFAILED/INTERRUPTED EXPERIMENTS:")
        for exp in report['FAILED'] + report['INTERRUPTED']:
            print(f"  {exp['name']}")
    
    print("\n" + "="*80)


def save_report(report, output_dir):
    """Save report to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'experiment_status.json')
    
    # Flatten for JSON
    flat_report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {status: len(exps) for status, exps in report.items()},
        'experiments': report
    }
    
    with open(report_file, 'w') as f:
        json.dump(flat_report, f, indent=2)
    
    print(f"Report saved to: {report_file}")
    
    # Also save lists for easy scripting
    pending_file = os.path.join(output_dir, 'pending_experiments.txt')
    with open(pending_file, 'w') as f:
        for exp in report['PENDING']:
            f.write(f"{exp['name']}\n")
    
    failed_file = os.path.join(output_dir, 'failed_experiments.txt')
    with open(failed_file, 'w') as f:
        for exp in report['FAILED'] + report['INTERRUPTED']:
            f.write(f"{exp['name']}\n")


def main():
    parser = argparse.ArgumentParser(description='Track experiment status')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--pending-only', action='store_true', help='Show pending experiments only')
    parser.add_argument('--failed-only', action='store_true', help='Show failed experiments only')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    
    args = parser.parse_args()
    
    # Get directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, 'results', 'elpv_benchmark', 'experiments')
    output_dir = os.path.join(base_dir, 'results', 'elpv_benchmark', 'aggregated')
    
    # Get expected experiments
    expected = get_expected_experiments()
    
    if not expected:
        print("No experiments configured. Run 02_generate_configs.py first.")
        return 1
    
    # Scan actual experiments
    actual = scan_experiments(results_dir)
    
    # Generate and print report
    report = generate_report(expected, actual)
    print_report(report, args)
    
    if args.save:
        save_report(report, output_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
