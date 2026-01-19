#!/usr/bin/env python
"""
07_track_experiments.py - Comprehensive Experiment Tracking Dashboard

Features:
- Real-time status monitoring (PENDING, RUNNING, COMPLETED, FAILED, INTERRUPTED)
- Progress tracking with ETA estimation
- SLURM queue integration
- Detailed metrics for completed experiments
- Log file analysis for running/failed jobs
- Summary statistics and breakdowns by backbone/algorithm

Usage:
    python 07_track_experiments.py              # Full dashboard
    python 07_track_experiments.py --summary    # Quick summary only
    python 07_track_experiments.py --watch      # Auto-refresh every 30s
    python 07_track_experiments.py --running    # Show running jobs details
    python 07_track_experiments.py --failed     # Show failed jobs with errors
    python 07_track_experiments.py --completed  # Show completed with metrics
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / 'results' / 'elpv_benchmark' / 'experiments'
CONFIGS_DIR = BASE_DIR / 'configs' / 'elpv_benchmark'
SLURM_LOGS_DIR = BASE_DIR / 'scripts' / 'elpv_benchmark' / 'slurm' / 'logs'

# Experiment matrix
SSL_ALGORITHMS = ['supervised', 'fixmatch', 'flexmatch', 'freematch', 'softmatch', 
                  'meanteacher', 'abc', 'darp', 'daso']
BACKBONES = ['wrn_28_2', 'vit_b_16']
LABEL_AMOUNTS = [10, 50, 200, 800]
SEEDS = [0, 1, 2]


# ============================================================================
# SLURM INTEGRATION
# ============================================================================
def get_slurm_jobs():
    """Get current SLURM job status for this user."""
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', 'unknown'), '-o', '%A,%a,%j,%T,%M,%N,%r'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {}
        
        jobs = {}
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # Skip header
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 6:
                    job_id = parts[0]
                    array_id = parts[1]
                    name = parts[2]
                    state = parts[3]
                    time_used = parts[4]
                    node = parts[5]
                    reason = parts[6] if len(parts) > 6 else ''
                    
                    full_id = f"{job_id}_{array_id}" if array_id else job_id
                    jobs[full_id] = {
                        'job_id': job_id,
                        'array_id': array_id,
                        'name': name,
                        'state': state,
                        'time': time_used,
                        'node': node,
                        'reason': reason
                    }
        return jobs
    except Exception as e:
        return {}


def get_slurm_history():
    """Get recently completed SLURM jobs."""
    try:
        result = subprocess.run(
            ['sacct', '-u', os.environ.get('USER', 'unknown'), 
             '--format=JobID,JobName,State,ExitCode,Elapsed,Start,End',
             '--starttime=now-2days', '--parsable2'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {}
        
        history = {}
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split('|')
                if len(parts) >= 4 and 'elpv_ssl' in parts[1]:
                    job_id = parts[0]
                    state = parts[2]
                    exit_code = parts[3]
                    history[job_id] = {
                        'state': state,
                        'exit_code': exit_code
                    }
        return history
    except Exception:
        return {}


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
def get_expected_experiments():
    """Get list of all expected experiments."""
    experiments = []
    for algo in SSL_ALGORITHMS:
        for backbone in BACKBONES:
            for labels in LABEL_AMOUNTS:
                for seed in SEEDS:
                    exp_name = f"{algo}_{backbone}_{labels}_seed{seed}"
                    experiments.append({
                        'name': exp_name,
                        'algorithm': algo,
                        'backbone': backbone,
                        'labels': labels,
                        'seed': seed
                    })
    return experiments


def get_experiment_status(exp_name):
    """Get detailed status of a single experiment."""
    exp_dir = RESULTS_DIR / exp_name
    status_file = exp_dir / 'status.json'
    metrics_file = exp_dir / 'metrics.json'
    checkpoint_file = exp_dir / 'metrics_checkpoint.json'
    
    result = {
        'name': exp_name,
        'status': 'PENDING',
        'dir': str(exp_dir),
        'has_status_file': False,
        'has_metrics': False,
        'has_checkpoint': False,
    }
    
    if not exp_dir.exists():
        return result
    
    # Check status.json
    if status_file.exists():
        result['has_status_file'] = True
        try:
            with open(status_file) as f:
                status_data = json.load(f)
            result.update(status_data)
        except json.JSONDecodeError:
            result['status'] = 'CORRUPTED'
    
    # Check metrics.json (final results)
    if metrics_file.exists():
        result['has_metrics'] = True
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            result['metrics'] = {
                'val_best_accuracy': metrics.get('val_best_accuracy'),
                'test_accuracy': metrics.get('test_accuracy'),
                'test_balanced_accuracy': metrics.get('test_balanced_accuracy'),
                'test_f1': metrics.get('test_f1'),
                'test_precision': metrics.get('test_precision'),
                'test_recall': metrics.get('test_recall'),
                'total_time_hours': metrics.get('runtime', {}).get('total_time_seconds', 0) / 3600,
            }
            # If we have final metrics, it's completed
            if result['status'] != 'COMPLETED':
                result['status'] = 'COMPLETED'
        except json.JSONDecodeError:
            pass
    
    # Check checkpoint (intermediate progress)
    if checkpoint_file.exists():
        result['has_checkpoint'] = True
        try:
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            result['progress'] = {
                'current_iter': checkpoint.get('runtime', {}).get('current_iteration', 0),
                'total_iter': checkpoint.get('runtime', {}).get('total_iterations', 0),
                'progress_percent': checkpoint.get('runtime', {}).get('progress_percent', 0),
                'elapsed_hours': checkpoint.get('runtime', {}).get('elapsed_seconds', 0) / 3600,
            }
            # If has checkpoint but no final metrics, might be interrupted
            if result['status'] == 'PENDING' or result['status'] == 'RUNNING':
                # Check if status file says running but no recent activity
                if result.get('status') == 'RUNNING':
                    # Check modification time
                    mtime = checkpoint_file.stat().st_mtime
                    age_minutes = (time.time() - mtime) / 60
                    if age_minutes > 10:  # No update in 10 minutes
                        result['status'] = 'INTERRUPTED'
                        result['stale_minutes'] = age_minutes
        except json.JSONDecodeError:
            pass
    
    return result


def analyze_slurm_log(exp_name, log_type='out'):
    """Analyze SLURM log file for an experiment."""
    # Find matching log file
    for log_file in SLURM_LOGS_DIR.glob(f'elpv_*_{exp_name[-1]}.{log_type}'):
        # This is a rough match, we'd need the actual array task ID
        pass
    
    # For now, return None - we'd need to map exp_name to array task ID
    return None


# ============================================================================
# REPORTING
# ============================================================================
def generate_report(experiments):
    """Generate comprehensive status report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total': len(experiments),
        'by_status': defaultdict(list),
        'by_backbone': defaultdict(lambda: defaultdict(int)),
        'by_algorithm': defaultdict(lambda: defaultdict(int)),
        'by_labels': defaultdict(lambda: defaultdict(int)),
        'metrics_summary': defaultdict(list),
    }
    
    for exp in experiments:
        status_info = get_experiment_status(exp['name'])
        status = status_info.get('status', 'PENDING')
        
        # Merge experiment info with status
        exp_full = {**exp, **status_info}
        report['by_status'][status].append(exp_full)
        
        # Breakdowns
        report['by_backbone'][exp['backbone']][status] += 1
        report['by_algorithm'][exp['algorithm']][status] += 1
        report['by_labels'][exp['labels']][status] += 1
        
        # Collect metrics for completed
        if status == 'COMPLETED' and 'metrics' in status_info:
            report['metrics_summary'][exp['algorithm']].append({
                'backbone': exp['backbone'],
                'labels': exp['labels'],
                'seed': exp['seed'],
                **status_info['metrics']
            })
    
    return report


def print_dashboard(report, args):
    """Print comprehensive dashboard."""
    clear_screen = '\033[2J\033[H' if args.watch else ''
    print(clear_screen)
    
    print("╔" + "═"*78 + "╗")
    print("║" + " ELPV SSL BENCHMARK - EXPERIMENT DASHBOARD ".center(78) + "║")
    print("╠" + "═"*78 + "╣")
    print(f"║  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<64}║")
    print("╚" + "═"*78 + "╝")
    
    # ========== SUMMARY ==========
    total = report['total']
    status_counts = {s: len(exps) for s, exps in report['by_status'].items()}
    
    completed = status_counts.get('COMPLETED', 0)
    running = status_counts.get('RUNNING', 0)
    pending = status_counts.get('PENDING', 0)
    failed = status_counts.get('FAILED', 0)
    interrupted = status_counts.get('INTERRUPTED', 0)
    
    print("\n┌─ OVERALL PROGRESS " + "─"*59 + "┐")
    
    # Progress bar
    pct_done = 100 * completed / total if total > 0 else 0
    bar_width = 50
    filled = int(bar_width * pct_done / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"│  [{bar}] {pct_done:5.1f}%  │")
    
    print(f"│  {'Total:':<15} {total:>6}                                              │")
    print(f"│  {'✅ Completed:':<15} {completed:>6} ({100*completed/total:5.1f}%)                              │")
    print(f"│  {'🔄 Running:':<15} {running:>6} ({100*running/total:5.1f}%)                              │")
    print(f"│  {'⏳ Pending:':<15} {pending:>6} ({100*pending/total:5.1f}%)                              │")
    print(f"│  {'❌ Failed:':<15} {failed:>6} ({100*failed/total:5.1f}%)                              │")
    print(f"│  {'⚠️  Interrupted:':<15} {interrupted:>6} ({100*interrupted/total:5.1f}%)                              │")
    print("└" + "─"*78 + "┘")
    
    # ========== ETA ESTIMATION ==========
    if completed > 0:
        # Calculate average time per experiment from completed ones
        total_time = sum(
            exp.get('metrics', {}).get('total_time_hours', 0) 
            for exp in report['by_status'].get('COMPLETED', [])
        )
        avg_time = total_time / completed if completed > 0 else 1.0
        remaining = pending + interrupted
        eta_hours = remaining * avg_time / 4  # 4 parallel jobs
        
        print(f"\n┌─ TIME ESTIMATES " + "─"*61 + "┐")
        print(f"│  Avg time per experiment: {avg_time:.2f} hours                              │")
        print(f"│  Remaining experiments: {remaining}                                        │")
        print(f"│  Estimated time remaining: {eta_hours:.1f} hours (~{eta_hours/24:.1f} days)                   │")
        print("└" + "─"*78 + "┘")
    
    if args.summary:
        return
    
    # ========== BREAKDOWN BY BACKBONE ==========
    print("\n┌─ BY BACKBONE " + "─"*64 + "┐")
    print(f"│  {'Backbone':<15} {'Done':>8} {'Run':>6} {'Pend':>6} {'Fail':>6} {'Total':>8}  │")
    print("│  " + "─"*55 + "  │")
    for backbone in BACKBONES:
        stats = report['by_backbone'][backbone]
        done = stats.get('COMPLETED', 0)
        run = stats.get('RUNNING', 0)
        pend = stats.get('PENDING', 0)
        fail = stats.get('FAILED', 0) + stats.get('INTERRUPTED', 0)
        tot = done + run + pend + fail
        print(f"│  {backbone:<15} {done:>8} {run:>6} {pend:>6} {fail:>6} {tot:>8}  │")
    print("└" + "─"*78 + "┘")
    
    # ========== BREAKDOWN BY ALGORITHM ==========
    print("\n┌─ BY ALGORITHM " + "─"*63 + "┐")
    print(f"│  {'Algorithm':<15} {'Done':>8} {'Run':>6} {'Pend':>6} {'Fail':>6} {'Total':>8}  │")
    print("│  " + "─"*55 + "  │")
    for algo in SSL_ALGORITHMS:
        stats = report['by_algorithm'][algo]
        done = stats.get('COMPLETED', 0)
        run = stats.get('RUNNING', 0)
        pend = stats.get('PENDING', 0)
        fail = stats.get('FAILED', 0) + stats.get('INTERRUPTED', 0)
        tot = done + run + pend + fail
        print(f"│  {algo:<15} {done:>8} {run:>6} {pend:>6} {fail:>6} {tot:>8}  │")
    print("└" + "─"*78 + "┘")
    
    # ========== RUNNING JOBS DETAILS ==========
    if args.running or (running > 0 and not args.completed and not args.failed):
        running_exps = report['by_status'].get('RUNNING', [])
        if running_exps:
            print(f"\n┌─ RUNNING JOBS ({len(running_exps)}) " + "─"*56 + "┐")
            for exp in running_exps[:10]:
                progress = exp.get('progress', {})
                pct = progress.get('progress_percent', 0)
                elapsed = progress.get('elapsed_hours', 0)
                node = exp.get('node', 'unknown')
                print(f"│  {exp['name']:<45} {pct:5.1f}% {elapsed:.1f}h {node:<10}│")
            if len(running_exps) > 10:
                print(f"│  ... and {len(running_exps) - 10} more                                           │")
            print("└" + "─"*78 + "┘")
    
    # ========== FAILED JOBS DETAILS ==========
    if args.failed or (failed + interrupted > 0 and not args.running and not args.completed):
        failed_exps = report['by_status'].get('FAILED', []) + report['by_status'].get('INTERRUPTED', [])
        if failed_exps:
            print(f"\n┌─ FAILED/INTERRUPTED ({len(failed_exps)}) " + "─"*48 + "┐")
            for exp in failed_exps[:10]:
                status = exp.get('status', 'UNKNOWN')
                exit_code = exp.get('exit_code', '?')
                print(f"│  {exp['name']:<50} {status:<12} exit:{exit_code:<3}│")
            if len(failed_exps) > 10:
                print(f"│  ... and {len(failed_exps) - 10} more                                           │")
            print("└" + "─"*78 + "┘")
    
    # ========== COMPLETED METRICS SUMMARY ==========
    if args.completed and completed > 0:
        print(f"\n┌─ COMPLETED EXPERIMENTS METRICS " + "─"*45 + "┐")
        print(f"│  {'Algorithm':<12} {'Backbone':<12} {'Labels':>6} {'TestAcc':>8} {'BalAcc':>8} {'F1':>8} │")
        print("│  " + "─"*60 + "  │")
        
        for algo in SSL_ALGORITHMS:
            metrics_list = report['metrics_summary'].get(algo, [])
            if metrics_list:
                # Average across seeds
                for backbone in BACKBONES:
                    for labels in LABEL_AMOUNTS:
                        matching = [m for m in metrics_list 
                                   if m['backbone'] == backbone and m['labels'] == labels]
                        if matching:
                            avg_acc = sum(m.get('test_accuracy', 0) or 0 for m in matching) / len(matching)
                            avg_bal = sum(m.get('test_balanced_accuracy', 0) or 0 for m in matching) / len(matching)
                            avg_f1 = sum(m.get('test_f1', 0) or 0 for m in matching) / len(matching)
                            print(f"│  {algo:<12} {backbone:<12} {labels:>6} {avg_acc:>8.4f} {avg_bal:>8.4f} {avg_f1:>8.4f} │")
        print("└" + "─"*78 + "┘")
    
    # ========== SLURM QUEUE ==========
    slurm_jobs = get_slurm_jobs()
    if slurm_jobs:
        print(f"\n┌─ SLURM QUEUE ({len(slurm_jobs)} jobs) " + "─"*53 + "┐")
        running_jobs = [j for j in slurm_jobs.values() if j['state'] == 'RUNNING']
        pending_jobs = [j for j in slurm_jobs.values() if j['state'] == 'PENDING']
        print(f"│  Running: {len(running_jobs):<5} Pending in queue: {len(pending_jobs):<5}                        │")
        for job in list(running_jobs)[:4]:
            print(f"│    [{job['job_id']}_{job['array_id']}] {job['state']:<10} {job['time']:<10} {job['node']:<12}│")
        print("└" + "─"*78 + "┘")
    
    print()


def save_report(report, output_dir):
    """Save detailed report to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full JSON report
    report_file = output_dir / 'experiment_status.json'
    
    # Convert defaultdicts to regular dicts for JSON
    json_report = {
        'timestamp': report['timestamp'],
        'total': report['total'],
        'summary': {s: len(exps) for s, exps in report['by_status'].items()},
        'by_backbone': {k: dict(v) for k, v in report['by_backbone'].items()},
        'by_algorithm': {k: dict(v) for k, v in report['by_algorithm'].items()},
        'by_labels': {k: dict(v) for k, v in report['by_labels'].items()},
    }
    
    with open(report_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"Report saved to: {report_file}")
    
    # Lists for scripting
    for status in ['PENDING', 'FAILED', 'INTERRUPTED', 'COMPLETED']:
        list_file = output_dir / f'{status.lower()}_experiments.txt'
        with open(list_file, 'w') as f:
            for exp in report['by_status'].get(status, []):
                f.write(f"{exp['name']}\n")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='ELPV SSL Benchmark - Experiment Tracking Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 07_track_experiments.py              # Full dashboard
  python 07_track_experiments.py --summary    # Quick summary
  python 07_track_experiments.py --watch      # Auto-refresh every 30s
  python 07_track_experiments.py --running    # Details on running jobs
  python 07_track_experiments.py --failed     # Details on failed jobs
  python 07_track_experiments.py --completed  # Show metrics for completed
  python 07_track_experiments.py --save       # Save report to files
        """
    )
    parser.add_argument('--summary', action='store_true', 
                        help='Show summary only')
    parser.add_argument('--watch', action='store_true', 
                        help='Auto-refresh every 30 seconds')
    parser.add_argument('--running', action='store_true', 
                        help='Show details for running experiments')
    parser.add_argument('--failed', action='store_true', 
                        help='Show details for failed experiments')
    parser.add_argument('--completed', action='store_true', 
                        help='Show metrics for completed experiments')
    parser.add_argument('--save', action='store_true', 
                        help='Save report to files')
    parser.add_argument('--interval', type=int, default=30,
                        help='Refresh interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Get expected experiments
    experiments = get_expected_experiments()
    
    if not experiments:
        print("No experiments configured.")
        return 1
    
    while True:
        # Generate report
        report = generate_report(experiments)
        
        # Print dashboard
        print_dashboard(report, args)
        
        # Save if requested
        if args.save:
            output_dir = BASE_DIR / 'results' / 'elpv_benchmark' / 'aggregated'
            save_report(report, output_dir)
        
        if not args.watch:
            break
        
        print(f"Refreshing in {args.interval} seconds... (Ctrl+C to exit)")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
