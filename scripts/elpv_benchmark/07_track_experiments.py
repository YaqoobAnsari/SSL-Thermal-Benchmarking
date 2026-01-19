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
            ['squeue', '-u', os.environ.get('USER', 'unknown'), '-o', '%i %j %T %M %N %r'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {}
        
        jobs = {}
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # Skip header
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 4:
                    job_id_full = parts[0]  # e.g., "2237_[5-108%4]" or "2237_1"
                    name = parts[1]
                    state = parts[2]
                    time_used = parts[3]
                    node = parts[4] if len(parts) > 4 else ''
                    reason = parts[5] if len(parts) > 5 else ''
                    
                    # Parse job_id and array_id
                    if '_' in job_id_full:
                        job_id, array_id = job_id_full.split('_', 1)
                    else:
                        job_id = job_id_full
                        array_id = ''
                    
                    jobs[job_id_full] = {
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


def parse_log_for_progress(log_file):
    """Parse log.txt to get current iteration progress."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for iteration logs like "256 iteration" or "512 iteration"
        import re
        matches = re.findall(r'(\d+) iteration', content)
        if matches:
            current_iter = int(matches[-1])  # Get last iteration number
            # Total iterations from config (default 16384 for WRN, 8192 for ViT)
            total_iter = 16384  # Default
            if 'vit' in log_file.lower():
                total_iter = 8192
            progress_pct = 100 * current_iter / total_iter
            return {
                'current_iter': current_iter,
                'total_iter': total_iter,
                'progress_percent': progress_pct,
            }
    except Exception:
        pass
    return None


def get_experiment_status(exp_name):
    """Get detailed status of a single experiment."""
    exp_dir = RESULTS_DIR / exp_name
    status_file = exp_dir / 'status.json'
    metrics_file = exp_dir / 'metrics.json'
    checkpoint_file = exp_dir / 'metrics_checkpoint.json'
    log_file = exp_dir / 'log.txt'
    
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
    
    # Parse log.txt for progress (even before checkpoint exists)
    if log_file.exists():
        log_progress = parse_log_for_progress(str(log_file))
        if log_progress:
            result['progress'] = log_progress
        
        # Check if log was updated recently (within 15 mins) to determine if truly running
        log_mtime = log_file.stat().st_mtime
        log_age_minutes = (time.time() - log_mtime) / 60
        if result.get('status') == 'RUNNING' and log_age_minutes < 15:
            # Log is fresh, job is definitely running
            pass
        elif result.get('status') == 'RUNNING' and log_age_minutes >= 15:
            # Log is stale, might be interrupted
            result['status'] = 'INTERRUPTED'
            result['stale_minutes'] = log_age_minutes
    
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
            # Note: staleness check now done via log.txt above
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
                current_iter = progress.get('current_iter', 0)
                total_iter = progress.get('total_iter', 16384)
                # Calculate elapsed from start time
                elapsed_hrs = 0
                if 'started_at' in exp:
                    try:
                        start = datetime.fromisoformat(exp['started_at'])
                        elapsed_hrs = (datetime.now() - start).total_seconds() / 3600
                    except:
                        pass
                node = exp.get('node', 'n/a')
                print(f"│  {exp['name']:<40} {current_iter:>5}/{total_iter} {pct:5.1f}% {elapsed_hrs:.1f}h │")
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
        running_jobs = [j for j in slurm_jobs.values() if j['state'] == 'RUNNING']
        pending_jobs = [j for j in slurm_jobs.values() if j['state'] == 'PENDING']
        # Count pending array jobs properly
        pending_count = 0
        for j in pending_jobs:
            # Array job format: [5-108%4] means 104 jobs pending
            if '[' in j.get('array_id', ''):
                import re
                match = re.search(r'\[(\d+)-(\d+)', j['array_id'])
                if match:
                    pending_count += int(match.group(2)) - int(match.group(1)) + 1
                else:
                    pending_count += 1
            else:
                pending_count += 1
        
        print(f"\n┌─ SLURM QUEUE " + "─"*64 + "┐")
        print(f"│  Running: {len(running_jobs):<5} Pending: {pending_count:<5} (limited to 4 concurrent)           │")
        for job in list(running_jobs)[:4]:
            job_time = job.get('time', '0:00')
            job_node = job.get('node', 'n/a')
            print(f"│    Job {job['job_id']}_{job['array_id']}: {job['state']:<10} {job_time:<10} {job_node:<12}│")
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
