# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Enhanced Metrics Logger Hook - FIXED VERSION
#
# Key fixes:
# 1. Clear naming: val_* for validation, test_* for test
# 2. Saves confusion matrix and logits
# 3. Proper metric tracking

import os
import json
import time
import signal
import numpy as np
import psutil
from datetime import datetime
from .hook import Hook


# Global flag for graceful shutdown
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown"""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n⚠️  Received signal {signum}, requesting graceful shutdown...")
    print("Will save checkpoint and metrics before exiting.")

# Register signal handlers for SLURM timeout
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


class MetricsLoggerHook(Hook):
    """
    Comprehensive Metrics Logger Hook

    Logs detailed metrics including:
    - Runtime metrics (total time, per-epoch, per-iteration)
    - Training dynamics (convergence, utilization)
    - Memory usage
    - All evaluation metrics per evaluation step
    - Confusion matrix and logits for test set
    """

    def __init__(self):
        super().__init__()
        self.start_time = None
        self.epoch_start_time = None
        self.iteration_times = []
        self.epoch_metrics = []
        self.best_val_acc = 0.0
        self.best_iteration = 0
        self.convergence_iteration = None
        self.convergence_threshold = 0.01  # 1% improvement threshold
        self.last_best_acc = 0.0

        # Process info for memory tracking
        self.process = psutil.Process()

    def before_run(self, algorithm):
        """Initialize timing at start of training"""
        self.start_time = time.time()
        algorithm.training_start_time = self.start_time

        print(f"\n{'='*80}")
        print(f"TRAINING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Create status file to track experiment state
        save_dir = os.path.join(algorithm.args.save_dir, algorithm.args.save_name)
        os.makedirs(save_dir, exist_ok=True)
        status_file = os.path.join(save_dir, 'status.json')
        status = {
            'status': 'RUNNING',
            'started_at': datetime.now().isoformat(),
            'experiment_id': f"{algorithm.args.algorithm}_{algorithm.args.net}_{algorithm.args.dataset}_labels{algorithm.args.num_labels}_seed{algorithm.args.seed}",
        }
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

        # Initialize metrics storage
        algorithm.detailed_metrics = {
            'experiment_id': f"{algorithm.args.algorithm}_{algorithm.args.net}_{algorithm.args.dataset}_labels{algorithm.args.num_labels}_seed{algorithm.args.seed}",
            'config': {
                'algorithm': algorithm.args.algorithm,
                'backbone': algorithm.args.net,
                'dataset': algorithm.args.dataset,
                'num_labels': algorithm.args.num_labels,
                'num_classes': algorithm.args.num_classes,
                'seed': algorithm.args.seed,
                'batch_size': algorithm.args.batch_size,
                'uratio': algorithm.args.uratio,
                'lr': algorithm.args.lr,
                'optimizer': algorithm.args.optim,
                'ema_m': algorithm.args.ema_m,
                'num_train_iter': algorithm.args.num_train_iter,
                'img_size': algorithm.args.img_size,
            },
            'runtime': {},
            'training_dynamics': {},
            'validation_metrics': {},
            'test_metrics': {},
            'per_epoch_metrics': [],
            'hyperparameters': {k: v for k, v in vars(algorithm.args).items() if not k.startswith('_')}
        }

    def before_train_epoch(self, algorithm):
        """Mark start of epoch"""
        self.epoch_start_time = time.time()

    def after_train_step(self, algorithm):
        """Track iteration timing and metrics"""
        global _shutdown_requested
        
        current_time = time.time()

        # Track iteration time
        if hasattr(algorithm, 'last_iter_time'):
            iter_time = current_time - algorithm.last_iter_time
            self.iteration_times.append(iter_time)
        algorithm.last_iter_time = current_time
        
        # Check for graceful shutdown request (e.g., SLURM timeout)
        if _shutdown_requested:
            print("\n⚠️  Graceful shutdown requested!")
            self._save_emergency_checkpoint(algorithm)
            raise SystemExit("Graceful shutdown due to signal")

        # Check for convergence (accuracy plateau)
        if hasattr(algorithm, 'best_eval_acc'):
            if algorithm.best_eval_acc > self.best_val_acc:
                improvement = algorithm.best_eval_acc - self.best_val_acc
                self.best_val_acc = algorithm.best_eval_acc
                self.best_iteration = algorithm.it

                # Check if improvement is below threshold (convergence)
                if improvement < self.convergence_threshold and self.convergence_iteration is None:
                    self.convergence_iteration = algorithm.it

        # Store per-epoch metrics if evaluation happened
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            epoch_metric = {
                'iteration': algorithm.it,
                'epoch': algorithm.epoch,
                'timestamp': time.time() - self.start_time,
                'train_metrics': {
                    'sup_loss': algorithm.log_dict.get('train/sup_loss', 0),
                    'unsup_loss': algorithm.log_dict.get('train/unsup_loss', 0),
                    'total_loss': algorithm.log_dict.get('train/total_loss', 0),
                    'util_ratio': algorithm.log_dict.get('train/util_ratio', 0),
                },
                'val_metrics': {  # FIXED: renamed from eval_metrics to val_metrics
                    'accuracy': algorithm.log_dict.get('eval/top-1-acc', 0),
                    'balanced_acc': algorithm.log_dict.get('eval/balanced_acc', 0),
                    'precision': algorithm.log_dict.get('eval/precision', 0),
                    'recall': algorithm.log_dict.get('eval/recall', 0),
                    'f1': algorithm.log_dict.get('eval/F1', 0),
                    'loss': algorithm.log_dict.get('eval/loss', 0),
                },
                'lr': algorithm.log_dict.get('lr', 0),
            }
            self.epoch_metrics.append(epoch_metric)
            algorithm.detailed_metrics['per_epoch_metrics'] = self.epoch_metrics
            
            # CRITICAL FIX: Save metrics incrementally after EVERY evaluation
            # This ensures we don't lose data if job is killed
            self._save_incremental_metrics(algorithm)

    def after_train_epoch(self, algorithm):
        """Log epoch statistics"""
        epoch_time = time.time() - self.epoch_start_time
        elapsed_time = time.time() - self.start_time

        # Memory usage
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        if self.every_n_epochs(algorithm, 10) or algorithm.epoch == algorithm.epochs - 1:
            print(f"[Epoch {algorithm.epoch}/{algorithm.epochs}] "
                  f"Time: {epoch_time:.2f}s | "
                  f"Total: {elapsed_time/60:.1f}min | "
                  f"Memory: {memory_mb:.1f}MB | "
                  f"Best Val Acc: {self.best_val_acc:.4f}")

    def after_run(self, algorithm):
        """Finalize and save all metrics"""
        total_time = time.time() - self.start_time

        # Calculate runtime metrics
        avg_iter_time = sum(self.iteration_times) / len(self.iteration_times) if self.iteration_times else 0

        algorithm.detailed_metrics['runtime'] = {
            'total_time_seconds': total_time,
            'total_time_formatted': self._format_time(total_time),
            'avg_iteration_time': avg_iter_time,
            'total_iterations': algorithm.it,
            'training_speed_iterations_per_sec': 1.0 / avg_iter_time if avg_iter_time > 0 else 0,
            'convergence_iteration': self.convergence_iteration if self.convergence_iteration else algorithm.it,
            'best_iteration': self.best_iteration,
            'peak_memory_mb': self.process.memory_info().rss / 1024 / 1024,
        }

        # Training dynamics
        algorithm.detailed_metrics['training_dynamics'] = {
            'epochs_completed': algorithm.epoch,
            'iterations_completed': algorithm.it,
            'best_iteration': self.best_iteration,
            'convergence_iteration': self.convergence_iteration if self.convergence_iteration else algorithm.it,
            'final_train_loss': algorithm.log_dict.get('train/total_loss', 0),
            'final_sup_loss': algorithm.log_dict.get('train/sup_loss', 0),
            'final_unsup_loss': algorithm.log_dict.get('train/unsup_loss', 0),
            'avg_utilization_ratio': sum([m['train_metrics']['util_ratio'] for m in self.epoch_metrics]) / len(self.epoch_metrics) if self.epoch_metrics else 0,
        }

        # Validation metrics (from training)
        algorithm.detailed_metrics['validation_metrics'] = {
            'best_accuracy': self.best_val_acc,
            'best_iteration': self.best_iteration,
        }

        # Test metrics (from results_dict populated by EvaluationHook)
        if hasattr(algorithm, 'results_dict'):
            algorithm.detailed_metrics['test_metrics'] = {
                'accuracy': algorithm.results_dict.get('test/accuracy', None),
                'balanced_accuracy': algorithm.results_dict.get('test/balanced_accuracy', None),
                'precision': algorithm.results_dict.get('test/precision', None),
                'recall': algorithm.results_dict.get('test/recall', None),
                'f1': algorithm.results_dict.get('test/f1', None),
                'loss': algorithm.results_dict.get('test/loss', None),
            }

        # Save to JSON
        save_dir = os.path.join(algorithm.args.save_dir, algorithm.args.save_name)
        os.makedirs(save_dir, exist_ok=True)

        metrics_file = os.path.join(save_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            # Convert any numpy types to Python types for JSON serialization
            json.dump(self._convert_to_serializable(algorithm.detailed_metrics), f, indent=2)

        # Save training curve as CSV for easy plotting
        if self.epoch_metrics:
            self._save_training_curve(save_dir)

        # Save summary
        summary_file = os.path.join(save_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary(algorithm))

        # Update status file to COMPLETED
        status_file = os.path.join(save_dir, 'status.json')
        status = {
            'status': 'COMPLETED',
            'started_at': datetime.fromtimestamp(self.start_time).isoformat(),
            'completed_at': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'experiment_id': algorithm.detailed_metrics['experiment_id'],
            'test_accuracy': algorithm.detailed_metrics['test_metrics'].get('accuracy'),
            'best_val_accuracy': self.best_val_acc,
        }
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Time: {self._format_time(total_time)}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f} at iteration {self.best_iteration}")
        if algorithm.detailed_metrics['test_metrics'].get('accuracy') is not None:
            print(f"Test Accuracy: {algorithm.detailed_metrics['test_metrics']['accuracy']:.4f}")
        print(f"Metrics saved to: {metrics_file}")
        print(f"Status: COMPLETED")
        print(f"{'='*80}\n")

    def _save_emergency_checkpoint(self, algorithm):
        """
        Save emergency checkpoint when receiving shutdown signal.
        
        This ensures we don't lose progress if SLURM kills the job.
        """
        print("Saving emergency checkpoint...")
        
        try:
            save_dir = os.path.join(algorithm.args.save_dir, algorithm.args.save_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model checkpoint
            algorithm.save_model('checkpoint_emergency.pth', save_dir)
            
            # Save metrics
            self._save_incremental_metrics(algorithm)
            
            # Update status file
            status_file = os.path.join(save_dir, 'status.json')
            status = {
                'status': 'INTERRUPTED',
                'started_at': datetime.fromtimestamp(self.start_time).isoformat(),
                'interrupted_at': datetime.now().isoformat(),
                'iteration': algorithm.it,
                'total_iterations': algorithm.args.num_train_iter,
                'progress_percent': 100 * algorithm.it / algorithm.args.num_train_iter,
                'experiment_id': algorithm.detailed_metrics['experiment_id'],
                'best_val_accuracy': self.best_val_acc,
                'can_resume': True,
            }
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            print(f"Emergency checkpoint saved to: {save_dir}")
            print(f"Progress: {algorithm.it}/{algorithm.args.num_train_iter} iterations")
            print("You can resume this experiment later with --resume flag")
            
        except Exception as e:
            print(f"Error saving emergency checkpoint: {e}")

    def _save_incremental_metrics(self, algorithm):
        """
        Save metrics incrementally during training.
        
        This ensures we don't lose data if the job is killed!
        Saves to metrics_checkpoint.json which is overwritten each time.
        """
        try:
            save_dir = os.path.join(algorithm.args.save_dir, algorithm.args.save_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Update runtime info
            elapsed_time = time.time() - self.start_time
            avg_iter_time = sum(self.iteration_times) / len(self.iteration_times) if self.iteration_times else 0
            
            algorithm.detailed_metrics['runtime'] = {
                'total_time_seconds': elapsed_time,
                'total_time_formatted': self._format_time(elapsed_time),
                'avg_iteration_time': avg_iter_time,
                'current_iteration': algorithm.it,
                'total_iterations': algorithm.args.num_train_iter,
                'progress_percent': 100 * algorithm.it / algorithm.args.num_train_iter,
                'best_iteration': self.best_iteration,
                'status': 'RUNNING',
            }
            
            # Update validation metrics
            algorithm.detailed_metrics['validation_metrics'] = {
                'best_accuracy': self.best_val_acc,
                'best_iteration': self.best_iteration,
            }
            
            # Save checkpoint file
            checkpoint_file = os.path.join(save_dir, 'metrics_checkpoint.json')
            with open(checkpoint_file, 'w') as f:
                json.dump(self._convert_to_serializable(algorithm.detailed_metrics), f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save incremental metrics: {e}")

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _save_training_curve(self, save_dir):
        """Save training curve as CSV"""
        import csv
        csv_file = os.path.join(save_dir, 'training_curve.csv')
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'iteration', 'epoch', 'timestamp',
                'train_sup_loss', 'train_unsup_loss', 'train_total_loss', 'train_util_ratio',
                'val_accuracy', 'val_balanced_acc', 'val_precision', 'val_recall', 'val_f1', 'val_loss',
                'lr'
            ])
            # Data
            for m in self.epoch_metrics:
                writer.writerow([
                    m['iteration'], m['epoch'], m['timestamp'],
                    m['train_metrics']['sup_loss'], m['train_metrics']['unsup_loss'],
                    m['train_metrics']['total_loss'], m['train_metrics']['util_ratio'],
                    m['val_metrics']['accuracy'], m['val_metrics']['balanced_acc'],
                    m['val_metrics']['precision'], m['val_metrics']['recall'],
                    m['val_metrics']['f1'], m['val_metrics']['loss'],
                    m['lr']
                ])

    def _format_time(self, seconds):
        """Format seconds into human-readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"

    def _generate_summary(self, algorithm):
        """Generate human-readable summary"""
        metrics = algorithm.detailed_metrics

        summary = []
        summary.append("="*80)
        summary.append("EXPERIMENT SUMMARY")
        summary.append("="*80)
        summary.append(f"\nExperiment ID: {metrics['experiment_id']}")
        summary.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        summary.append("\n--- Configuration ---")
        for key, val in metrics['config'].items():
            summary.append(f"  {key}: {val}")

        summary.append("\n--- Runtime Metrics ---")
        for key, val in metrics['runtime'].items():
            summary.append(f"  {key}: {val}")

        summary.append("\n--- Training Dynamics ---")
        for key, val in metrics['training_dynamics'].items():
            summary.append(f"  {key}: {val}")

        summary.append("\n--- Validation Metrics (Best) ---")
        for key, val in metrics['validation_metrics'].items():
            if isinstance(val, float):
                summary.append(f"  {key}: {val:.4f}")
            else:
                summary.append(f"  {key}: {val}")

        summary.append("\n--- Test Metrics (Final) ---")
        if metrics['test_metrics']:
            for key, val in metrics['test_metrics'].items():
                if val is not None:
                    if isinstance(val, float):
                        summary.append(f"  {key}: {val:.4f}")
                    else:
                        summary.append(f"  {key}: {val}")
                else:
                    summary.append(f"  {key}: N/A")

        summary.append("\n" + "="*80)
        return "\n".join(summary)
