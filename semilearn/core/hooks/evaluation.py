# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py
# 
# FIXED: Proper test set evaluation with all metrics

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training and final test evaluation.
    
    Key behaviors:
    1. During training: Evaluate on validation set every num_eval_iter iterations
    2. After training: Load best model and evaluate on TEST set (separate from validation)
    3. Save best model based on validation accuracy
    """
    
    def after_train_step(self, algorithm):
        """Evaluate on validation set during training."""
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("Validating on eval set...")
            eval_dict = algorithm.evaluate('eval')
            algorithm.log_dict.update(eval_dict)

            # Update best metrics based on validation accuracy
            if algorithm.log_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
                algorithm.best_eval_acc = algorithm.log_dict['eval/top-1-acc']
                algorithm.best_it = algorithm.it
                
                # Save best model
                if not algorithm.args.multiprocessing_distributed or \
                   (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
                    save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
                    algorithm.save_model('model_best.pth', save_path)
                    algorithm.print_fn(f"New best model saved! Val Acc: {algorithm.best_eval_acc:.4f} at iter {algorithm.best_it}")
    
    def after_run(self, algorithm):
        """
        After training completes:
        1. Save latest model
        2. Load best model
        3. Evaluate on TEST set (not validation!)
        4. Store all results
        """
        save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
        
        # Save latest model
        if not algorithm.args.multiprocessing_distributed or \
           (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            algorithm.save_model('latest_model.pth', save_path)

        # Initialize results dict with validation metrics
        results_dict = {
            'eval/best_acc': algorithm.best_eval_acc, 
            'eval/best_it': algorithm.best_it
        }
        
        # Evaluate on TEST set if available
        if 'test' in algorithm.loader_dict and algorithm.loader_dict['test'] is not None:
            algorithm.print_fn("\n" + "="*60)
            algorithm.print_fn("FINAL TEST SET EVALUATION")
            algorithm.print_fn("="*60)
            
            # Load the best model (based on validation performance)
            best_model_path = os.path.join(save_path, 'model_best.pth')
            if os.path.exists(best_model_path):
                algorithm.print_fn(f"Loading best model from: {best_model_path}")
                algorithm.load_model(best_model_path)
            else:
                algorithm.print_fn(f"Warning: Best model not found at {best_model_path}, using current model")
            
            # Evaluate on test set
            test_dict = algorithm.evaluate('test')
            
            # Store all test metrics
            results_dict['test/accuracy'] = test_dict['test/top-1-acc']
            results_dict['test/balanced_accuracy'] = test_dict['test/balanced_acc']
            results_dict['test/precision'] = test_dict['test/precision']
            results_dict['test/recall'] = test_dict['test/recall']
            results_dict['test/f1'] = test_dict['test/F1']
            results_dict['test/loss'] = test_dict['test/loss']
            
            algorithm.print_fn(f"\nTest Results:")
            algorithm.print_fn(f"  Accuracy:          {results_dict['test/accuracy']:.4f}")
            algorithm.print_fn(f"  Balanced Accuracy: {results_dict['test/balanced_accuracy']:.4f}")
            algorithm.print_fn(f"  Precision:         {results_dict['test/precision']:.4f}")
            algorithm.print_fn(f"  Recall:            {results_dict['test/recall']:.4f}")
            algorithm.print_fn(f"  F1 Score:          {results_dict['test/f1']:.4f}")
            algorithm.print_fn(f"  Loss:              {results_dict['test/loss']:.4f}")
            algorithm.print_fn("="*60 + "\n")
        else:
            algorithm.print_fn("Warning: No test set available for final evaluation!")
        
        algorithm.results_dict = results_dict
