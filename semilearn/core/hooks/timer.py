# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import time
from .hook import Hook


class TimerHook(Hook):
    """
    Timer Hook
    """

    def before_run(self, algorithm):
        if torch.cuda.is_available():
            algorithm.start_batch = torch.cuda.Event(enable_timing=True)
            algorithm.end_batch = torch.cuda.Event(enable_timing=True)
            algorithm.start_run = torch.cuda.Event(enable_timing=True)
            algorithm.end_run = torch.cuda.Event(enable_timing=True)
            algorithm.start_batch.record()
            algorithm.use_cuda_events = True
        else:
            algorithm.start_batch_time = time.time()
            algorithm.end_batch_time = time.time()
            algorithm.use_cuda_events = False

    def before_train_step(self, algorithm):
        if algorithm.use_cuda_events:
            algorithm.end_batch.record()
        else:
            algorithm.end_batch_time = time.time()

    def after_train_step(self, algorithm):
        algorithm.log_dict['lr'] = algorithm.optimizer.param_groups[-1]['lr']
        if algorithm.use_cuda_events:
            algorithm.log_dict['train/prefecth_time'] = algorithm.start_batch.elapsed_time(algorithm.end_batch) / 1000.
            algorithm.start_batch.record()
        else:
            algorithm.log_dict['train/prefecth_time'] = algorithm.end_batch_time - algorithm.start_batch_time
            algorithm.start_batch_time = time.time()