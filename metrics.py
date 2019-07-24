import numpy as np
import torch

class MetricsCollection:
    def __init__(self):
        self.metrics = {}

    def add(self, phase, key, value, count=1):
        if phase not in self.metrics:
            self.metrics[phase] = {}

        if key not in self.metrics[phase]:
            self.metrics[phase][key] = AverageMeter(name=key)

        self.metrics[phase][key].add(value, count)

    def __getitem__(self, slice_key):
        return self.metrics[slice_key]


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def add(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(value)

    def best(self, mode='auto'):
        if len(self.history) == 0:
            return None, None

        if mode == 'auto':
            if 'acc' in self.name:
                mode = 'max'
            else:
                mode = 'min'

        if mode == 'min':
            best_index = np.argmin(self.history)
            best_value = np.min(self.history)
        elif mode == 'max':
            best_index = np.argmax(self.history)
            best_value = np.max(self.history)
        else:
            raise Exception('mode not supported: ' + mode)

        return best_value, best_index
