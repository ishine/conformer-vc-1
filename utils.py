import os
import re
import random
import pathlib
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def latest_checkpoint(model_dir):
    d = pathlib.Path(model_dir)
    assert d.exists(), 'directory is not exists.'
    checkpoints = list(d.glob('*.ckpt'))
    latest_ckpt = sorted(checkpoints, key=lambda x: re.sub(r'\D', '', x.name))
    return latest_ckpt[-1]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def mean(self):
        return self.avg


class Tracker(defaultdict):
    def __init__(self):
        super().__init__(AverageMeter)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k].update(v)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)