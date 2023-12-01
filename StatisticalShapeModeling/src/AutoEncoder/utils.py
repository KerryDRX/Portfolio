import torch
import numpy as np
import random
import os
from config import cfg
from collections import defaultdict


# def log(msg='', pt=True, pbar=None):
#     if pbar:
#         pbar.write(msg)
#     elif pt:
#         print(msg)
#     mkdir(cfg.PATHS.OUTPUT_DIR)
#     with open(f'{cfg.PATHS.OUTPUT_DIR}/log.txt', 'a') as f:
#         f.write(f'{msg}\n')

def print_DSC(DSC):
    if len(DSC) == 1:
        return f'{DSC[0]:.3f}'
    else:
        msg = ' '.join([f'{dsc:.3f}' for dsc in DSC])
        return f'[{msg} mean={np.mean(DSC):.3f}]'

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def set_random_seed(seed=-1):
#     if seed == -1:
#         seed = np.random.randint(0, 100000)
#     log(f'Seed: {seed}')
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

class Metrics:
    def __init__(self):
        self.metrics = defaultdict(list)
    def log(self, dictionary):
        for key, value in dictionary.items():
            self.metrics[key].append(value)
    def get(self, key):
        return self.metrics[key] if key in self.metrics else []
    