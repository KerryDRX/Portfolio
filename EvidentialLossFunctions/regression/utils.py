import json
import numpy as np
from collections import defaultdict


class Logger:
    def __init__(self):
        self.logger = defaultdict(list)
    def log(self, k, v):
        self.logger[k].append(v)
    def average(self, k):
        self.logger[k] = np.mean(self.logger[k])
    def save(self, path):
        with open(path, 'w') as file: json.dump(self.logger, file, indent=4)