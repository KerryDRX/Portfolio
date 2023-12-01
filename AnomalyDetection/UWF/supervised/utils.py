import random
import numpy as np
import torch
import sklearn
import imblearn
import gc
import os
from collections import defaultdict
from config import cfg
import json


metric_functions = {
    'accuracy': lambda y_true, y_pred : sklearn.metrics.accuracy_score(y_true, y_pred),
    'recall': lambda y_true, y_pred : sklearn.metrics.recall_score(y_true, y_pred, zero_division=0),
    'precision': lambda y_true, y_pred : sklearn.metrics.precision_score(y_true, y_pred, zero_division=0),
    'f1': lambda y_true, y_pred : sklearn.metrics.f1_score(y_true, y_pred, zero_division=0),
    'specificity': lambda y_true, y_pred : imblearn.metrics.specificity_score(y_true, y_pred),
    'auc': lambda y_true, y_score : sklearn.metrics.roc_auc_score(y_true, y_score),
}

def log(msg='', pt=True):
    '''
    Print and a message and save it to log.txt.

    Parameters:
    ----------
        msg: str
            Message to print and save.
        pt: bool
            Print the message or not.
    '''
    if pt: print(msg)
    with open(f'{cfg.PATHS.OUTPUT_DIR}/log.txt', 'a') as f: f.write(f'{msg}\n')

def mkdir(path):
    '''
    Create a directory if it does not exist.

    Parameters:
    ----------
        path: str
            Directory to create.
    '''
    if not os.path.exists(path): os.makedirs(path)

def set_random_seed():
    '''
    Set random seed for model training.
    '''
    random.seed(cfg.TRAIN_SEED)
    np.random.seed(cfg.TRAIN_SEED)
    torch.manual_seed(cfg.TRAIN_SEED)
    torch.cuda.manual_seed(cfg.TRAIN_SEED)

class Logger:
    def __init__(self, path):
        self.path = path
        if os.path.exists(self.path):
            with open(self.path, 'r') as file:
                self.metrics = defaultdict(list, json.loads(file.read()))
        else:
            self.metrics = defaultdict(list)
    def log(self, dictionary):
        for key, value in dictionary.items():
            self.metrics[key].append(value)
        with open(self.path, 'w') as file:
            json.dump(self.metrics, file, indent=4)
    def get(self, key):
        return self.metrics[key]
    