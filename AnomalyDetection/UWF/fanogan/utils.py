import random
import numpy as np
import torch
from collections import defaultdict
from config import cfg
import json
import os
from pytorch_gan_metrics.core import get_inception_feature
from collections import defaultdict
from config import cfg
from pynvml import *


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

def infinitelooper(iterable):
    '''
    Create an infinite data looper from a data loader.

    Parameters:
    ----------
        iterable: torch.utils.data.dataloader.DataLoader
            The dataloader to loop.
    '''
    while True:
        for x in iterable:
            yield x

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

def calc_and_save_stats(dataloader, fid_stats_path):
    '''
    Calculate and save statistics of a dataset.
    The statistics will be used for FID calculation.

    Parameters:
    ----------
        dataloader: torch.utils.data.dataloader.DataLoader
            The dataloader to load the dataset.
        fid_stats_path: str
            The file path to store statistics.
    '''
    acts, = get_inception_feature(dataloader, dims=[2048], verbose=True)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    np.savez_compressed(fid_stats_path, mu=mu, sigma=sigma)

def compute_gradient_penalty(discriminator, real_images, fake_images):
    '''
    Calculate gradient penalty.

    Parameters:
    ----------
        discriminator: Discriminator
            The discriminator of f-AnoGAN.
        real_images: torch.Tensor
            The original training images, with shape [B, C, H, W].
        fake_images: torch.Tensor
            The generated images, with shape [B, C, H, W].
    
    Returns:
    ----------
        gradient_penalty: torch.Tensor
            Gradient penalty.
    '''
    alpha = torch.rand(*real_images.shape[:2], 1, 1,).cuda()
    interpolates = (alpha * real_images + (1 - alpha) * fake_images)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(*d_interpolates.shape).cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates, 
        inputs=interpolates,
        grad_outputs=fake, 
        create_graph=True,
        retain_graph=True, 
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def enable_grad(model, requires_grad):
    '''
    Enable or disable gradient calculation of the model.

    Parameters:
    ----------
        model: torch.nn.Module
            The model to freeze/unfreeze.
        requires_grad: bool
            Enable/disable gradient calculation.
    '''
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def cuda_memory():
    '''
    Show used GPU memory.

    Returns:
    ----------
        msg: str
            Used GPU memory.
    '''
    nvmlInit()
    info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    msg = f'[Used GPU Memory: {info.used/1024**3:.1f}/{info.total/1024**3:.1f}GiB]'
    return msg
