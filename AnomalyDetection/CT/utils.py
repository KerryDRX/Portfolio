import random
import numpy as np
import torch
import torch.autograd as autograd
import torchvision.transforms as T
from pytorch_gan_metrics import get_inception_score_and_fid
from torch.utils.data import DataLoader
from pytorch_gan_metrics.inception import InceptionV3
from pytorch_gan_metrics.core import get_inception_feature
import gc
import os
from tqdm.auto import tqdm
from collections import defaultdict
from config import cfg

def log(msg='', pt=True):
    if pt:
        print(msg)
    with open(f'{cfg.log_dir}/log.txt', 'a') as f:
        f.write(f'{msg}\n')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Metrics:
    def __init__(self):
        self.metrics = defaultdict(list)
    def log(self, dictionary):
        for key, value in dictionary.items():
            self.metrics[key].append(value)
    def get(self, key):
        return self.metrics[key] if key in self.metrics else []

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(*real_samples.shape[:2], 1, 1,).cuda()
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(*d_interpolates.shape).cuda()
    gradients = autograd.grad(
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

def get_inception_feature(images, dims, verbose=False):
    assert all(dim in InceptionV3.BLOCK_INDEX_BY_DIM for dim in dims)
    num_images = min(len(images.dataset), images.batch_size * len(images))
    block_idxs = [InceptionV3.BLOCK_INDEX_BY_DIM[dim] for dim in dims]
    model = InceptionV3(block_idxs).cuda()
    model.eval()
    features = [np.empty((num_images, dim)) for dim in dims]

    pbar = tqdm(
        total=num_images, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_feature")
    looper = iter(images)
    start = 0
    while start < num_images:
        # get a batch of images from iterator
        batch_images = next(looper)[0]
        end = start + len(batch_images)
        # calculate inception feature
        batch_images = batch_images.cuda()
        with torch.no_grad():
            outputs = model(batch_images)
            for feature, output, dim in zip(features, outputs, dims):
                feature[start: end] = output.view(-1, dim).cpu().numpy()
        start = end
        pbar.update(len(batch_images))
    pbar.close()
    return features

def calc_and_save_stats(dataloader):
    acts, = get_inception_feature(dataloader, dims=[2048], verbose=True)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    mkdir(os.path.dirname(cfg.trainer.fid_stats_path))
    np.savez_compressed(cfg.trainer.fid_stats_path, mu=mu, sigma=sigma)

def evaluate_generator(generator):
    resize = T.Compose([
        T.Resize((256, 256), antialias=True),
        T.Lambda(lambda x: x.repeat(1,3,1,1)),
    ])
    generator.eval()
    images = []
    with torch.no_grad():
        for start in range(0, 5000, cfg.latent_dim):
            end = min(start + cfg.latent_dim, 5000)
            z = torch.randn(end - start, cfg.latent_dim).cuda()
            image = generator(z).cpu()
            image = resize(image)
            images.append(image)
    images = (torch.cat(images, dim=0) + 1) / 2
    IS, FID = get_inception_score_and_fid(images, cfg.trainer.fid_stats_path, verbose=False)
    return IS, FID
