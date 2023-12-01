from config import cfg
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transforms import build_transforms


def retrieve_paths():
    '''
    Split images into train/validation/test set. Save their paths into a dictionary.
        train:      60% good images + 60% bad images.
        validation: 20% good images + 20% bad images.
        test:       20% good images + 20% bad images.

    Returns:
    ----------
        paths: dict
            A dictionary storing images paths of train/validation/test sets.
    '''
    good_paths = sorted(glob(f'{cfg.PATHS.DATA_DIR}/{cfg.DATA.GOOD_LABEL}/*.tif'))
    bad_paths = sorted(glob(f'{cfg.PATHS.DATA_DIR}/{cfg.DATA.BAD_LABEL}/*.tif'))
    np.random.default_rng(cfg.DATA_SEED).shuffle(good_paths)
    np.random.default_rng(cfg.DATA_SEED).shuffle(bad_paths)
    paths = {
        'train':      good_paths[:int(len(good_paths) * 0.6)] +                           bad_paths[:int(len(bad_paths) * 0.6)],
        'validation': good_paths[int(len(good_paths) * 0.6):int(len(good_paths) * 0.8)] + bad_paths[int(len(bad_paths) * 0.6):int(len(bad_paths) * 0.8)],
        'test':       good_paths[int(len(good_paths) * 0.8):] +                           bad_paths[int(len(bad_paths) * 0.8):],
    }
    return paths

class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms):
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.transforms(Image.open(image_path).convert('RGB'))
        label = torch.ByteTensor([0 if cfg.DATA.GOOD_LABEL in image_path else 1])
        return image, label
    
def build_dataloaders():
    '''
    Construct dataloaders.
        train: training dataloader with augmentation.
        train_identity: training dataloader without augmentation.
        validation: validation dataloader without augmentation.
        test: test dataloader without augmentation.

    Returns:
    ----------
        dataloaders: dict
            A dictionary storing dataloaders of train/validation/test sets.
    '''
    paths = retrieve_paths()
    transforms = build_transforms()
    datasets = {
        'train':          ImageDataset(paths['train'],      transforms['augmentation']),
        'train_identity': ImageDataset(paths['train'],      transforms['identity']),
        'validation':     ImageDataset(paths['validation'], transforms['identity']),
        'test':           ImageDataset(paths['test'],       transforms['identity']),
    }
    dataloaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=cfg.TRAINER.BATCH_SIZE if mode == 'train' else 1,
            shuffle=(mode == 'train'),
            drop_last=(mode == 'train'),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
        )
        for mode in datasets
    }
    return dataloaders
