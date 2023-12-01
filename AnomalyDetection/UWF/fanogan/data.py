from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from transforms import build_transforms
from config import cfg


def retrieve_paths():
    '''
    Split images into train/validation/test set. Save their paths into a dictionary.
        train:      60% good images.
        validation: 20% good images + 50% bad images.
        test:       20% good images + 50% bad images.

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
        'train':      good_paths[:int(len(good_paths) * 0.6)],
        'validation': good_paths[int(len(good_paths) * 0.6):int(len(good_paths) * 0.8)] + bad_paths[:int(len(bad_paths) * 0.5)],
        'test':       good_paths[int(len(good_paths) * 0.8):] +                           bad_paths[int(len(bad_paths) * 0.5):],
    }
    return paths

class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms, return_label=True):
        self.image_paths = image_paths
        self.transforms = transforms
        self.return_label = return_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.transforms(image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))['image']
        if self.return_label:
            label = torch.ByteTensor([0 if image_path.split('/')[-2] == cfg.DATA.GOOD_LABEL else 1])
            return image, label
        else:
            return image
    
def build_dataloaders():
    '''
    Construct dataloaders.
        train: training dataloader with augmentation.
        train_identity2: training dataloader without augmentation, with 2 channels.
        train_identity3: training dataloader without augmentation, with 3 channels.
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
        'train':           ImageDataset(paths['train'],      transforms['augmentation']),
        'train_identity2': ImageDataset(paths['train'],      transforms['identity_2channels']),
        'train_identity3': ImageDataset(paths['train'],      transforms['identity_3channels'], return_label=False),
        'validation':      ImageDataset(paths['validation'], transforms['identity_2channels']),
        'test':            ImageDataset(paths['test'],       transforms['identity_2channels']),
    }
    dataloaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=cfg.TRAINER.BATCH_SIZE if mode in ['train', 'train_identity3'] else 1,
            shuffle=(mode == 'train'),
            drop_last=(mode == 'train'),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
        )
        for mode in datasets
    }
    return dataloaders
