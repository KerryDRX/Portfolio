import logging
import numpy as np
import torchio as tio
from glob import glob
from config import cfg
from collections import defaultdict
from torch.utils.data import DataLoader
from transforms import build_transforms

def build_dataloaders_MALPEM(data_dir):
    def _build_subjects(image_paths):
        return [tio.Subject(
            image=tio.ScalarImage(image_path),
            label=tio.LabelMap(image_path.replace('MRI', 'Segmentation')),
        ) for image_path in image_paths]
    
    image_paths = sorted(glob(f'{data_dir}/MRI/*.nii.gz'))
    assert cfg.DATASET.TRAIN_SIZE + cfg.DATASET.VAL_SIZE + cfg.DATASET.TEST_SIZE <= len(image_paths)
    np.random.default_rng(cfg.SEEDS.DATA).shuffle(image_paths)
    image_paths = {
        'train': image_paths[:cfg.DATASET.TRAIN_SIZE],
        'val': image_paths[-cfg.DATASET.VAL_SIZE-cfg.DATASET.TEST_SIZE:-cfg.DATASET.TEST_SIZE],
        'test': image_paths[-cfg.DATASET.TEST_SIZE:],
    }
    subjects = {mode: _build_subjects(image_paths[mode]) for mode in image_paths}
    for mode in ['train', 'val', 'test']: logging.info(f'{mode}: {len(subjects[mode])}')

    transforms = build_transforms()
    datasets = {
        mode: tio.SubjectsDataset(
            subjects[mode],
            transform=transforms['augmentation' if mode == 'train' else 'identity']
        ) for mode in subjects
    }
    dataloaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            shuffle=(mode == 'train'),
        ) for mode in subjects
    }
    return dataloaders

def build_dataloaders_FAST(data_dir):
    def _build_subjects(folders):
        return [tio.Subject(
            image=tio.ScalarImage(f'{folder}/MRI.nii.gz'),
            label=tio.LabelMap(f'{folder}/seg.nii.gz'),
        ) for folder in folders]

    datafolders = {site: sorted(glob(f'{data_dir}/{site}_sub*')) for site in cfg.DATASET.SITES}
    for site_paths in datafolders.values():
        np.random.default_rng(cfg.SEEDS.DATA).shuffle(site_paths)
    
    selected_datafolders = defaultdict(list)
    assert cfg.DATASET.TRAIN_SIZE + cfg.DATASET.VAL_SIZE + cfg.DATASET.TEST_SIZE <= len(datafolders[cfg.DATASET.TRAIN_VAL_SITE])
    selected_datafolders['train'] = datafolders[cfg.DATASET.TRAIN_VAL_SITE][:cfg.DATASET.TRAIN_SIZE]
    selected_datafolders['val'] = datafolders[cfg.DATASET.TRAIN_VAL_SITE][-cfg.DATASET.VAL_SIZE-cfg.DATASET.TEST_SIZE:-cfg.DATASET.TEST_SIZE]
    for site in sorted(cfg.DATASET.SITES): selected_datafolders['test'].extend(datafolders[site][-cfg.DATASET.TEST_SIZE:])

    for mode in ['train', 'val', 'test']: logging.info(f'{mode}: {len(selected_datafolders[mode])}')
    
    subjects = {mode: _build_subjects(selected_datafolders[mode]) for mode in selected_datafolders}
    transforms = build_transforms()
    datasets = {
        mode: tio.SubjectsDataset(
            subjects[mode],
            transform=transforms['augmentation' if mode == 'train' else 'identity']
        ) for mode in subjects
    }
    dataloaders = {
        mode: DataLoader(
            datasets[mode],
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            shuffle=(mode == 'train'),
        ) for mode in subjects
    }
    return dataloaders

if cfg.DATASET.NAME == 'FAST':
    build_dataloaders = build_dataloaders_FAST
if 'MALPEM' in cfg.DATASET.NAME:
    build_dataloaders = build_dataloaders_MALPEM
