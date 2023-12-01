from transforms import build_transforms
from torch.utils.data import DataLoader
import numpy as np
from config import cfg
import torchio as tio
from glob import glob
import logging


def build_subjects():
    image_paths = sorted(glob(f'{cfg.PATHS.DATA_DIR}/MRI/*.nii.gz'))
    label_paths = sorted(glob(f'{cfg.PATHS.DATA_DIR}/Seg/*.nii.gz'))
    subjects = [
        tio.Subject(
            # image=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path),
        ) for image_path, label_path in zip(image_paths, label_paths)
    ]
    np.random.default_rng(cfg.SEED.DATA).shuffle(subjects)
    
    assert sum(cfg.DATASET.TRAIN_VAL_TEST_RATIO) == 1
    train_ratio, val_ratio, test_ratio = cfg.DATASET.TRAIN_VAL_TEST_RATIO
    train_size = int(train_ratio * len(subjects))
    test_size = int(test_ratio *len(subjects))
    subjects = {
        'train': subjects[:train_size],
        'validation': subjects[train_size:-test_size],
        'test': subjects[-test_size:],
    }
    return subjects

def build_datasets_dataloaders():
    subjects = build_subjects()
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
            batch_size=cfg.DATALOADER.BATCH_SIZE_TRAIN if mode == 'train' else cfg.DATALOADER.BATCH_SIZE_EVAL,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            shuffle=(mode == 'train'),
            drop_last=(mode == 'train'),
        ) for mode in subjects
    }
    logging.info(f'Train/Validation/Test: {"/".join([str(len(datasets[mode])) for mode in datasets])}')
    return datasets, dataloaders
