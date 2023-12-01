import numpy as np
import torchio as tio
from tqdm import tqdm
from collections import defaultdict
from transforms import build_transforms
from torch.utils.data import DataLoader


def build_dataloaders(val_fold=0):
    def _load_subject(pid):
        # data_folder = f'/projects/seafan-artifacts/kits19/data/case_{pid:05}'
        data_folder = f'C:/Users/kerry/OneDrive/Desktop/Projects/Datasets/KiTS19/case_{pid:05}'
        return tio.Subject(
            image=tio.ScalarImage(f'{data_folder}/imaging.nii.gz'),
            label=tio.LabelMap(f'{data_folder}/segmentation.nii.gz'),
            pid=pid,
        )
    
    def _load_pids(val_fold):
        pids = [pid for pid in range(210) if pid not in {15, 23, 37, 68, 125, 133}]
        np.random.default_rng(0).shuffle(pids)
        test_size = round(len(pids) * 0.2)
        train_val_pids = np.array_split(pids[:-test_size], 5)
        return {
            'train': np.concatenate([train_val_pids[fold] for fold in range(5) if fold != val_fold]),
            'val': train_val_pids[val_fold],
            'test': pids[-test_size:],
        }
    
    modes = ['train', 'val', 'test']
    pids = _load_pids(val_fold)
    transforms = build_transforms()
    subjects = defaultdict(list)
    for mode in modes:
        for pid in tqdm(pids[mode], desc=f'Load {mode}'):
            subjects[mode].append(transforms['resample'](_load_subject(pid)))
    datasets = {
        mode: tio.SubjectsDataset(
            subjects[mode],
            transform=transforms['augmentation' if mode == 'train' else 'identity']
        ) for mode in modes
    }
    train_dataloader = DataLoader(
        tio.Queue(
            datasets['train'],
            max_length=300,
            samples_per_volume=1,
            sampler=tio.data.UniformSampler((80, 160, 160)),
            num_workers=4,
        ),
        batch_size=2,
        pin_memory=False,
        num_workers=0,
    )
    return datasets, train_dataloader
