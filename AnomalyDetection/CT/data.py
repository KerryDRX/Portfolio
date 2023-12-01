from config import cfg
from glob import glob
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict


def load_paths(data_dir, good_label, bad_label):
    good_paths = sorted(glob(f'{data_dir}/train/{good_label}/*')) + sorted(glob(f'{data_dir}/test/{good_label}/*'))
    bad_paths = sorted(glob(f'{data_dir}/test/{bad_label}/*'))
    # good_paths = sorted(glob(f'{data_dir}/{good_label}/*'))
    # bad_paths = sorted(glob(f'{data_dir}/{bad_label}/*'))
    return good_paths, bad_paths

def kfold(paths, k):
    kf = KFold(n_splits=k, random_state=cfg.seed, shuffle=True)
    folds = {fold: [paths[index] for index in indices] for fold, (_, indices) in enumerate(kf.split(paths))}
    return folds

def build_paths(good_folds, bad_folds, good_round, bad_round):
    folds = {
        'good': good_folds,
        'bad': bad_folds,
    }
    fold_indices = {
        'good': {
            'train_orig': [good_round%cfg.k, (good_round+1)%cfg.k, (good_round+2)%cfg.k],
            'train': [good_round%cfg.k, (good_round+1)%cfg.k, (good_round+2)%cfg.k],
            'validation': [(good_round+3)%cfg.k],
            'test': [(good_round+4)%cfg.k],
        },
        'bad' : {
            'validation': [bad_round%2],
            'test': [(bad_round+1)%2],
        },
    }
    paths = defaultdict(list)
    for mode in cfg.modes:
        for label in ['good', 'bad']:
            if mode in fold_indices[label]:
                for fold_index in fold_indices[label][mode]:
                    paths[mode].extend(folds[label][fold_index])
    return paths

def build_transforms():
    resize = T.Resize((cfg.image_size, cfg.image_size))
    random_horizontal_flip = T.RandomHorizontalFlip(p=0.5)
    random_affine = T.RandomAffine(
        degrees=15,
        translate=(0.01, 0.01),
        scale=(0.98, 1.02),
        shear=(-5, 5, -5, 5),
    )
    color_jitter = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01)
    to_tensor = T.ToTensor()

    transforms = {
        'original256': T.Compose([
            T.Resize((256, 256)),
            to_tensor,
            T.Lambda(lambda x: x.repeat(3,1,1)),
        ]),
        'original': T.Compose([
            resize,
            to_tensor, 
        ]),
        'augmentation': T.Compose([
            resize,
            random_horizontal_flip,
            random_affine,
            color_jitter,
            to_tensor, 
        ]),
    }
    return transforms

class ImageDataset(Dataset):
    def __init__(self, image_paths, transforms, image_channels=cfg.image_channels):
        self.image_paths = image_paths
        self.transforms = transforms
        self.image_channels = image_channels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = self.transforms(Image.open(image_path).convert('L' if self.image_channels == 1 else 'RGB'))
        image = (image - image.min()) / (image.max() - image.min())
        label = torch.FloatTensor([0 if cfg.good_label in image_path else 1])
        return image, label

class PatchDataset(Dataset):
    def __init__(self, image_paths, transforms):
        self.image_paths = image_paths
        self.transforms = transforms
        self.images = dict()
        
    def __len__(self):
        return len(self.image_paths) * cfg.ppd**2

    def __getitem__(self, index):
        image_index, patch_index = index/(cfg.ppd**2), index%(cfg.ppd**2)
        row_index, col_index = patch_index/cfg.ppd, patch_index%cfg.ppd
        row_index *= cfg.stride
        col_index *= cfg.stride
        image_path = self.image_paths[image_index]
        if image_path not in self.images:
            image = Image.open(image_path).convert('L' if cfg.image_channels == 1 else 'RGB')
            self.images[image_path] = image
        else:
            image = self.images[image_path]
        image = self.transforms(image)
        image = (image - image.min()) / (image.max() - image.min())
        patch = image[:, row_index:row_index+cfg.patch_size, col_index:col_index+cfg.patch_size]
        label = torch.FloatTensor([0 if cfg.good_label in image_path else 1])
        return patch, label
    
def build_dataloaders(good_folds, bad_folds, good_round, bad_round, patch=False):
    DS = ImageDataset if not patch else PatchDataset
    paths = build_paths(good_folds, bad_folds, good_round, bad_round)
    transforms = build_transforms()
    dataloaders = {
        mode: DataLoader(
            DS(
                paths[mode], 
                transforms['augmentation' if mode=='train' else 'original256' if mode=='train_orig' else 'original'],
                # image_channels=cfg.image_channels if mode!='train_orig' else 3,
            ),
            batch_size=cfg.trainer.batch_size if 'train' in mode else 1 if not patch else cfg.ppd**2,
            shuffle=(mode=='train'),
            drop_last=(mode=='train'),
        )
        for mode in cfg.modes
    }
    return dataloaders
