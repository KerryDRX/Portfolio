import os
import numpy as np
from glob import glob
from PIL import Image
from config import cfg
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, mode, image_paths):
        assert mode in {'train', 'val', 'test'}
        self.mode = mode
        self.image_paths = image_paths
        self.label_paths = [image_path.replace('images', 'labels') for image_path in image_paths]
        assert all([os.path.exists(path) for path in np.concatenate([self.image_paths, self.label_paths])])
        self.flip = T.RandomHorizontalFlip(p=1)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image, label = Image.open(self.image_paths[index]), Image.open(self.label_paths[index]).convert('L')
        if self.mode == 'train' and np.random.random() < 0.5:
            image, label = self.flip(image), self.flip(label)
        image, label = self.to_tensor(image), self.to_tensor(label)
        image = (image - image.min()) / (image.max() - image.min())
        label = (label > 0.5).to(float)
        return image, label
    
def build_dataloaders(val_fold):
    image_paths = sorted(glob(f'{cfg.PATHS.DATA}/{cfg.DATASET.NAME}/train/images/*.png'))
    np.random.default_rng(cfg.SEEDS.DATA).shuffle(image_paths)
    image_paths = np.array_split(image_paths, cfg.DATASET.FOLDS)
    image_paths = {
        'train': np.concatenate([image_paths[fold] for fold in range(cfg.DATASET.FOLDS) if fold != val_fold]),
        'val': image_paths[val_fold],
        'test': sorted(glob(f'{cfg.PATHS.DATA}/{cfg.DATASET.NAME}/test/images/*.png')),
    }
    return {mode: DataLoader(
        ImageDataset(mode, image_paths[mode]),
        batch_size=cfg.DATALOADER.BATCH_SIZE,
        pin_memory=cfg.DATALOADER.PIN_MEMORY,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        shuffle=(mode == 'train'),
    ) for mode in image_paths}
    