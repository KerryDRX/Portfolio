from config import cfg
from transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, SVHN


Datasets = {
    'MNIST': MNIST,
    'CIFAR10': CIFAR10,
    'SVHN': SVHN,
}

def get_dataloaders(dataset_name):
    return {
        mode: DataLoader(
            Datasets[dataset_name](
                root=f'{cfg.PATHS.DATA_DIR}/{dataset_name}',
                transform=transforms[dataset_name]['augmentation' if mode == 'train' else 'identity'],
                download=True,
                **({'train': mode == 'train'} if dataset_name in {'CIFAR10'} else {'split': mode}),
            ),
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
            shuffle=(mode == 'train'),
        ) for mode in ['train', 'test']
    }
