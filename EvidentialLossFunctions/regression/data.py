import torch
from config import cfg
from torch.utils.data import Dataset, DataLoader


class CubicRegression(Dataset):
    def __init__(self, mode):
        assert mode in {'train', 'test'}
        self.n = 1000
        if mode == 'train':
            torch.manual_seed(cfg.SEEDS.DATA)
            self.x = torch.linspace(-4, 4, self.n)
            self.y = self.x**3 + torch.normal(torch.zeros(self.n), 3)
        else:
            self.x = torch.linspace(-6, 6, self.n)
            self.y = self.x**3
    def __len__(self):
        return self.n
    def __getitem__(self, index):
        return self.x[index:index+1], self.y[index:index+1]
    
def build_dataloaders():
    dataloaders = {
        mode: DataLoader(
            CubicRegression(mode),
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            shuffle=(mode == 'train'),
            drop_last=(mode == 'train'),
        ) for mode in ['train', 'test']
    }
    return dataloaders
