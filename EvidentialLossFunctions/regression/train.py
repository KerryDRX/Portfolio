import os
import torch
import numpy as np
from config import cfg
from tqdm import trange
from utils import Logger
from models import build_model
from data import build_dataloaders
from edl.losses import EDL_Criterion


def train():
    dataloaders = build_dataloaders()
    model = build_model().to(cfg.DEVICE)
    criterion = EDL_Criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAINER.LR)
    scaler = torch.cuda.amp.GradScaler()
    logger = Logger()
    best_loss = np.Inf
    for epoch in trange(1, cfg.TRAINER.MAX_EPOCHS+1, desc='Training'):
        for mode in ['train', 'test']:
            if mode == 'train': model.train()
            elif epoch % cfg.TRAINER.EVAL_INTERVAL == 0: model.eval()
            else: continue

            for x, y in dataloaders[mode]:
                x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
                with torch.cuda.amp.autocast():
                    out = model(x)
                    loss0, reg = criterion(out, y)
                    loss = loss0 + cfg.LOSS.REG_COEF * reg
                if mode == 'train':
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                with torch.no_grad():
                    mse = ((y - out[2])**2).mean()
                logger.log(f'{mode}_loss_epoch_{epoch}', loss.item())
                logger.log(f'{mode}_loss0_epoch_{epoch}', loss0.item())
                logger.log(f'{mode}_reg_epoch_{epoch}', reg.item())
                logger.log(f'{mode}_mse_epoch_{epoch}', mse.item())
            logger.average(f'{mode}_loss_epoch_{epoch}')
            logger.average(f'{mode}_loss0_epoch_{epoch}')
            logger.average(f'{mode}_reg_epoch_{epoch}')
            logger.average(f'{mode}_mse_epoch_{epoch}')
            logger.save(f'{cfg.PATHS.OUTPUT_DIR}/results.json')

            if mode == 'test':
                test_loss = logger.logger[f'{mode}_loss_epoch_{epoch}']
                if test_loss <= best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), f'{cfg.PATHS.OUTPUT_DIR}/best.pt')

    torch.save(model.state_dict(), f'{cfg.PATHS.OUTPUT_DIR}/final.pt')

if __name__ == '__main__':
    os.makedirs(cfg.PATHS.OUTPUT_DIR, exist_ok=True)
    train()
