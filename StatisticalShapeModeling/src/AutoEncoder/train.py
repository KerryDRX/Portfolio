from datetime import datetime
from tqdm import trange
import torch
import torchio as tio
from config import cfg
from evaluate import evaluate
from data import build_datasets_dataloaders
from model import build_model
from utils import *
import numpy as np
import monai
import logging, sys


def train_autoencoder():
    datasets, dataloaders = build_datasets_dataloaders()

    model = build_model('autoencoder').to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAINER.LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAINER.MAX_EPOCHS)
    criterion = monai.losses.DiceLoss(squared_pred=True, sigmoid=True)
    metric = monai.metrics.DiceMetric(include_background=False, reduction='mean_batch')
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_meanDSC = 0
    for epoch in trange(1, 1+cfg.TRAINER.MAX_EPOCHS, desc='Training'):
        model.train()
        train_losses = []
        for data in dataloaders['train']:
            label = data['label'][tio.DATA].to(torch.float16).to(cfg.DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(label)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            metric(logits > 0, label)
            
        train_loss = np.mean(train_losses)
        train_DSC = metric.aggregate().tolist()
        metric.reset()
        logging.info(f'Epoch {epoch}/{cfg.TRAINER.MAX_EPOCHS} train_loss={train_loss:.3f} train_DSC={print_DSC(train_DSC)}')
        # scheduler.step()
        
        if epoch % cfg.TRAINER.EVAL_INTERVAL == 0:
            val_loss, val_DSC = evaluate(model, dataloaders['validation'])
            logging.info(f'Epoch {epoch}/{cfg.TRAINER.MAX_EPOCHS} val_loss={val_loss:.3f} val_DSC={print_DSC(val_DSC)}')
            val_meanDSC = np.mean(val_DSC)
            if val_meanDSC >= best_val_meanDSC:
                best_val_meanDSC = val_meanDSC
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                }
                torch.save(checkpoint, f'{cfg.PATHS.OUTPUT_DIR}/autoencoder3.pt')
                logging.info(f'Best model saved.')

    checkpoint = torch.load(f'{cfg.PATHS.OUTPUT_DIR}/autoencoder.pt')
    model.load_state_dict(checkpoint['model'])
    test_loss, test_DSC = evaluate(model, dataloaders['test'])
    logging.info(f'Test results: test_loss={test_loss:.3f} test_DSC={print_DSC(test_DSC)}')

def main():
    mkdir(cfg.PATHS.OUTPUT_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=f'{cfg.PATHS.OUTPUT_DIR}/output.log'),
            # logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f'{"="*100}\n{cfg}\n{"="*100}')
    t0 = datetime.now()
    train_autoencoder()
    logging.info(f'Time: {datetime.now()-t0}\n')

if __name__ == '__main__':
    main()
