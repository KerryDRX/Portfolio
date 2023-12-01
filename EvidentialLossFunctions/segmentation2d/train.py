import os
import json
import monai
import torch
import numpy as np
from config import cfg
from losses import loss_fns
from tqdm import tqdm, trange
from data import build_dataloaders
from collections import defaultdict
from lightning_fabric import seed_everything


def test(model, dataloader):
    model.eval()
    metric = monai.metrics.DiceMetric(include_background=False, reduction='mean_batch')
    with torch.no_grad():
        for image, label in dataloader:
            with torch.cuda.amp.autocast():
                metric((model(image.to(cfg.DEVICE)).detach().cpu() > 0).to(float), label)
    return metric.aggregate().item()

def train(loss_fn, val_fold, model_seed):
    output_dir = f'{cfg.PATHS.OUTPUT}/{cfg.DATASET.NAME}/{loss_fn}/val_fold{val_fold}/seed{model_seed}'
    os.makedirs(output_dir, exist_ok=True)

    dataloaders = build_dataloaders(val_fold)
    seed_everything(model_seed)
    model = monai.networks.nets.UNet(
        spatial_dims=2, in_channels=cfg.DATASET.NUM_CHANNELS, out_channels=1,
        channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    ).to(cfg.DEVICE)
    # criterion = loss_fns[loss_fn]
    # criterion = monai.losses.DiceCELoss(sigmoid=True, squared_pred=False, ce_weight=torch.tensor([20926/327680, 306754/327680], device=cfg.DEVICE), lambda_dice=0.0, lambda_ce=1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([306754/20926], device=cfg.DEVICE))
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAINER.LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAINER.NUM_EPOCHS)
    metric = monai.metrics.DiceMetric(include_background=False, reduction='mean_batch')
    scaler = torch.cuda.amp.GradScaler()
    results = defaultdict(list)
    best_val_DSC = 0
    for _ in trange(1, 1+cfg.TRAINER.NUM_EPOCHS, desc=f'{loss_fn} val{val_fold}'):
        for mode in ['train', 'val']:
            if mode == 'train': model.train()
            else: model.eval()

            losses = []
            for image, label in dataloaders[mode]:
                with torch.cuda.amp.autocast():
                    logits = model(image.to(cfg.DEVICE))
                    loss = criterion(logits, label.to(cfg.DEVICE))
                if mode == 'train':
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                losses.append(loss.item())
                metric((logits.detach().cpu() > 0).to(float), label)

            if mode == 'train': scheduler.step()
            loss = np.mean(losses)
            DSC = metric.aggregate().item()
            metric.reset()
            results[f'{mode}_loss'].append(loss)
            results[f'{mode}_DSC'].append(DSC)
            with open(f'{output_dir}/results.json', 'w') as file: json.dump(results, file, indent=4)
        
        if DSC >= best_val_DSC:
            best_val_DSC = DSC
            torch.save(model.state_dict(), f'{output_dir}/best.pt')
            
    model.load_state_dict(torch.load(f'{output_dir}/best.pt'))
    results['test_DSC'].append(test(model, dataloaders['test']))
    with open(f'{output_dir}/results.json', 'w') as file: json.dump(results, file, indent=4)
    os.remove(f'{output_dir}/best.pt')

if __name__ == '__main__':
    for val_fold in range(5):
        for model_seed in range(10):
            train('WCE', val_fold, model_seed)
    # for loss_fn in loss_fns:
    #     for val_fold in range(5):
    #         for model_seed in range(10):
    #             train(loss_fn, val_fold, model_seed)
