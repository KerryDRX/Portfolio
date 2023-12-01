import os
import json
import monai
import torch
import numpy as np
from utils import *
import torchio as tio
from config import cfg
from tqdm import tqdm, trange
from data import build_dataloaders
from collections import defaultdict
from edl.model import ENN
from edl.losses import EDL_Criterion


def test(model, dataloader):
    metric = monai.metrics.DiceMetric(include_background=False, reduction='mean_batch', num_classes=cfg.DATASET.NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Testing'):
            image, label = data['image'][tio.DATA].to(cfg.DEVICE), data['label'][tio.DATA].to(cfg.DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(image)
                pred = logits.argmax(1, keepdim=True)
            metric(pred, label)
    DSCs = metric.aggregate().detach().cpu().tolist()
    return DSCs

def train(loss_fn, criterion):
    output_dir = f'{cfg.PATHS.OUTPUT_DIR}/{loss_fn}'
    os.makedirs(output_dir, exist_ok=True)

    dataloaders = build_dataloaders(cfg.PATHS.DATA_DIR)
    model = monai.networks.nets.UNet(
        spatial_dims=3, in_channels=1, out_channels=cfg.DATASET.NUM_CLASSES,
        channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    ).to(cfg.DEVICE)
    # model = ENN(
    #     model=monai.networks.nets.UNet(
    #         spatial_dims=3, in_channels=1, out_channels=cfg.DATASET.NUM_CLASSES,
    #         channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
    #     ),
    #     act=cfg.MODEL.ACTIVATION,
    # ).to(cfg.DEVICE)
    # criterion = EDL_Criterion(loss_fn, alpha=cfg.LOSS.ALPHA, beta=cfg.LOSS.BETA, gamma=cfg.LOSS.GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAINER.LR, weight_decay=5e-4)
    metric = monai.metrics.DiceMetric(include_background=False, reduction='mean_batch', num_classes=cfg.DATASET.NUM_CLASSES)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = np.Inf
    results = defaultdict(list)
    pbar = trange(1, 1+cfg.TRAINER.MAX_EPOCHS, desc='Training')
    for epoch in pbar:
        for mode in ['train', 'val']:
            if mode == 'train': model.train()
            else: model.eval()

            losses = []
            for data in dataloaders[mode]:
                image, label = data['image'][tio.DATA].to(cfg.DEVICE), data['label'][tio.DATA].to(cfg.DEVICE)
                with torch.cuda.amp.autocast():
                    logits = model(image)
                    loss = criterion(logits, label)
                    pred = logits.argmax(1, keepdim=True)
                    # alpha = model(image)
                    # loss, kld = criterion(alpha, label)
                    # loss = loss + cfg.LOSS.REG_COEF * min(1, 2*epoch/cfg.TRAINER.MAX_EPOCHS) * kld
                    # pred = alpha.argmax(1, keepdim=True)
                if mode == 'train':
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                losses.append(loss.item()); pbar.set_postfix({f'{mode}_loss': loss.item()})
                metric(pred, label)
            loss = np.mean(losses)
            DSCs = metric.aggregate().detach().cpu().tolist()
            metric.reset()
            results[f'{mode}_loss'].append(loss)
            results[f'{mode}_DSCs'].append(DSCs)
            with open(f'{output_dir}/results.json', 'w') as file: json.dump(results, file, indent=4)

            if mode == 'val' and loss <= best_val_loss:
                best_val_loss = loss
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, f'{output_dir}/best.pt')

    model.load_state_dict(torch.load(f'{output_dir}/best.pt')['model'])
    DSCs = test(model, dataloaders['test'])
    results['test_DSCs'] = DSCs
    with open(f'{output_dir}/results.json', 'w') as file: json.dump(results, file, indent=4)

if __name__ == '__main__':
    for loss_fn, criterion in [
        ['softmax-sdice', monai.losses.DiceCELoss(softmax=True, squared_pred=True, lambda_dice=1.0, lambda_ce=0.0)],
        ['softmax-dice', monai.losses.DiceCELoss(softmax=True, squared_pred=False, lambda_dice=1.0, lambda_ce=0.0)],
        ['softmax-ce', monai.losses.DiceCELoss(softmax=True, squared_pred=False, lambda_dice=0.0, lambda_ce=1.0)],
        # ['softmax-sdice-ce', monai.losses.DiceCELoss(softmax=True, squared_pred=True, lambda_dice=1.0, lambda_ce=1.0)],
        # ['softmax-dice-ce', monai.losses.DiceCELoss(softmax=True, squared_pred=False, lambda_dice=1.0, lambda_ce=1.0)],
        ['softmax-focal', monai.losses.FocalLoss(gamma=2.0)],
        ['softmax-tversky', monai.losses.TverskyLoss(softmax=True, alpha=0.3, beta=0.7)],
    ]:
        train(loss_fn, criterion)
