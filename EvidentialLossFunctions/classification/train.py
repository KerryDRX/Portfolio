import os
import json
import torch
import numpy as np
from config import cfg
from tqdm import trange
from models import get_model
from data import get_dataloaders
from collections import defaultdict
from edl.model import ENN
from edl.losses import EDL_Criterion


def train(cfg, evidential=True):
    if not evidential:
        cfg.MODEL.ACTIVATION = 'identity'
        cfg.MODEL.EPS = 0
        cfg.LOSS.FUNCTION = 'baseline-ce'
    output_dir = f'{cfg.PATHS.OUTPUT_DIR}/{cfg.DATASET.NAME}/{cfg.LOSS.FUNCTION}_p{cfg.LOSS.P}'
    os.makedirs(output_dir, exist_ok=True)
    
    dataloaders = get_dataloaders(cfg.DATASET.NAME)
    model = ENN(
        get_model(cfg.DATASET.NAME),
        act=cfg.MODEL.ACTIVATION,
        eps=cfg.MODEL.EPS,
    ).to(cfg.DEVICE)
    criterion = EDL_Criterion(cfg.LOSS.FUNCTION, p=cfg.LOSS.P) if evidential else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAINER.LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAINER.MAX_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()
    results = defaultdict(list)
    
    for epoch in trange(1, cfg.TRAINER.MAX_EPOCHS+1, desc='Training'):
        for mode in ['train', 'test']:
            if mode == 'train': model.train()
            else: model.eval()

            losses = []
            correct, total = 0, 0
            for image, label in dataloaders[mode]:
                image, label = image.to(cfg.DEVICE), label.to(cfg.DEVICE)
                with torch.cuda.amp.autocast():
                    logits = model(image)
                    if evidential:
                        loss, kld = criterion(logits, label)
                        loss = loss + cfg.LOSS.REG_COEF * min(1, 2*epoch/cfg.TRAINER.MAX_EPOCHS) * kld
                    else:
                        loss = criterion(logits, label)
                    pred = logits.argmax(1)
                if mode == 'train':
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                correct += (pred == label).sum().item(); total += label.shape[0]
                losses.append(loss.item())
            if mode == 'train': scheduler.step()

            results[f'{mode}_loss'].append(np.mean(losses))
            results[f'{mode}_accuracy'].append(correct / total)
            with open(f'{output_dir}/results.json', 'w') as file: json.dump(results, file, indent=4)
    
    torch.save(model.cpu().state_dict(), f'{output_dir}/model.pt')
    return results

if __name__ == '__main__':
    for p in [2, 3, 4, 5]:
        cfg.LOSS.P = p
        print(f'p: {p}')
        train(cfg)
