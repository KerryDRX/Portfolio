import os
import json
import monai
import torch
import numpy as np
import torchio as tio
from losses import build_loss
from model import build_model
from tqdm import tqdm, trange
from data import build_dataloaders
from collections import defaultdict
from monai.networks.utils import one_hot


def evaluate(model, dataset, criterion, device):
    model.eval()
    metric = monai.metrics.DiceMetric(include_background=False, reduction='mean_batch')
    losses = []
    for subject in dataset:
        sampler = tio.inference.GridSampler(subject, patch_size=(80, 160, 160), patch_overlap=(40, 80, 80))
        dataloader = torch.utils.data.DataLoader(sampler, batch_size=4)
        aggregator = tio.inference.GridAggregator(sampler)
        with torch.no_grad():
            for patch in dataloader:
                with torch.cuda.amp.autocast():
                    logits = model(patch['image'][tio.DATA].to(device))
                    loss = criterion(logits, patch['label'][tio.DATA].to(device))
                losses.append(loss.item())
                aggregator.add_batch(logits.argmax(dim=1, keepdim=True).to(torch.int32), patch[tio.LOCATION].to(torch.int32))
            pred = aggregator.get_output_tensor()
            label = subject['label'][tio.DATA]
            metric(one_hot(pred, num_classes=3, dim=0).unsqueeze(0), label.unsqueeze(0))
    return metric.aggregate().tolist(), np.mean(losses)

def train(datasets, train_dataloader, loss_fn, val_fold):
    device = 'cuda:0'
    output_dir = f'outputs/{loss_fn}/val_fold{val_fold}'
    os.makedirs(output_dir, exist_ok=True)

    model = build_model(act='exp', eps=0).to(device)
    criterion = build_loss(loss_fn)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    scaler = torch.cuda.amp.GradScaler()
    patience = 20

    if os.path.exists(f'{output_dir}/best.pt'):
        checkpoint = torch.load(f'{output_dir}/best.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        with open(f'{output_dir}/logs.json', 'r') as file: logger = json.load(file)
        logger['test_DSC'], logger['test_loss'] = [], []
        best_val_loss = min(logger['val_loss'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        logger = defaultdict(list)
        best_val_loss = np.Inf
        start_epoch = 1

    for epoch in trange(start_epoch, 1001, desc='Training'):
        model.train()
        losses = []
        for subject in train_dataloader:
            image, label = subject['image'][tio.DATA].to(device), subject['label'][tio.DATA].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(image)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        logger[f'train_loss'].append(np.mean(losses))
        val_DSC, val_loss = evaluate(model, datasets['val'], criterion, device)
        logger[f'val_DSC'].append(val_DSC)
        logger[f'val_loss'].append(val_loss)
        with open(f'{output_dir}/logs.json', 'w') as file: json.dump(logger, file, indent=4)
        scheduler.step(val_loss)
        if val_loss <= best_val_loss:
            patience = 20
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, f'{output_dir}/best.pt')
        else:
            patience -= 1
            if patience < 0: break

    model.load_state_dict(torch.load(f'{output_dir}/best.pt')['model'])
    test_DSC, test_loss = evaluate(model, datasets['test'], criterion, device)
    logger[f'test_DSC'].append(test_DSC)
    logger[f'test_loss'].append(test_loss)
    with open(f'{output_dir}/logs.json', 'w') as file: json.dump(logger, file, indent=4)

if __name__ == '__main__':
    datasets, train_dataloader = build_dataloaders()
    for loss_fn in ['sDice-SOS']:
        print(loss_fn)
        train(datasets, train_dataloader, loss_fn=loss_fn, val_fold=0)
