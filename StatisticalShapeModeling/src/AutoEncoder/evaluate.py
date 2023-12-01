import torch
import torchio as tio
import monai
import numpy as np
from tqdm import tqdm
from config import cfg


def evaluate(model, dataloader):
    criterion = monai.losses.DiceLoss(squared_pred=True, sigmoid=True)
    metric = monai.metrics.DiceMetric(include_background=False, reduction='mean_batch')
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for data in dataloader:
            label = data['label'][tio.DATA].to(torch.float16).to(cfg.DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(label)
                loss = criterion(logits, label)
            eval_losses.append(loss.item())
            metric(logits > 0, label)
    eval_loss = np.mean(eval_losses)
    eval_DSC = metric.aggregate().tolist()
    return eval_loss, eval_DSC
