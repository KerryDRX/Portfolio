import torch
import numpy as np
from tqdm import tqdm
from config import cfg
from itertools import chain
from data import get_dataloaders
from collections import defaultdict
from torcheval.metrics import functional as F
from torchmetrics.functional.classification import multiclass_calibration_error


def evaluate(model, base_dataset_name, ood_dataset_name=None):
    base_dataloader = get_dataloaders(base_dataset_name)['test']
    ood_dataloader = get_dataloaders(ood_dataset_name)['test'] if ood_dataset_name else None
    dataloader = chain(base_dataloader, ood_dataloader) if ood_dataset_name else base_dataloader
    base_num = len(base_dataloader.dataset)
    
    model.eval()
    results = defaultdict(list)
    with torch.no_grad():
        for image, label in tqdm(dataloader):
            results['label'].append(label)
            with torch.cuda.amp.autocast():
                for k, v in model.uncertainty(image.to(cfg.DEVICE)).items(): results[k].append(v.detach().cpu())
        for k, v in results.items(): results[k] = torch.cat(v)
    
    misclassified = (results['label'][:base_num] != results['pred'][:base_num]).to(int)
    if ood_dataset_name:
        ood_num = len(ood_dataloader.dataset)
        ood = torch.cat([torch.zeros(base_num), torch.ones(ood_num)])
    
    metrics = dict()
    for k, v in results.items():
        if k in {'p', 'label', 'pred'}: continue
        if k in {'margin_of_confidence', 'ratio_of_confidence'}: v = -v
        metrics[k] = {
            'misc_AUROC': F.binary_auroc(v[:base_num], misclassified).item(),
            'misc_AUPRC': F.binary_auprc(v[:base_num], misclassified).item(),
        }
        if ood_dataset_name:
            metrics[k].update({
                'ood_AUROC': F.binary_auroc(v, ood).item(),
                'ood_AUPRC': F.binary_auprc(v, ood).item(),
            })
    metrics['classification_accuracy'] = 1 - (misclassified.sum() / base_num).item()
    metrics['ECE'] = multiclass_calibration_error(
        preds=results['p'][:base_num], target=results['label'][:base_num], num_classes=cfg.DATASET.NUM_CLASSES, n_bins=15, norm='l1'
    )
    return metrics
