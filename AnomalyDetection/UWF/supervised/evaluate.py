import numpy as np
import torch
import torch.nn.functional as F
from config import cfg
from model import *
from data import *
from utils import *
from visualization import *
from loss import *


def evaluate(model, dataloaders, mode, metrics, save=True, hist=False):
    '''
    Evaluate model on validation/test set and save the results.

    Parameters:
    ----------
        model: torch.nn.Module
            The model to test.
        dataloaders: dict
            Dictionary of dataloaders.
        mode: str
            Evaluation mode ('validation' or 'test').
        metrics: Logger
            Metric logger.
        save: bool
            Save results to metric logger or not.
        hist: bool
            Plot histogram or not.
    
    Returns:
    ----------
        probs: torch.Tensor
            Predicted probabilities of being positive.
        preds: torch.Tensor
            Predicted labels.
    '''
    model.eval()
    criterion = loss_fn(weight_calculation(dataloaders))
    probs, preds, labels, losses = [[] for _ in range(4)]
    with torch.no_grad():
        for image, label in dataloaders[mode]:
            with torch.cuda.amp.autocast():
                logit = model(image.cuda())
                loss = criterion(logit, label[:, 0].cuda())
                prob = F.softmax(logit, dim=1)[:, 1]
                pred = (prob >= 0.5).float()
            probs.append(prob.detach().cpu())
            preds.append(pred.detach().cpu())
            labels.append(label)
            losses.append(loss.item())
    probs = torch.concat(probs).squeeze().numpy()
    preds = torch.concat(preds).squeeze().numpy()
    labels = torch.concat(labels).squeeze().numpy()
    
    auc = metric_functions['auc'](labels, probs)
    if hist:
        plt.hist(probs[labels==0], bins=100, alpha=0.3, label='Good')
        plt.hist(probs[labels==1], bins=100, alpha=0.3, label='Poor')
        plt.xlim((0, 1)); plt.xlabel(f'Predicted Probability of {cfg.DATA.BAD_LABEL}'); plt.ylabel('Count'); plt.legend()
        plt.title(f'{mode.capitalize()} Histogram (AUC={auc:.3f})')
        plt.savefig(f'{cfg.PATHS.OUTPUT_DIR}/{mode}/anomaly_histogram.jpg', dpi=300)
        plt.close('all')

    metrics1 = {
        f'{mode}_{metric_name}': metric_function(labels, preds) if metric_name != 'auc' else auc
        for metric_name, metric_function in metric_functions.items()
    }
    metrics2 = {
        f'{mode}_loss': np.mean(losses),
    }
    if save:
        metrics.log(metrics1)
        metrics.log(metrics2)
    else:
        log(metrics1)
        log(metrics2)
    return probs, preds

def test():
    '''
    Test model performance on the test set and perform GradCAM visualization.
    '''
    dataloaders = build_dataloaders()
    model = build_model(cfg.MODEL).cuda()
    model.load_state_dict(torch.load(f'{cfg.PATHS.OUTPUT_DIR}/models/best.pt'))
    _, preds = evaluate(model, dataloaders, 'test', metrics=None, save=False, hist=True)
    gradcam_visualization(model, dataloaders, preds, 'test')
    