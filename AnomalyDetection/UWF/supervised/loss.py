import numpy as np
import torch


def weight_calculation(dataloaders):
    '''
    Calculate the weight of cross entropy loss, based on the number of training images.
    If negative : positive = n_neg : n_pos, then the weight is n_pos/n : n_neg/n, in which n = n_neg + n_pos.

    Parameters:
    ----------
        dataloaders: dict
            Dictionary of dataloaders.
    
    Returns:
    ----------
        weight: numpy.ndarray
            Weight of cross entropy loss.
    '''
    train_labels = [label.item() for _, label in dataloaders['train_identity']]
    count = np.unique(train_labels, return_counts=True)[1]
    weight = count[::-1] / count.sum()
    return weight

def loss_fn(weight):
    '''
    Loss function: weighted cross entropy.

    Returns:
    ----------
        criterion: torch.nn.CrossEntropyLoss
            Loss function.
    '''
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(weight).cuda())
    return criterion
