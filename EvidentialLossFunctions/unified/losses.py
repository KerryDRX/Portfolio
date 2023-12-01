import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None): 
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, y):
        log_p = logits.log_softmax(1)
        loss = - (y * log_p)
        if self.gamma > 0:
            loss = loss * (1 - log_p.exp()).pow(self.gamma)
        if self.alpha is not None:
            loss = loss * torch.tensor(
                [1 - self.alpha] + [self.alpha] * (y.shape[1] - 1)
            ).to(loss).view([-1] + [1] * (y.ndim - 2))
        
        # class_weights = 1 / y.clone().detach().sum(tuple(range(2, y.ndim)))
        # infs = torch.isinf(class_weights)
        # class_weights[infs] = 0.0
        # class_weights = class_weights + infs * torch.max(class_weights, dim=1)[0].unsqueeze(dim=1)
        # loss = loss * class_weights.to(loss).view(list(logits.shape[:2]) + [1] * (y.ndim - 2))

        loss = loss.sum(1).mean()
        return loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1, soft=False, smooth=1e-5): 
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.gamma = gamma
        self.soft = soft
        self.smooth = smooth

    def forward(self, logits, y):
        reduce_axes = tuple(range(2, y.ndim))
        p = logits.softmax(1)
        numerator = (y * p).sum(reduce_axes) + self.smooth
        denominator = self.alpha * (p.pow(2) if self.soft else p).sum(reduce_axes) + self.beta * (y.pow(2) if self.soft else y).sum(reduce_axes) + self.smooth
        loss = 1.0 - numerator / denominator
        if self.gamma != 1:
            loss = loss.pow(self.gamma)
        loss = loss.mean()
        return loss

class ComboLoss(nn.Module):
    def __init__(self, loss1=None, loss2=None, lambda1=0.5, lambda2=0.5): 
        super(ComboLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def forward(self, logits, y):
        return self.lambda1 * self.loss1(logits, y) + self.lambda2 * self.loss2(logits, y)

def build_loss(loss_fn):
    loss_fns = {
        'CE': FocalLoss(gamma=0, alpha=None),
        'Focal': FocalLoss(gamma=2, alpha=0.25),
        'Dice': FocalTverskyLoss(alpha=0.5, gamma=1, soft=False),
        'sDice': FocalTverskyLoss(alpha=0.5, gamma=1, soft=True),
        'Tversky': FocalTverskyLoss(alpha=0.3, gamma=1, soft=False),
        'sTversky': FocalTverskyLoss(alpha=0.3, gamma=1, soft=True),
        'FocalTversky': FocalTverskyLoss(alpha=0.3, gamma=0.75, soft=False),
        'FocalsTversky': FocalTverskyLoss(alpha=0.3, gamma=0.75, soft=True),
    }
    if loss_fn in loss_fns: return loss_fns[loss_fn]
    loss1, loss2 = loss_fn.split('-')
    return ComboLoss(loss1=loss_fns[loss1], loss2=loss_fns[loss2])
