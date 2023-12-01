import torch
from torch import nn


class VoxelLoss(nn.Module):
    def __init__(self, loss_fn, gamma=0):
        super(VoxelLoss, self).__init__()
        self.loss_fn = loss_fn
        self.gamma = gamma
    
    def forward(self, alpha, y):
        S = alpha.sum(1, keepdim=True)
        if self.loss_fn == 'NLL':
            loss = y * (S.log() - alpha.log())
        elif self.loss_fn == 'CE':
            loss = y * (S.digamma() - alpha.digamma())
        elif self.loss_fn == 'Focal':
            loss = y * (((S + self.gamma).digamma() - alpha.digamma()) * (
                S.lgamma() + (S - alpha + self.gamma).lgamma() - (S - alpha).lgamma() - (S + self.gamma).lgamma()
            ).exp())
        elif self.loss_fn == 'SOS':
            err = (y - alpha / S) ** 2
            var = alpha * (S - alpha) / (S * S * (S + 1))
            loss = err + var
        loss = loss.sum(1).mean()
        return loss
    
class RegionLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1, soft=False, smooth=1e-5):
        super(RegionLoss, self).__init__()
        self.a = alpha
        self.b = 1 - alpha
        self.gamma = gamma
        self.soft = soft
        self.smooth = smooth
    
    def forward(self, alpha, y):
        S = alpha.sum(1, keepdim=True)
        reduce_axes = list(range(2, y.ndim))
        numerator = (y * (alpha / S)).sum(reduce_axes) + self.smooth
        if self.soft:
            den_a = (alpha / S).pow(2) + alpha * (S - alpha) / (S * S * (S + 1))
            den_b = y.pow(2)
        else:
            den_a = alpha / S
            den_b = y
        denominator = (self.a * den_a + self.b * den_b).sum(reduce_axes) + self.smooth
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
        'NLL': VoxelLoss('NLL'),
        'CE': VoxelLoss('CE'),
        'Focal': VoxelLoss('Focal', gamma=2),
        'SOS': VoxelLoss('SOS'),

        'Dice': RegionLoss(alpha=0.5, gamma=1, soft=False),
        'sDice': RegionLoss(alpha=0.5, gamma=1, soft=True),
        'Tversky': RegionLoss(alpha=0.3, gamma=1, soft=False),
        'sTversky': RegionLoss(alpha=0.3, gamma=1, soft=True),
        'FocalTversky': RegionLoss(alpha=0.3, gamma=0.75, soft=False),
        'FocalsTversky': RegionLoss(alpha=0.3, gamma=0.75, soft=True),
    }
    if loss_fn in loss_fns: return loss_fns[loss_fn]
    loss1, loss2 = loss_fn.split('-')
    return ComboLoss(loss1=loss_fns[loss1], loss2=loss_fns[loss2])

class KLD(nn.Module):
    def __init__(self):
        super(KLD, self).__init__()

    def forward(self, alpha, y):
        alpha = y + (1.0 - y) * alpha
        S = alpha.sum(1)
        ones = torch.ones((1, alpha.shape[1])).to(alpha)
        ones.requires_grad = False
        kld = (
            S.lgamma()
            - ones.sum(1).lgamma()
            - alpha.lgamma().sum(1)
            + ((alpha - 1) * (alpha.digamma() - S.unsqueeze(1).digamma())).sum(1)
        ).mean()
        return kld
    