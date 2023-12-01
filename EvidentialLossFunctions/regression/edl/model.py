import torch
from torch import nn
from torch.nn import functional as F


class DenseNormalGamma(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, 4*out_features)
        
    def _evidence(self, x):
        return F.softplus(x)
    
    def forward(self, x):
        output = self.linear(x)
        log_alpha, log_beta, gamma, log_nu = torch.split(output, self.out_features, -1)
        alpha = self._evidence(log_alpha) + 1
        beta = self._evidence(log_beta)
        nu = self._evidence(log_nu)
        return alpha, beta, gamma, nu

def uncertainty(out):
    alpha, beta, _, nu = out
    AU = beta / (alpha - 1)
    EU = AU / nu
    return AU, EU
