import torch
import scipy, math
import numpy as np


class EDL_Criterion(torch.nn.Module):
    def __init__(self):
        super(EDL_Criterion, self).__init__()
    
    def forward(self, out, y):
        loss0 = self._br(out, y)
        reg = self._reg(out, y)
        return loss0, reg
    
    def _nll(self, out, y):
        alpha, beta, gamma, nu = out
        omega = 2 * beta * (1 + nu)
        return (
            (torch.pi / nu).log() / 2
            - alpha * omega.log()
            + (alpha + 0.5) * ((y - gamma) ** 2 * nu + omega).log()
            + alpha.lgamma()
            - (alpha + 0.5).lgamma()
        ).mean()
    
    def _reg(self, out, y):
        alpha, _, gamma, nu = out
        return ((y - gamma).abs() * (2 * nu + alpha)).mean()

    def _br(self, out, y, p=2):
        alpha, beta, gamma, nu = out
        t1 = scipy.special.gamma(p/2 + 0.5) / math.pi**0.5 / alpha.lgamma().exp()
        t2 = (2 * beta / nu) ** (p/2)
        t3 = 0
        for n in range(2):
            coef = np.prod([(m - p/2) for m in range(n)])
            if coef == 0: break
            t31 = (alpha + n - p/2).lgamma().exp() / math.factorial(2 * n)
            t32 = (-2 * (y - gamma) ** 2 * nu / beta) ** n
            t3 += coef * t31 * t32
        return (t1 * t2 * t3).mean()


