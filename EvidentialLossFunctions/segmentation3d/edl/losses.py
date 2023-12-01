import torch


class EDL_Criterion(torch.nn.Module):
    def __init__(self, loss_fn, **kwargs):
        super(EDL_Criterion, self).__init__()
        self.loss_fn = loss_fn
        self.kwargs = kwargs
    
    def forward(self, alpha, y):
        loss = self._cal_loss(alpha, y)
        kld = self._cal_kld(y + (1.0 - y) * alpha)
        return loss, kld

    def _cal_loss(self, alpha, y):
        S = alpha.sum(1, keepdim=True)
        if self.loss_fn == 'nll':
            loss = y * (S.log() - alpha.log())
            loss = loss.sum(1).mean()
        elif self.loss_fn == 'ce':
            loss = y * (S.digamma() - alpha.digamma())
            loss = loss.sum(1).mean()
        elif self.loss_fn == 'focal':
            gamma = self.kwargs['gamma']
            loss = y * ((S + gamma).digamma() - alpha.digamma()) * (
                S.lgamma() + (S - alpha + gamma).lgamma() - (S - alpha).lgamma() - (S + gamma).lgamma()
            ).exp()
            loss = loss.sum(1).mean()
        elif self.loss_fn == 'ss':
            err = (y - alpha / S) ** 2
            var = alpha * (S - alpha) / (S * S * (S + 1))
            loss = err + var
            loss = loss.sum(1).mean()
        elif self.loss_fn == 'lpnorm':
            p = self.kwargs['p']
            nongt_alpha = (1.0 - y) * alpha
            x1 = (alpha.sum(1).lgamma() - (p + alpha.sum(1)).lgamma()).exp()
            x2 = ((p + nongt_alpha.sum(1)).lgamma() - nongt_alpha.sum(1).lgamma()).exp()
            x3 = ((1.0 - y) * ((p + alpha).lgamma() - alpha.lgamma()).exp()).sum(1)
            loss = (x1 * (x2 + x3)) * (1/p)
            loss = loss.mean()
        elif self.loss_fn == 'sdice':
            reduce_axes = list(range(2, y.dim()))
            K = y.shape[1]
            num = (y * alpha / S).sum(reduce_axes)
            den = (y ** 2 + (alpha / S) ** 2 + alpha * (S - alpha) / (S * S * (S + 1))).sum(reduce_axes)
            loss = (1 - 2/K * (num/den).sum(1)).mean()
        elif self.loss_fn == 'dice':
            reduce_axes = list(range(2, y.dim()))
            K = y.shape[1]
            num = (y * alpha / S).sum(reduce_axes)
            den = (y + alpha / S).sum(reduce_axes)
            loss = (1 - 2/K * (num/den).sum(1)).mean()
        elif self.loss_fn == 'tversky':
            a, b = self.kwargs['alpha'], self.kwargs['beta']
            reduce_axes = list(range(2, y.dim()))
            K = y.shape[1]
            num = (y * alpha / S).sum(reduce_axes)
            den = ((1-a-b) * y * alpha / S + a * alpha / S + b * y).sum(reduce_axes)
            loss = (1 - 1/K * (num/den).sum(1)).mean()
        elif self.loss_fn == 'focal_tversky':
            a, b, gamma = self.kwargs['alpha'], self.kwargs['beta'], self.kwargs['gamma']
            reduce_axes = list(range(2, y.dim()))
            K = y.shape[1]
            num = (y * alpha / S).sum(reduce_axes)
            den = ((1-a-b) * y * alpha / S + a * alpha / S + b * y).sum(reduce_axes)
            loss = (1/K * ((1 - num/den) ** (1 / gamma)).sum(1)).mean()
        else:
            raise NotImplementedError
        return loss
    
    def _cal_kld(self, alpha):
        S = alpha.sum(1)
        K = alpha.shape[1]
        ones = torch.ones((1, K)).to(alpha.device)
        ones.requires_grad = False
        kld = (
            S.lgamma()
            - ones.sum(1).lgamma()
            - alpha.lgamma().sum(1)
            + ((alpha - 1) * (alpha.digamma() - S.unsqueeze(1).digamma())).sum(1)
        ).mean()
        return kld
    