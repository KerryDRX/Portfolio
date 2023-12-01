import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.linalg.cholesky(a, upper=False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_tensors
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - torch.autograd.Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

def to_var(x, volatile=False):
    return torch.autograd.Variable(x.cuda(), volatile=volatile)

class DAGMM(torch.nn.Module):
    def __init__(self):
        super(DAGMM, self).__init__()

        def encoder_block(in_channels, out_channels, bn=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
        def decoder_block(in_filters, out_filters, bn=True):
            return nn.Sequential(
                nn.ConvTranspose2d(in_filters, out_filters, 3, 2, 1, 1),
                nn.BatchNorm2d(out_filters) if bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
        dim = cfg.image_size / 2 ** len(cfg.channels)
        
        self.encoder = torch.nn.Sequential(
            *[
                encoder_block(
                    cfg.channels[i-1] if i > 0 else 1, cfg.channels[i], bn=True
                ) for i in range(len(cfg.channels))
            ],

            nn.Flatten(),
            nn.Linear(cfg.channels[-1] * dim * dim, cfg.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(cfg.latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.channels[-1] * dim * dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(cfg.channels[-1] * dim * dim),
            nn.Unflatten(1, (cfg.channels[-1], dim, dim)),
            
            *[
                decoder_block(
                    cfg.channels[i], cfg.channels[i-1], bn=True
                ) for i in range(len(cfg.channels)-1, 0, -1)
            ],
            nn.ConvTranspose2d(cfg.channels[0], 1, 3, 2, 1, 1),
        )
        self.reducer = torch.nn.Linear(cfg.latent_dim, cfg.reduced_dim)
        self.estimation = torch.nn.Sequential(
            nn.Linear(cfg.reduced_dim + 2, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(32, cfg.n_gmm),
            nn.Softmax(dim=1),
        )
        self.register_buffer('phi', torch.zeros(cfg.n_gmm))
        self.register_buffer('mu', torch.zeros(cfg.n_gmm, cfg.latent_dim))
        self.register_buffer('cov', torch.zeros(cfg.n_gmm, cfg.latent_dim, cfg.latent_dim))

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)

        x = x.flatten(start_dim=1)
        enc = self.reducer(enc)
        dec = dec.flatten(start_dim=1)

        rec_cosine = F.cosine_similarity(x, dec, dim=1)
        rec_euclidean = (x-dec).norm(2, dim=1) / x.norm(2, dim=1)

        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)

        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=0)  
        phi = sum_gamma / N
        self.phi = phi.data
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            noise = torch.diag(1e-3 + torch.randn(D)*1e-4)
            #noise = torch.eye(D)
            cov_k = cov[i] + to_var(noise)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            #det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            #det_cov.append((torch.cholesky(cov_k * (2*np.pi), False).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).cuda()
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))
        
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        x = x.flatten(start_dim=1)
        recon_error = ((x - x_hat) ** 2).mean()
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag
    