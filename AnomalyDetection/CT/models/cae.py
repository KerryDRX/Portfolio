import torch.nn as nn
from config import cfg

    
class AutoEncoder(nn.Module):
    def __init__(self, patch=False):
        super(AutoEncoder, self).__init__()

        def encoder_block(in_channels, out_channels, bn=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
            )
        
        def decoder_block(in_filters, out_filters, bn=True):
            return nn.Sequential(
                nn.ConvTranspose2d(in_filters, out_filters, 3, 2, 1, 1),
                nn.BatchNorm2d(out_filters) if bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
            )
        cfg.channels = (32,64,128,256,512)
        dim = (cfg.image_size if not patch else cfg.patch_size) / 2 ** len(cfg.channels)
        self.encoder = nn.Sequential(
            *[
                encoder_block(
                    cfg.channels[i-1] if i > 0 else 1, cfg.channels[i], bn=(i > 0)
                ) for i in range(len(cfg.channels))
            ],

            nn.Flatten(),
            nn.Linear(cfg.channels[-1] * dim * dim, cfg.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.25),
            nn.BatchNorm1d(cfg.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.channels[-1] * dim * dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.25),
            nn.BatchNorm1d(cfg.channels[-1] * dim * dim),
            nn.Unflatten(1, (cfg.channels[-1], dim, dim)),
            
            *[
                decoder_block(
                    cfg.channels[i], cfg.channels[i-1], bn=(i > 1)
                ) for i in range(len(cfg.channels)-1, 0, -1)
            ],
            nn.ConvTranspose2d(cfg.channels[0], 1, 3, 2, 1, 1),
        )
        
    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return code, recon
    