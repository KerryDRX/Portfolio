import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from config import cfg


class AutoEncoder(nn.Module):
    def __init__(self, channels, image_dim, latent_dim=None):
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
        
        dim = image_dim / 2 ** len(channels)
        self.encoder = nn.Sequential(*[
            *[
                encoder_block(
                    (cfg.image_channels if i==0 else channels[i-1]), channels[i], bn=(i > 0)
                ) for i in range(len(channels))
            ],
            # *[
            #     nn.Flatten(),
            #     nn.Linear(channels[-1] * dim * dim, latent_dim),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.Dropout1d(0.25),
            #     nn.BatchNorm1d(latent_dim),
            # ],
        ])
        self.decoder = nn.Sequential(*[
            # *[
            #     nn.Linear(latent_dim, channels[-1] * dim * dim),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     nn.Dropout1d(0.25),
            #     nn.BatchNorm1d(channels[-1] * dim * dim),
            #     nn.Unflatten(1, (channels[-1], dim, dim)),
            # ],
            *[
                decoder_block(
                    channels[i], channels[i-1], bn=(i > 1)
                ) for i in range(len(channels)-1, 0, -1)
            ],
            nn.ConvTranspose2d(channels[0], 1, 3, 2, 1, 1),
        ])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ScaleSpaceAutoEncoder(nn.Module):
    def __init__(self):
        super(ScaleSpaceAutoEncoder, self).__init__()
        self.autoencoder0 = AutoEncoder(channels=(32,64,128,256,512,), image_dim=cfg.image_size, latent_dim=None)
        self.autoencoder1 = AutoEncoder(channels=(32,64,128,256,), image_dim=cfg.image_size/2, latent_dim=None)
        self.autoencoder2 = AutoEncoder(channels=(32,64,128,), image_dim=cfg.image_size/4, latent_dim=None)
        self.autoencoders = [self.autoencoder0, self.autoencoder1, self.autoencoder2]
        
    def forward(self, image0):
        align_corners = False
        down_image1 = gaussian_blur(image0, kernel_size=5)
        down_image1 = F.interpolate(down_image1, scale_factor=0.5, mode='bilinear', align_corners=align_corners)
        down_image2 = gaussian_blur(down_image1, kernel_size=5)
        down_image2 = F.interpolate(down_image2, scale_factor=0.5, mode='bilinear', align_corners=align_corners)
        down_image3 = gaussian_blur(down_image2, kernel_size=5)
        down_image3 = F.interpolate(down_image3, scale_factor=0.5, mode='bilinear', align_corners=align_corners)
        
        up_image0 = F.interpolate(down_image1, scale_factor=2, mode='bilinear', align_corners=align_corners)
        up_image1 = F.interpolate(down_image2, scale_factor=2, mode='bilinear', align_corners=align_corners)
        up_image2 = F.interpolate(down_image3, scale_factor=2, mode='bilinear', align_corners=align_corners)
        
        diff2 = down_image2 - up_image2
        recon2 = self.autoencoder2(diff2) + up_image2
        diff1 = down_image1 - up_image1
        recon1 = self.autoencoder1(diff1) + F.interpolate(recon2, scale_factor=2, mode='bilinear', align_corners=align_corners)
        diff0 = image0 - up_image0
        recon0 = self.autoencoder0(diff0) + F.interpolate(recon1, scale_factor=2, mode='bilinear', align_corners=align_corners)
        
        return recon0, recon1, recon2, down_image1, down_image2
    