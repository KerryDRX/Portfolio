import torch.nn as nn
from config import cfg
import torch.nn.init as init
import math
import numpy as np

# class ResGenBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.residual = nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1),

#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#         )
#         self.shortcut = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0)
#         )
#         for m in self.residual.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight, math.sqrt(2))
#                 init.zeros_(m.bias)
#         for m in self.shortcut.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight)
#                 init.zeros_(m.bias)
        
#     def forward(self, x):
#         return self.residual(x) + self.shortcut(x)

# class OptimizedResDisblock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.residual = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#             nn.AvgPool2d(2),
#         )
#         self.shortcut = nn.Sequential(
#             nn.AvgPool2d(2),
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0),
#         )
#         for m in self.residual.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight, math.sqrt(2))
#                 init.zeros_(m.bias)
#         for m in self.shortcut.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight)
#                 init.zeros_(m.bias)

#     def forward(self, x):
#         return self.residual(x) + self.shortcut(x)

# class ResDisBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, down=False):
#         super().__init__()
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0) if in_channels != out_channels or down else nn.Identity(),
#             nn.AvgPool2d(2) if down else nn.Identity(),
#         )
#         self.residual = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1),
#             nn.AvgPool2d(2) if down else nn.Identity(),
#         )
#         for m in self.residual.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight, math.sqrt(2))
#                 init.zeros_(m.bias)
#         for m in self.shortcut.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight)
#                 init.zeros_(m.bias)

#     def forward(self, x):
#         return self.residual(x) + self.shortcut(x)
    
# class ResGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(cfg.latent_dim, 256*4*4)
#         self.conv = nn.Sequential(
#             nn.Unflatten(1, (256, 4, 4)),
#             *[ResGenBlock(256, 256) for _ in range(int(np.log2(cfg.image_size / 4)))],
#         )
#         self.output = nn.Sequential(
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             nn.Conv2d(256, cfg.image_channels, 3, 1, 1),
#             nn.Tanh(),
#         )
#         init.xavier_uniform_(self.linear.weight)
#         init.zeros_(self.linear.bias)
#         for m in self.output.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.xavier_uniform_(m.weight)
#                 init.zeros_(m.bias)
        
#     def forward(self, x):
#         x = self.linear(x)
#         x = self.conv(x)
#         x = self.output(x)
#         return x

# class ResDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             OptimizedResDisblock(cfg.image_channels, 128),
#             ResDisBlock(128, 128, down=True),
#             *[ResDisBlock(128, 128) for _ in range(3)],
#             nn.ReLU(),
#         )
#         self.linear = nn.Linear(128, 1)
#         init.xavier_uniform_(self.linear.weight)

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.linear(x)
#         return x

#     def forward_features(self, img):
#         features = self.model(img).mean(axis=[2, 3])
#         return features

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.generator = nn.Sequential(
#             # latent_dim
#             nn.Linear(cfg.latent_dim, 256 * 4 ** 2),
#             nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 256x4x4
#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             # 256x8x8
#             nn.Conv2d(256, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             # 128x16x16
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             # 64x32x32
#             nn.Conv2d(64, 32, 3, stride=1, padding=1),
#             nn.BatchNorm2d(32, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             # 32x64x64
#             nn.Conv2d(32, 16, 3, stride=1, padding=1),
#             nn.BatchNorm2d(16, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             # 16x128x128
#             nn.Conv2d(16, 8, 3, stride=1, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             # 8x256x256
#             nn.Conv2d(8, cfg.channels, 3, stride=1, padding=1),
#             nn.Tanh(),
#             # 3x256x256
#         )

#     def forward(self, z):
#         return self.generator(z)

# class Discriminator(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     def discriminator_block(in_filters, out_filters, bn=True):
    #         return [
    #             nn.Conv2d(in_filters, out_filters, 3, 2, 1),
    #             nn.LeakyReLU(0.2, inplace=True),
    #             nn.Dropout2d(0.25),
    #             nn.BatchNorm2d(out_filters, 0.8) if bn else nn.Identity(),
    #         ]

    #     self.model = nn.Sequential(
    #         *discriminator_block(cfg.channels, 8, bn=False),
    #         *discriminator_block(8, 16),
    #         *discriminator_block(16, 32),
    #         *discriminator_block(32, 64),
    #         *discriminator_block(64, 128),
    #         *discriminator_block(128, 256),
    #         nn.Flatten(start_dim=1),
    #     )
    #     self.adv_layer = nn.Sequential(
    #         nn.Linear(256 * (cfg.img_size / 2 ** 6) ** 2, 128),
    #         nn.LeakyReLU(0.2, inplace=True),
    #         nn.Linear(128, 1),
    #     )

    # def forward(self, img):
    #     features = self.forward_features(img)
    #     validity = self.adv_layer(features)
    #     return validity

    # def forward_features(self, img):
    #     features = self.model(img)
    #     return features

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(out_filters, 0.8) if bn else nn.Identity(),
            )
        
        self.model = nn.Sequential(
            encoder_block(cfg.image_channels, 16, bn=False),
            encoder_block(16, 32),
            encoder_block(32, 64),
            encoder_block(64, 128),
            nn.Flatten(start_dim=1),
        )
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * (cfg.image_size / 2 ** 4) ** 2, cfg.latent_dim),
            nn.Tanh(),
        )

    def forward(self, img):
        features = self.model(img)
        validity = self.adv_layer(features)
        return validity

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        dim = cfg.image_size / 4
        self.generator = nn.Sequential(
            nn.Linear(cfg.latent_dim, 128 * dim ** 2),
            nn.Unflatten(dim=1, unflattened_size=(128, dim, dim)),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, cfg.image_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.generator(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
                nn.BatchNorm2d(out_filters) if bn else nn.Identity(),
            )

        self.model = nn.Sequential(
            discriminator_block(cfg.image_channels, 16, bn=False),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
            discriminator_block(64, 128),
            nn.Flatten(start_dim=1),
        )
        self.adv_layer = nn.Linear(128 * (cfg.image_size / 2 ** 4) ** 2, 1)

    def forward(self, img):
        features = self.forward_features(img)
        validity = self.adv_layer(features)
        return validity

    def forward_features(self, img):
        features = self.model(img)
        return features
    