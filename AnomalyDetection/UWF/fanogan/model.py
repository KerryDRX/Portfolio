import torch.nn as nn
from config import cfg


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        def encoder_block(in_filters, out_filters, bn=True):
            return nn.Sequential(
                *([
                    nn.Conv2d(in_filters, out_filters, 3, 1, 1),
                ] + ([
                    nn.BatchNorm2d(out_filters, momentum=0.7),
                ] if bn else []) + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.AvgPool2d(2),
                ])
            )
        
        self.encoder = nn.Sequential(
            encoder_block(cfg.DATA.IMAGE_CHANNELS, 16, bn=False),
            encoder_block(16, 32),
            encoder_block(32, 64),
            encoder_block(64, 128),
            encoder_block(128, 256),
            encoder_block(256, 512),
            nn.Flatten(start_dim=1),
            nn.Linear(8192, cfg.DATA.LATENT_DIM),
            nn.Tanh(),
        )

    def forward(self, image):
        return self.encoder(image)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        def generator_block(in_filters, out_filters, bn=True):
            return nn.Sequential(
                *([
                    nn.ConvTranspose2d(in_filters, out_filters, 4, 2, 1, bias=False),
                ] + ([
                    nn.BatchNorm2d(out_filters, momentum=0.7),
                ] if bn else []) + [
                    nn.ReLU(inplace=True),
                ])
            )
        self.generator = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(cfg.DATA.LATENT_DIM, 1, 1)),
            nn.ConvTranspose2d(cfg.DATA.LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            generator_block(512, 256),
            generator_block(256, 128),
            generator_block(128, 64),
            generator_block(64, 32),
            generator_block(32, 16),
            nn.ConvTranspose2d(16, cfg.DATA.IMAGE_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, image):
        return self.generator(image)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            return nn.Sequential(
                *([
                    nn.Conv2d(in_filters, out_filters, 3, 1, 1),
                ] + ([
                    nn.BatchNorm2d(out_filters, momentum=0.7),
                ] if bn else []) + [
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.25),
                    nn.AvgPool2d(2),
                ])
            )
        self.discriminator = nn.Sequential(
            discriminator_block(cfg.DATA.IMAGE_CHANNELS, 16, bn=False),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            nn.Flatten(start_dim=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8192, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, image):
        return self.classifier(self.discriminator(image))

    def extract_features(self, image):
        return self.discriminator(image)
    