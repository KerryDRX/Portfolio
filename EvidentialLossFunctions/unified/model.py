import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels)
        self.act = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck: self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        feat = self.act(self.norm1(self.conv1(x)))
        feat = self.act(self.norm2(self.conv2(feat)))
        out = feat if self.bottleneck else self.pooling(feat)
        return out, feat

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, feat_channels, num_classes=None):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+feat_channels, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm3d(num_features=in_channels//2)
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm3d(num_features=in_channels//2)
        self.act = nn.ReLU()
        self.num_classes = num_classes
        if num_classes is not None:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=1)
        
    def forward(self, x, feat):
        out = self.upconv(x)
        out = torch.cat((out, feat), 1)
        out = self.act(self.norm1(self.conv1(out)))
        out = self.act(self.norm2(self.conv2(out)))
        if self.num_classes is not None:
            out = self.conv3(out)
        return out

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, channels=[32, 64, 128, 256]):
        super(UNet3D, self).__init__()
        self.encoder1 = EncoderBlock(in_channels=in_channels, out_channels=channels[0])
        self.encoder2 = EncoderBlock(in_channels=channels[0], out_channels=channels[1])
        self.encoder3 = EncoderBlock(in_channels=channels[1], out_channels=channels[2])
        self.bottleneck = EncoderBlock(in_channels=channels[2], out_channels=channels[3], bottleneck=True)
        self.decoder3 = DecoderBlock(in_channels=channels[3], feat_channels=channels[2])
        self.decoder2 = DecoderBlock(in_channels=channels[2], feat_channels=channels[1])
        self.decoder1 = DecoderBlock(in_channels=channels[1], feat_channels=channels[0], num_classes=num_classes,)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        out, feat1 = self.encoder1(input)
        out, feat2 = self.encoder2(out)
        out, feat3 = self.encoder3(out)
        out, _ = self.bottleneck(out)
        out = self.decoder3(out, feat3)
        out = self.decoder2(out, feat2)
        out = self.decoder1(out, feat1)
        return out

def build_model():
    return UNet3D(in_channels=1, num_classes=3)
