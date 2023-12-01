import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import cfg


def _rescale(image, **kwargs):
    '''
    Linearly rescale image intensity from [0th percentile, 99.99th percentile] to [-1, 1].
    Out-of-range intensities are clipped to [-1, 1].

    Parameters:
    ----------
        image: numpy.ndarray
            An image to rescale.
    
    Returns:
    ----------
        image: numpy.ndarray
            Rescaled image with intensities in the range of [-1, 1].
    '''
    image = image.astype(np.float32)
    min_intensity, max_intensity = np.percentile(image, 0), np.percentile(image, 99.99)
    image = (image - min_intensity) / (max_intensity - min_intensity)
    image = np.clip(image, 0, 1)
    image = image * 2 - 1
    return image

def _channel_3to2(image, **kwargs):
    '''
    Reduce image channels from 3 (RGB) to 2 (RG).

    Parameters:
    ----------
        image: numpy.ndarray
            An image with shape [H, W, 3].
    
    Returns:
    ----------
        image: numpy.ndarray
            An image with shape [H, W, 2].
    '''
    image = image[:, :, :2]
    return image

def _channel_2to3(image, **kwargs):
    '''
    Increase image channels from 2 (RG) to 3 (RGB).
    The third channel is the average of the first two channels.
    This transformation is for inception score calculation in GAN evaluation.

    Parameters:
    ----------
        image: torch.Tensor
            Image tensor with shape [B, 2, H, W] or [2, H, W].
    
    Returns:
    ----------
        image: torch.Tensor
            Image tensor with shape [B, 3, H, W] or [3, H, W].
    '''
    r = image[:, :1] if image.ndim == 4 else image[:1]                # shape [B, 1, H, W] or [1, H, W]
    g = image[:, 1:2] if image.ndim == 4 else image[1:2]              # shape [B, 1, H, W] or [1, H, W]
    b = (r + g) / 2                                                   # shape [B, 1, H, W] or [1, H, W]
    image = torch.cat([image, b], dim=(1 if image.ndim == 4 else 0))  # shape [B, 3, H, W] or [3, H, W]
    return image

channel_3to2 = A.Lambda(image=_channel_3to2)
channel_2to3 = A.Lambda(image=_channel_2to3)
resize = A.Resize(height=cfg.DATA.IMAGE_SIZE[0], width=cfg.DATA.IMAGE_SIZE[1])
horizontal_flip = A.HorizontalFlip(p=0.5)
random_affine = A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=3, p=0.5)
rescale = A.Lambda(image=_rescale)
to_tensor = ToTensorV2()

def build_transforms():
    '''
    Image transformations of three types:
        augmentation: data augmentation transforms.
        identity_2channels: identity transforms with 2 image channels.
        identity_3channels: identity transforms with 3 image channels.

    Returns:
    ----------
        transforms: dict
            A dictionary of image transformations.
    '''
    transforms = {
        'augmentation':       A.Compose([channel_3to2, resize, horizontal_flip, random_affine, rescale, to_tensor]),
        'identity_2channels': A.Compose([channel_3to2, resize,                                 rescale, to_tensor]),
        'identity_3channels': A.Compose([channel_3to2, resize,                                 rescale, to_tensor, channel_2to3]),
    }
    return transforms
