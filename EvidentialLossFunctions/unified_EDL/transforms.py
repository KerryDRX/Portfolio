import random
import torchio as tio
from torchio import transforms as T

def random_brightness(x):
    return random.uniform(0.5, 2) * x

def build_transforms():
    def _resample(subject):
        subject = T.Resample((3.22, 1.62, 1.62))(subject)
        image_sizes = subject['image'][tio.DATA].shape[1:]
        patch_sizes = (80, 160, 160)
        if all([image_size >= patch_size for image_size, patch_size in zip(image_sizes, patch_sizes)]):
            return subject
        padding = ()
        for image_size, patch_size in zip(image_sizes, patch_sizes):
            if image_size >= patch_size:
                padding_ini = padding_fin = 0
            else:
                diff = patch_size - image_size
                padding_ini = diff // 2
                padding_fin = diff - padding_ini
            padding += (padding_ini, padding_fin)
        return T.Pad(padding=padding, padding_mode='constant')(subject)
    
    augmentation = T.Compose([
        T.RandomAffine(scales=(0.85, 1.25), degrees=15),
        T.RandomFlip(axes=2),
        # T.RandomElasticDeformation(),
        T.Lambda(random_brightness, types_to_apply=[tio.INTENSITY]),
    ], p=0.15)
    signal = T.RescaleIntensity(in_min_max=(-79, 304), out_min_max=(0, 1))
    onehot = T.OneHot(num_classes=3)
    transforms = {
        'resample': _resample,
        'augmentation': T.Compose([augmentation, signal, onehot]),
        'identity': T.Compose([signal, onehot]),
    }
    return transforms
