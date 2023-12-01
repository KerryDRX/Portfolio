from config import cfg
from torchio import transforms as T


def build_transforms():
    resample = T.Compose([T.Resample(cfg.DATASET.RESAMPLE), T.CropOrPad(cfg.DATASET.IMAGE_SIZE)])
    spatial = T.RandomAffine(scales=0.05, degrees=3, translation=1)
    signal = T.RescaleIntensity(percentiles=(0.1, 99.99), out_min_max=(0, 1))
    onehot = T.OneHot(num_classes=cfg.DATASET.NUM_CLASSES)
    if cfg.DATASET.NAME == 'FAST':
        transforms = {
            'augmentation': T.Compose([resample, spatial, signal, onehot]),
            'identity': T.Compose([resample, signal, onehot]),
        }
    if cfg.DATASET.NAME == 'MALPEM':
        remap = T.RemapLabels(remapping={i: (
            1 if i in {21, 23}
            else 2 if i in {22, 24}
            else 3 if i == 1
            else 4 if i == 2
            else 0
        ) for i in range(139)})
        transforms = {
            'augmentation': T.Compose([resample, spatial, signal, remap, onehot]),
            'identity': T.Compose([resample, signal, remap, onehot]),
        }
    if cfg.DATASET.NAME == 'MALPEM2':
        remap = T.RemapLabels(remapping={i: (
            1 if i == 5
            else 2 if i == 6
            else 3 if i == 19
            else 4 if i == 20
            else 0
        ) for i in range(139)})
        transforms = {
            'augmentation': T.Compose([resample, spatial, signal, remap, onehot]),
            'identity': T.Compose([resample, signal, remap, onehot]),
        }
    return transforms
