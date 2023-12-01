from config import cfg
import torchio as tio
import torchio.transforms as T


def build_transforms():
    assert tio.__version__ == '0.18.88'
    spatial = T.RandomAffine(scales=0.05, degrees=3, translation=1)
    resample = T.Compose([
        T.Resample(cfg.DATASET.RESAMPLE),
        T.CropOrPad((cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
    ])
    signal = T.RescaleIntensity(percentiles=(0.1, 99.9), out_min_max=(0, 1))
    remapping = T.RemapLabels({i: (1 if i == 1 else 0) for i in range(139)})
    # remapping = T.RemapLabels({
    #     i: (
    #         1 if i in {21, 23}
    #         else 2 if i in {22, 24}
    #         else 3 if i == 1
    #         else 4 if i == 2
    #         else 0
    #     )
    #     for i in range(139)
    # })
    # onehot = T.OneHot(num_classes=cfg.DATASET.NUM_CLASSES)
    transforms = {
        'augmentation': T.Compose([resample, signal, remapping]),
        'identity': T.Compose([resample, signal, remapping]),
    }
    return transforms
