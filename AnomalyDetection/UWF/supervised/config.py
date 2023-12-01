from yacs.config import CfgNode as CN


_C = CN()
_C.DATA_SEED = 0  # seed for train/validation/test split
_C.TRAIN_SEED = 0  # seed for model training
_C.MODEL = 'densenet121'  # model name

_C.PATHS = CN()
_C.PATHS.DATA_DIR = 'C:/Users/kerry/OneDrive/Desktop/Projects/Datasets/UWF Images/Set 1'  # directory that stores good/bad images
_C.PATHS.OUTPUT_DIR = 'C:/Users/kerry/OneDrive/Desktop/Projects/UAD/supervised/results'  # directory to which results are saved

_C.DATA = CN()
_C.DATA.GOOD_LABEL = 'Good'  # good label, also good image directory name
_C.DATA.BAD_LABEL = 'Poor'  # bad label, also bad image directory name
_C.DATA.IMAGE_SIZE = [224, 224]  # image size in [H, W]

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0  # number of workers for dataloaders
_C.DATALOADER.PIN_MEMORY = False  # pin memory for dataloaders

_C.TRAINER = CN()
_C.TRAINER.BATCH_SIZE = 64  # training batch size
_C.TRAINER.NUM_EPOCHS = 500  # number of training epochs
_C.TRAINER.LR = 1e-3  # learning rate


def get_cfg_defaults():
    return _C.clone()

cfg = _C
