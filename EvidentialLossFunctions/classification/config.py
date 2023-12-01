from yacs.config import CfgNode as CN


_C = CN()
_C.DEVICE = 'cuda:0'

_C.SEEDS = CN()
_C.SEEDS.DATA = 0

_C.PATHS = CN()
_C.PATHS.ROOT = 'C:/Users/kerry/OneDrive/Desktop/Projects/Code/EDL/classification'
_C.PATHS.DATA_DIR = f'{_C.PATHS.ROOT}/datasets'
_C.PATHS.OUTPUT_DIR = f'{_C.PATHS.ROOT}/outputs'

_C.DATASET = CN()
_C.DATASET.NAME = 'CIFAR10'
_C.DATASET.NUM_CLASSES = 10

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 128
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.NUM_WORKERS = 8

_C.TRAINER = CN()
_C.TRAINER.MAX_EPOCHS = 200
_C.TRAINER.LR = 0.1

_C.MODEL = CN()
_C.MODEL.ACTIVATION = 'exp'
_C.MODEL.EPS = 0

_C.LOSS = CN()
_C.LOSS.FUNCTION = 'lpnorm'
_C.LOSS.REG_COEF = 0.1


def get_cfg_defaults():
    return _C.clone()

cfg = _C
