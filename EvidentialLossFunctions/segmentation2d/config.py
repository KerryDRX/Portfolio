from yacs.config import CfgNode as CN


_C = CN()
_C.DEVICE = 'cuda:0'

_C.PATHS = CN()
_C.PATHS.DATA = f'C:/Users/kerry/OneDrive/Desktop/Projects/Datasets'
_C.PATHS.OUTPUT = f'outputs'

_C.SEEDS = CN()
_C.SEEDS.DATA = 0

_C.DATASET = CN()
_C.DATASET.NAME = 'RITE'
_C.DATASET.NUM_CHANNELS = 3
_C.DATASET.FOLDS = 5

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 16
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.NUM_WORKERS = 0

_C.TRAINER = CN()
_C.TRAINER.LR = 0.1
_C.TRAINER.NUM_EPOCHS = 150

# _C.MODEL = CN()
# _C.MODEL.ACTIVATION = 'exp'
# _C.MODEL.EPS = 0

# _C.LOSS = CN()
# _C.LOSS.FUNCTION = 'sdice'
# _C.LOSS.ALPHA = 0.3
# _C.LOSS.BETA = 0.7
# _C.LOSS.GAMMA = 4/3
# _C.LOSS.REG_COEF = 0.1


def get_cfg_defaults():
    return _C.clone()

cfg = _C
