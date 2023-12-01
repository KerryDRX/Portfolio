from yacs.config import CfgNode as CN


_C = CN()
_C.DEVICE = 'cuda:0'

_C.SEEDS = CN()
_C.SEEDS.DATA = 0

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.PIN_MEMORY = True

_C.TRAINER = CN()
_C.TRAINER.LR = 1e-3

_C.DATASET = CN()
_C.DATASET.NAME = 'MALPEM2'
if _C.DATASET.NAME == 'FAST':
    _C.DATASET.RESAMPLE = 2
    _C.DATASET.IMAGE_SIZE = (128, 128, 128)
    _C.DATASET.CLASS_NAMES = ['CSF', 'GM', 'WM']
    _C.DATASET.NUM_CLASSES = len(_C.DATASET.CLASS_NAMES) + 1
    _C.DATASET.SITES = ['AnnArbor', 'Atlanta', 'Baltimore', 'Bangor', 'Beijing', 'Berlin', 'Cambridge', 'Cleveland', 'Dallas', 'ICBM', 'Leiden', 'Milwaukee', 'Munchen', 'NewHaven', 'NewYork', 'Newark']
    _C.DATASET.TRAIN_VAL_SITE = 'Beijing'
    _C.DATASET.TRAIN_SIZE = 100
    _C.DATASET.VAL_SIZE = 20
    _C.DATASET.TEST_SIZE = 20
    _C.DATALOADER.NUM_WORKERS = 3
    _C.TRAINER.MAX_EPOCHS = 200
if _C.DATASET.NAME == 'MALPEM':
    _C.DATASET.RESAMPLE = 2
    _C.DATASET.IMAGE_SIZE = (128, 128, 128)
    _C.DATASET.CLASS_NAMES = ['V-R', 'V-L', 'V-3', 'V-4']
    _C.DATASET.NUM_CLASSES = len(_C.DATASET.CLASS_NAMES) + 1
    _C.DATASET.TRAIN_SIZE = 500
    _C.DATASET.VAL_SIZE = 90
    _C.DATASET.TEST_SIZE = 100
    _C.DATALOADER.NUM_WORKERS = 8
    _C.TRAINER.MAX_EPOCHS = 100
if _C.DATASET.NAME == 'MALPEM2':
    _C.DATASET.RESAMPLE = 2
    _C.DATASET.IMAGE_SIZE = (128, 128, 128)
    _C.DATASET.NUM_CLASSES = 5
    _C.DATASET.TRAIN_SIZE = 100
    _C.DATASET.VAL_SIZE = 50
    _C.DATASET.TEST_SIZE = 100
    _C.DATALOADER.NUM_WORKERS = 4
    _C.TRAINER.MAX_EPOCHS = 50

_C.PATHS = CN()
if 'MALPEM' in _C.DATASET.NAME:
    _C.PATHS.DATA_DIR = f'C:/Users/kerry/OneDrive/Desktop/Projects/Datasets/MALPEM'
_C.PATHS.OUTPUT_DIR = f'C:/Users/kerry/OneDrive/Desktop/Projects/Code/EDL/segmentation/outputs/{_C.DATASET.NAME}'

_C.MODEL = CN()
_C.MODEL.ACTIVATION = 'exp'
_C.MODEL.EPS = 0

_C.LOSS = CN()
# _C.LOSS.FUNCTION = 'sdice'
_C.LOSS.ALPHA = 0.3
_C.LOSS.BETA = 0.7
_C.LOSS.GAMMA = 4/3
_C.LOSS.REG_COEF = 0.1


def get_cfg_defaults():
    return _C.clone()

cfg = _C
