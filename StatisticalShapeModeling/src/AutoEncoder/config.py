from yacs.config import CfgNode as CN


_C = CN()
_C.DEVICE = 'cuda:0'

_C.SEED = CN()
_C.SEED.DATA = 0

_C.PATHS = CN()
_C.PATHS.DATA_DIR = 'C:/Users/kerry/OneDrive/Desktop/Projects/Datasets/MALPEM'
_C.PATHS.OUTPUT_DIR = 'C:/Users/kerry/OneDrive/Desktop/Projects/Code/Shape/results'

_C.DATASET = CN()
_C.DATASET.RESAMPLE = 2
_C.DATASET.IMAGE_SIZE = 128
_C.DATASET.TRAIN_VAL_TEST_RATIO = [0.7, 0.1, 0.2]

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE_TRAIN = 16
_C.DATALOADER.BATCH_SIZE_EVAL = 8
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.NUM_WORKERS = 8

_C.TRAINER = CN()
_C.TRAINER.LR = 1e-3
_C.TRAINER.MAX_EPOCHS = 200
_C.TRAINER.EVAL_INTERVAL = 5

def get_cfg_defaults():
    return _C.clone()


cfg = _C
