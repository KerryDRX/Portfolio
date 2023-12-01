from yacs.config import CfgNode as CN


_C = CN()
_C.DEVICE = 'cuda:0'

_C.SEEDS = CN()
_C.SEEDS.DATA = 1

_C.PATHS = CN()
_C.PATHS.ROOT = 'C:/Users/kerry/OneDrive/Desktop/Projects/Code/EDL/regression'
_C.PATHS.DATA_DIR = f'{_C.PATHS.ROOT}/datasets'
_C.PATHS.OUTPUT_DIR = f'{_C.PATHS.ROOT}/outputs_br'

_C.DATASET = CN()

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 128
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.NUM_WORKERS = 0

_C.TRAINER = CN()
_C.TRAINER.MAX_EPOCHS = 714
_C.TRAINER.LR = 5e-3
_C.TRAINER.EVAL_INTERVAL = 1

_C.LOSS = CN()
_C.LOSS.REG_COEF = 1


def get_cfg_defaults():
    return _C.clone()

cfg = _C
