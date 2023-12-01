from yacs.config import CfgNode as CN


_C = CN()
_C.DATA_SEED = 0  # seed for train/validation/test split
_C.TRAIN_SEED = 0  # seed for model training

_C.PATHS = CN()
_C.PATHS.DATA_DIR = 'C:/Users/kerry/OneDrive/Desktop/Projects/Datasets/UWF Images/Set 1'  # directory that stores good/bad images
_C.PATHS.OUTPUT_DIR = 'C:/Users/kerry/OneDrive/Desktop/Projects/UAD/fanogan/results'  # directory to which results are saved

_C.DATA = CN()
_C.DATA.GOOD_LABEL = 'Good'  # good label, also good image directory name
_C.DATA.BAD_LABEL = 'Poor'  # bad label, also bad image directory name
_C.DATA.IMAGE_SIZE = [256, 256]  # image size in [H, W]
_C.DATA.IMAGE_CHANNELS = 2  # number of image channels
_C.DATA.LATENT_DIM = 128  # image latent dimension

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0  # number of workers for dataloaders
_C.DATALOADER.PIN_MEMORY = False  # pin memory for dataloaders

_C.TRAINER = CN()
_C.TRAINER.BATCH_SIZE = 64  # training batch size

_C.TRAINER.GAN = CN()  # GAN training settings
_C.TRAINER.GAN.NUM_ITERATIONS = 10000  # number of training iterations
_C.TRAINER.GAN.LR = 2e-4  # learning rate
_C.TRAINER.GAN.CRITIC = 5  # times that discriminator is trained more than generator
_C.TRAINER.GAN.LAMBDA_GP = 10  # weight of gradient penalty in the loss function
_C.TRAINER.GAN.EVAL_INTERVAL = 500  # frequency to save the generated images and evaluate GAN performance

_C.TRAINER.ENCODER = CN()
_C.TRAINER.ENCODER.NUM_EPOCHS = 200  # number of training epochs
_C.TRAINER.ENCODER.LR = 2e-4  # learning rate
_C.TRAINER.ENCODER.KAPPA = 1.0  # weight of feature loss (discrimination loss) in the loss function
_C.TRAINER.ENCODER.EVAL_INTERVAL = 1  # frequency to evaluate encoder performance


def get_cfg_defaults():
    return _C.clone()

cfg = _C
