from yacs.config import CfgNode as CN


_C = CN()
_C.log_dir = 'C:/Users/kerry/OneDrive/Desktop/Projects/UAD/results'
_C.seed = 0
_C.k = 5
_C.modes = ['train_orig', 'train', 'validation', 'test']
_C.image_size = 64
_C.image_channels = 1
_C.good_label = 'good'
_C.bad_label = 'bad'
_C.latent_dim = 128

_C.trainer = CN()
_C.trainer.batch_size = 30
_C.trainer.num_epochs = 5000
_C.trainer.val_interval = 10
_C.trainer.lr = 2e-4
_C.trainer.b1 = 0.5
_C.trainer.b2 = 0.999

# CAE
_C.channels = (32,64,128,256,512,)

# SSAE
_C.trainer.lambda0 = 1
_C.trainer.lambda1 = 1
_C.trainer.lambda2 = 1

# Patch CAE
_C.patch_size = 64
_C.stride = 16
_C.ppd = len(range(0, _C.image_size-_C.patch_size+1, _C.stride))  # patches per dimension

# DAGMM
_C.reduced_dim = 3
_C.n_gmm = 4
_C.trainer.lambda_energy = 0.1
_C.trainer.lambda_cov_diag = 0.005

# f-AnoGAN
_C.trainer.encoder_epochs = 3000
_C.trainer.lambda_gp = 10
_C.trainer.kappa = 1.0
_C.trainer.n_critic = 2
_C.trainer.save_image_interval = 400
_C.trainer.save_model_interval = 100
_C.trainer.fid_stats_path = 'C:/Users/kerry/OneDrive/Desktop/Projects/UAD/results/fid_stats.npz'

def get_cfg_defaults():
    return _C.clone()

cfg = _C
