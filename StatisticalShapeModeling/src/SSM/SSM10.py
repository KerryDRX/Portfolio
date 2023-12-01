import os
from glob import glob
import platform
import torch
import random
import numpy as np
import subprocess
import json
from tqdm import tqdm
import shapeworks as sw
import DataAugmentationUtils
import DeepSSMUtils

# 10. Train DeepSSM Model ##############################################
print('Stage 10: Train DeepSSM Model')

embedded_dim = 17

dataset_directory = '../dataset/Baltimore/Ventricles_256_2/'
deepssm_directory = f'{dataset_directory}DeepSSM/'
aug_directory = f'{deepssm_directory}augmentation/'
loader_directory = f'{deepssm_directory}torch_loaders/'

model_name = "DeepSSM_ventricle2"

model_parameters = {
    "model_name": model_name,
    "num_latent_dim": int(embedded_dim),
    "paths": {
        "out_dir": deepssm_directory,
        "loader_dir": loader_directory,
        "aug_dir": aug_directory
    },
    "encoder": {
        "deterministic": True
    },
    "decoder": {
        "deterministic": True,
        "linear": True
    },
    "loss": {
        "function": "MSE",
        "supervised_latent": True,
    },
    "trainer": {
        "epochs": 10,
        "learning_rate": 0.001,
        "decay_lr": False,
        "val_freq": 1
    },
    "fine_tune": {
        "enabled": False,
    },
    "use_best_model": True
}
    
config_file = deepssm_directory + model_name + ".json"
with open(config_file, "w") as outfile:
    json.dump(model_parameters, outfile, indent=2)

DeepSSMUtils.trainDeepSSM(config_file)

print('Program Stage 10 Completed')
