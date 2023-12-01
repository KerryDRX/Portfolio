import warnings
warnings.filterwarnings("ignore")
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
import pandas as pd

print('\n' * 10)
print('Program started')

torch.multiprocessing.set_sharing_strategy('file_system')
if platform.system() == "Darwin":
    os.environ['OMP_NUM_THREADS'] = "1"

print(f'Platform: {platform.system()}')
print(f'GPU available: {torch.cuda.is_available()}')
print('\n' * 3)

# 9. Create Data Loaders ##############################################
print('Stage 9. Create Data Loaders')

city = 'Ellipsoids'
project_name = 'DeepSSM'
segmentation_type = 'segmentations' # Ventricles_64_3_cleaned
batch_size = 1
val_size = 40 #19/39/40/109/218

for explained_var in [99]:
    print(f'\nExplained variance: {explained_var}%\n')
    dataset_directory = f'../dataset/{city}/{segmentation_type}/'
    deepssm_directory = f'{dataset_directory}{project_name}/'
    groom_directory = f'{deepssm_directory}groomed/'
    mesh_directory = f'{deepssm_directory}mesh/'
    aug_directory = f'{deepssm_directory}augmentation{explained_var}/'
    loader_directory = f'{deepssm_directory}torch_loaders{explained_var}/'

    if not os.path.exists(loader_directory):
        os.mkdir(loader_directory)

    # val_test_images_directory = f'{groom_directory}val_and_test_images/'
    # val_test_images_directory = f'../dataset/{city}/Ventricles_64_3/'
    val_test_images_directory = f'../dataset/{city}/segmentations/'
    val_paticle_directory = f'{groom_directory}validation_particles/'

    aug_data_csv = aug_directory + 'PartialWorldData.csv'
    ########################################
    # df = pd.read_csv(aug_data_csv, header=None)
    # aug_data_csv = aug_directory + 'PartialWorldOrigData.csv'
    # if not os.path.exists(aug_data_csv):
    #     for i in tqdm(range(df.shape[0])):
    #         df.at[i, 0] = df.at[i, 0].replace(f'_cleaned/{project_name}/groomed/train_images', '')
    #     df.to_csv(
    #         aug_data_csv,
    #         header=False, 
    #         index=False
    #     )
    ########################################
    val_image_groomed_paths = sorted(glob(f'{val_test_images_directory}*.nrrd'))[-val_size:]
    val_world_particles = sorted(glob(f'{val_paticle_directory}*_world.particles'))[-val_size:]

    DeepSSMUtils.getTrainLoader(loader_directory, aug_data_csv, batch_size)
    DeepSSMUtils.getValidationLoader(loader_directory, val_image_groomed_paths, val_world_particles)

    print('\n' * 3)

    print('Stage 10: Train DeepSSM Model')

    embedded_dim = np.load(f'{aug_directory}PCA_Particle_Info/original_PCA_scores.npy').shape[1]

    model_name = f'DeepSSM{explained_var}'

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
            "supervised_latent": True
        },
        "trainer": {
            "epochs": 10,
            "learning_rate": 3e-4,
            "decay_lr": True,
            "val_freq": 1
        },
        "fine_tune": {
            "enabled": False,
            # "epochs": 10,
            # "learning_rate": 1e-4,
            # "decay_lr": True,
            # "val_freq": 1,
            # "loss": "MSE"
        },
        "use_best_model": True
    }

    config_file = deepssm_directory + model_name + ".json"
    with open(config_file, "w") as outfile:
        json.dump(model_parameters, outfile, indent=2)

    DeepSSMUtils.trainDeepSSM(config_file)
