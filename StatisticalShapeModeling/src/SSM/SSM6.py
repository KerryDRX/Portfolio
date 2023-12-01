import os
from glob import glob
import torch
import random
import numpy as np
import shapeworks as sw
import DataAugmentationUtils
import pandas as pd
from tqdm import tqdm

print('Stage 6. Augment Data')

city = 'All'
num_samples = 2000
explained_var = 90
percent_variability = explained_var / 100
sampler_type = 'kde'
processes = 8
train_size = 981 #158 #158 #872

project_name = 'DeepSSM_inc_proc_rw5_p256'
base_dir = f'../dataset/{city}/Ventricles_64_3_cleaned/{project_name}'
aug_directory = f'{base_dir}/augmentation{explained_var}/'
img_list = sorted(glob(f'{base_dir}/groomed/train_images/*.nrrd'))[:train_size]
local_point_list = sorted(glob(f'{base_dir}/groomed/train_particles/*_local.particles'))[:train_size]
world_point_list = sorted(glob(f'{base_dir}/groomed/train_particles/*_world.particles'))[:train_size]

if not os.path.exists(aug_directory):
    os.mkdir(aug_directory)

embedded_dim = DataAugmentationUtils.runDataAugmentation(
    out_dir=aug_directory,
    img_list=img_list, 
    local_point_list=local_point_list, 
    num_samples=num_samples, 
    #num_dim,
    percent_variability=percent_variability, 
    sampler_type=sampler_type,
    #mixture_num,
    processes=processes, 
    world_point_list=world_point_list,
)

print(f'PCA dimensions retained: {embedded_dim}')

print('Getting total world data ...')
aug_data_csv = aug_directory + 'TotalData.csv'
df = pd.read_csv(aug_data_csv, header=None)
for i in tqdm(range(df.shape[0])):
    df.at[i, 1] = df.at[i, 1].replace('local', 'world')
df.to_csv(
    aug_directory + 'TotalWorldData.csv', 
    header=False, 
    index=False
)

print('Getting partial data ...')
aug_data_csv = aug_directory + 'TotalData.csv'
df = pd.read_csv(aug_data_csv, header=None)
df[:-num_samples].to_csv(
    aug_directory + 'PartialData.csv', 
    header=False, 
    index=False
)

print('Getting partial world data ...')
df = pd.read_csv(aug_directory + 'PartialData.csv', header=None)
for i in tqdm(range(df.shape[0])):
    df.at[i, 0] = df.at[i, 0].replace(f'_cleaned/{project_name}/groomed/train_images', '')
    df.at[i, 1] = df.at[i, 1].replace('local', 'world')
df.to_csv(
    aug_directory + 'PartialWorldOrigData.csv', 
    header=False, 
    index=False
)

print('Data Augmentation Completed')
