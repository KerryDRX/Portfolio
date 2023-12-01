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

print('\n' * 3)
print('Program started')

torch.multiprocessing.set_sharing_strategy('file_system')
if platform.system() == "Darwin":
    os.environ['OMP_NUM_THREADS'] = "1"

print(f'Platform: {platform.system()}')
print(f'GPU available: {torch.cuda.is_available()}')
print('\n' * 3)

city = 'All'
batch_size = 1
val_size = 109 #39 #109 #218
explained_var = 90
project_name = 'DeepSSM_inc_proc_rw5_p256'

dataset_directory = f'../dataset/{city}/Ventricles_64_3_cleaned/'
deepssm_directory = f'{dataset_directory}{project_name}/'
groom_directory = f'{deepssm_directory}groomed/'
mesh_directory = f'{deepssm_directory}mesh/'
aug_directory = f'{deepssm_directory}augmentation{explained_var}/'
loader_directory = f'{deepssm_directory}torch_loaders{explained_var}/'

# val_test_images_directory = f'{groom_directory}val_and_test_images/'
val_test_images_directory = f'../dataset/{city}/Ventricles_64_3/'
val_paticle_directory = f'{groom_directory}validation_particles/'

aug_data_csv = aug_directory + 'PartialWorldData.csv'
val_image_groomed_paths = sorted(glob(f'{val_test_images_directory}*.nrrd'))[-val_size:]
val_world_particles = sorted(glob(f'{val_paticle_directory}*_world.particles'))[-val_size:]


with open(f'{deepssm_directory}DeepSSM{explained_var}.json', 'r') as f:
    embedded_dim = json.load(f)['num_latent_dim']

model_name = f'DeepSSM{explained_var}'

# model_parameters = {
#     "model_name": model_name,
#     "num_latent_dim": int(embedded_dim),
#     "paths": {
#         "out_dir": deepssm_directory,
#         "loader_dir": loader_directory,
#         "aug_dir": aug_directory
#     },
#     "encoder": {
#         "deterministic": True
#     },
#     "decoder": {
#         "deterministic": True,
#         "linear": True
#     },
#     "loss": {
#         "function": "MSE",
#         "supervised_latent": True,
#     },
#     "trainer": {
#         "epochs": 15,
#         "learning_rate": 3e-4,
#         "decay_lr": True,
#         "val_freq": 1
#     },
#     "fine_tune": {
#         "enabled": False,
#     },
#     "use_best_model": True
# }

config_file = deepssm_directory + model_name + ".json"
# with open(config_file, "w") as outfile:
#     json.dump(model_parameters, outfile, indent=2)

#####################################################################
train_val_test_ratio = (0.9, 0.1, 0)
image_paths = sorted(glob(f'{dataset_directory}*.nrrd'))
mesh_paths = sorted(glob(f'{mesh_directory}*.ply'))
meshes = [sw.Mesh(mesh_path) for mesh_path in mesh_paths]

total_size = len(image_paths)
val_size = int(total_size * train_val_test_ratio[1])
test_size = int(total_size * train_val_test_ratio[2])
train_size = total_size - val_size - test_size
print(f'Total:      {total_size}')
print(f'Train:      {train_size}')
print(f'Validation: {val_size}')
print(f'Test:       {test_size}')

# random.seed(0)
indices = list(range(total_size))
# random.shuffle(indices)
all_indices = {
    'train': sorted(indices[:train_size]),
    'val': sorted(indices[train_size:train_size+val_size]),
    'test': sorted(indices[train_size+val_size:])
}

all_image_paths, all_mesh_paths, all_images, all_meshes, all_uids = [dict() for _ in range(5)]
for mode in ['train', 'val', 'test']:
    all_image_paths[mode] = [image_paths[i] for i in all_indices[mode]]
    all_mesh_paths[mode] = [mesh_paths[i] for i in all_indices[mode]]
    all_meshes[mode] = [meshes[i] for i in all_indices[mode]]

val_test_image_paths = all_image_paths['val'] + all_image_paths['test']
val_test_mesh_paths = all_mesh_paths['val'] + all_mesh_paths['test']

ref_image_file = groom_directory + 'reference_image.nrrd'
ref_image = sw.Image(ref_image_file)
ref_center = ref_image.center()

val_test_transforms = []
for vt_image_path, vt_mesh_path in tqdm(zip(val_test_image_paths, val_test_mesh_paths), total=val_size+test_size, desc='Validation/test transforms'):
    vt_image = sw.Image(vt_image_path)

    transform = np.eye(4)
    translation = ref_center - vt_image.center()
    transform[:3,-1] += translation
    
    val_test_transforms.append(transform)

val_transforms = val_test_transforms[:val_size]


spreadsheet_file = f'{groom_directory}train.xlsx'
project = sw.Project()
project.load(spreadsheet_file)
train_local_particles = project.get_string_column("local_particles_1")

# 11. Predict Validation Particles and Analyze Accuracy ##############################################
print('Stage 11: Predict Validation Particles and Analyze Accuracy')

val_out_dir = deepssm_directory + model_name + '/validation_predictions/'
predicted_val_world_particles = DeepSSMUtils.testDeepSSM(config_file, loader='validation')

local_val_prediction_dir = val_out_dir + 'local_predictions/'
if not os.path.exists(local_val_prediction_dir):
    os.makedirs(local_val_prediction_dir)

predicted_val_local_particles = []
for particle_file, transform in zip(predicted_val_world_particles, val_transforms):
    particles = np.loadtxt(particle_file)
    local_particle_file = particle_file.replace("FT_Predictions/", "local_predictions/")
    local_particles = sw.utils.transformParticles(particles, transform, inverse=True)
    np.savetxt(local_particle_file, local_particles)
    predicted_val_local_particles.append(local_particle_file)
print("Validation local predictions written to: " + local_val_prediction_dir)

'''
Analyze validation accuracy in terms of:
- MSE between true and predicted world partcles
- Surface to surface distance between true mesh and mesh generated from predicted local particles
'''
mean_MSE, std_MSE = DeepSSMUtils.analyzeMSE(predicted_val_world_particles, val_world_particles)
print("Validation world particle MSE: "+str(mean_MSE)+" +- "+str(std_MSE))

ref_index = sw.find_reference_mesh_index(all_meshes['train'])
template_mesh = all_mesh_paths['train'][ref_index]
template_particles = train_local_particles[ref_index].replace("./", groom_directory)
# Get distabce between clipped true and predicted meshes
mean_dist = DeepSSMUtils.analyzeMeshDistance(
    predicted_val_local_particles, all_mesh_paths['val'], 
    template_particles, template_mesh, val_out_dir,
    #planes=val_planes
)
print("Validation mean mesh surface-to-surface distance: "+str(mean_dist))

# DeepSSMUtils.analyzeResults(out_dir, DT_dir, prediction_dir, mean_prefix)

