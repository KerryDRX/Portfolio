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

print('\n' * 3)
print('Program started')

torch.multiprocessing.set_sharing_strategy('file_system')
# if platform.system() == "Darwin":
#     os.environ['OMP_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"

print(f'Platform: {platform.system()}')
print(f'GPU available: {torch.cuda.is_available()}')
print('\n' * 3)

# 1. Load Images ##############################################
print('Stage 1: Load Images')

print('Creating directories ...')

city = 'All'
size = 64
vent = 3
project_name = 'DeepSSM_rw1'

dataset_directory = f'../dataset/{city}/Ventricles_{size}_{vent}_cleaned/'
deepssm_directory = f'{dataset_directory}{project_name}/'
groom_directory = f'{deepssm_directory}groomed/'
mesh_directory = f'{deepssm_directory}mesh/'

val_test_images_directory = f'{groom_directory}val_and_test_images/'
val_paticle_directory = f'{groom_directory}validation_particles/'

for directory in [
    deepssm_directory, mesh_directory, groom_directory,
    val_test_images_directory, val_paticle_directory
]:
    if not os.path.exists(directory):
        os.mkdir(directory)

print('Searching for images ...')
        
image_paths = sorted(glob(f'{dataset_directory}*.nrrd'))
uids = [image_path.split('/')[-1][:5] for image_path in image_paths]

print(f'First image path: {image_paths[0]}')
print(f'First image UID: {uids[0]}')

print('Loading images and converting to meshes ...')

images = [
    sw.Image(image_path)#.resample([1,1,1], sw.InterpolationType.Linear) 
    for image_path in image_paths
]
meshes = [image.toMesh(isovalue=0.5).fillHoles() for image in images]

print('Saving meshes ...')

sw.utils.save_meshes(
    outDir=mesh_directory,
    swMeshList=meshes,
    swMeshNames=uids,
    extension='ply',
    verbose=False
)
mesh_paths = sorted(glob(f'{mesh_directory}*.ply'))

print(f'First mesh path: {mesh_paths[0]}')
print('\n' * 3)

# 2. Define Split ##############################################
print('Stage 2: Define Split')

train_val_test_ratio = (0.9, 0.1, 0)

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

print('Spliting images and meshes ...')

all_image_paths, all_mesh_paths, all_images, all_meshes, all_uids = [dict() for _ in range(5)]
for mode in ['train', 'val', 'test']:
    all_image_paths[mode] = [image_paths[i] for i in all_indices[mode]]
    all_mesh_paths[mode] = [mesh_paths[i] for i in all_indices[mode]]
    all_images[mode] = [images[i] for i in all_indices[mode]]
    all_meshes[mode] = [meshes[i] for i in all_indices[mode]]
    all_uids[mode] = [uids[i] for i in all_indices[mode]]

print('\n' * 3)

# 3. Training Mesh Transforms ##############################################
print('Stage 3: Training Mesh Transforms')

ref_index = 25 #sw.find_reference_mesh_index(all_meshes['train'])

print(f'Reference index: {ref_index}')
print('Saving reference.vtk ...')

ref_mesh = all_meshes['train'][ref_index].copy()
ref_translate = ref_mesh.center()
ref_mesh.translate(-ref_translate)
ref_mesh.write(f'{mesh_directory}reference.vtk')

train_rigid_transforms = []
for train_mesh in tqdm(all_meshes['train'], desc='Train rigid transforms'):
    rigid_transform = train_mesh.createTransform(
        target=ref_mesh, 
        align=sw.Mesh.AlignmentType.Rigid, 
        iterations=100
    )
    train_mesh.applyTransform(rigid_transform)
    train_rigid_transforms.append(rigid_transform)

train_transforms = train_rigid_transforms

print('\n' * 3)

# 4. Optimize Training Particles ##############################################
print('Stage 4. Optimize Training Particles')

output_directory = deepssm_directory

mesh_files = all_mesh_paths['train']
meshes = all_meshes['train']

distances = np.zeros(len(meshes))
for i in range(len(meshes)):
    distances[i] = np.mean(meshes[i].distance(ref_mesh)[0])

sorted_indices = np.argsort(distances)
sorted_mesh_files = np.array(mesh_files)[sorted_indices]

batch_size = 200
batches = [sorted_mesh_files[i:i + batch_size] for i in range(0, len(sorted_mesh_files), batch_size)]
print(f"Created {len(batches)} batches of size {len(batches[0])}")

shape_model_dir = groom_directory + 'train_particles/'
    
# Set subjects
subjects = []
for i in range(len(batches[0])):
    subject = sw.Subject()
    
    subject.set_number_of_domains(1)
    
    rel_mesh_file = sw.utils.get_relative_paths([os.getcwd() + "/" + batches[0][i]], groom_directory)
    subject.set_original_filenames(rel_mesh_file)
    subject.set_groomed_filenames(rel_mesh_file)
    
    transform = [train_transforms[sorted_indices[i]].flatten()]
    subject.set_groomed_transforms(transform)
    
    subjects.append(subject)
    
# Set project
project = sw.Project()
project.set_subjects(subjects)
parameters = sw.Parameters()
     
# Create a dictionary for all the parameters required by optimization
parameter_dictionary = {
    "number_of_particles": 256, #
    "use_normals": 0,
    "normal_weight": 10.0,
    "checkpointing_interval": 300,
    "keep_checkpoints": 0,
    "iterations_per_split": 1500, #
    "optimization_iterations": 1500, #
    "starting_regularization": 20, # 200
    "ending_regularization": 0.1, #
    "recompute_regularization_interval": 1,
    "domains_per_shape": 1,
    "relative_weighting": 1, # 1
    "initial_relative_weighting": 0.05, #
    "procrustes": 1,
    "procrustes_interval": 1,
    "procrustes_scaling": 1,
    "save_init_splits": 1,
    "verbosity": 1,
    "multiscale": 1,
    "multiscale_particles": 64,
    "use_shape_statistics_after": 64,
}

# Add param dictionary to spreadsheet
for key in parameter_dictionary:
    parameters.set(key, sw.Variant([parameter_dictionary[key]]))
parameters.set("domain_type", sw.Variant('mesh'))
project.set_parameters("optimize", parameters)
spreadsheet_file = groom_directory + 'train.xlsx'
project.save(spreadsheet_file)

optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
subprocess.check_call(optimize_cmd)


parameter_dictionary["use_landmarks"] = 1
parameter_dictionary["iterations_per_split"] = 0
parameter_dictionary["optimization_iterations"] = 750 # fewer optimization iterations
parameter_dictionary["multiscale"] = 0

for batch_index in range(1, len(batches)):
    print(f'\nBatch {batch_index+1}/{len(batches)}')
    sw.utils.findMeanShape(shape_model_dir)
    mean_shape_path = shape_model_dir + 'meanshape_local.particles'
    
    subjects = []
    for i in range(batch_index):
        for j in range(len(batches[i])):
            subject = sw.Subject()
            
            subject.set_number_of_domains(1)
            
            rel_mesh_file = sw.utils.get_relative_paths([os.getcwd() + "/" + batches[i][j]], groom_directory)
            subject.set_original_filenames(rel_mesh_file)
            subject.set_groomed_filenames(rel_mesh_file)
            
            transform = [train_transforms[sorted_indices[i * batch_size + j]].flatten()]
            subject.set_groomed_transforms(transform)
            
            particle_file = shape_model_dir + os.path.basename(rel_mesh_file[0]).replace(".ply", "_local.particles")
            rel_particle_file = sw.utils.get_relative_paths([os.getcwd() + "/" + particle_file],  groom_directory)
            subject.set_landmarks_filenames(rel_particle_file)
            
            subjects.append(subject)
    
    for j in range(len(batches[batch_index])):
        subject = sw.Subject()
        
        subject.set_number_of_domains(1)
        
        rel_mesh_file = sw.utils.get_relative_paths([os.getcwd() + "/" + batches[batch_index][j]], groom_directory)
        subject.set_original_filenames(rel_mesh_file)
        subject.set_groomed_filenames(rel_mesh_file)
        
        transform = [train_transforms[sorted_indices[batch_index * batch_size + j]].flatten()]
        subject.set_groomed_transforms(transform)
        
        rel_particle_file = sw.utils.get_relative_paths([os.getcwd() + "/" + mean_shape_path], groom_directory)
        subject.set_landmarks_filenames(rel_particle_file)
        
        subjects.append(subject)
    
    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()

    for key in parameter_dictionary:
        parameters.set(key, sw.Variant([parameter_dictionary[key]]))
    parameters.set("domain_type", sw.Variant('mesh'))
    project.set_parameters("optimize", parameters)
    spreadsheet_file = groom_directory + 'train.xlsx'
    project.save(spreadsheet_file)

    optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    subprocess.check_call(optimize_cmd)


project = sw.Project()
project.load(spreadsheet_file)

print('Getting training alignments and procrustes ...')

train_alignments = [[float(x) for x in s.split()] for s in project.get_string_column("alignment_1")]
train_alignments = [np.array(x).reshape(4, 4) for x in train_alignments]
train_alignments = [train_alignments[i] for i in np.argsort(sorted_indices)]

train_procrustes = [[float(x) for x in s.split()] for s in project.get_string_column("procrustes_1")]
train_procrustes = [np.array(x).reshape(4, 4) for x in train_procrustes]
train_procrustes = [train_procrustes[i] for i in np.argsort(sorted_indices)]

train_local_particles = project.get_string_column("local_particles_1")
train_local_particles = [train_local_particles[i] for i in np.argsort(sorted_indices)]

train_world_particles = [x.replace("./", groom_directory) for x in project.get_string_column("world_particles_1")]
train_world_particles = [train_world_particles[i] for i in np.argsort(sorted_indices)]

print('\n' * 3)

# 5. Groom Training Images ##############################################
print('Stage 5. Groom Training Images')

print('Getting reference image and transform ...')
ref_image = all_images['train'][ref_index].copy()
ref_image.resample([1, 1, 1], sw.InterpolationType.Linear)
ref_image.setOrigin(ref_image.origin() - ref_translate)
ref_image.write(f'{groom_directory}reference_image.nrrd')
ref_procrustes = sw.utils.getITKtransform(train_procrustes[ref_index])

train_transforms = []
for train_image, train_align, train_proc in tqdm(zip(all_images['train'], train_alignments, train_procrustes), total=train_size, desc='Transform training images'):
    train_transform = np.matmul(train_proc, train_align)
    train_transforms.append(train_transform)
    train_image.applyTransform(
        train_transform,
        ref_image.origin(),  
        ref_image.dims(),
        ref_image.spacing(), 
        ref_image.coordsys(),
        sw.InterpolationType.NearestNeighbor, 
        meshTransform=True
    )

print('Saving transformed images ...')

train_image_files = sw.utils.save_images(
    f'{groom_directory}train_images/', 
    all_images['train'],
    all_uids['train'], 
    extension='nrrd',
    verbose=False
)

print('\n' * 3)

# 7. Find Test and Validation Transforms and Groom Images ##############################################
print('Stage 7. Find Test and Validation Transforms and Groom Images')

ref_image_file = groom_directory + 'reference_image.nrrd'
ref_image = sw.Image(ref_image_file)
ref_center = ref_image.center()

val_test_image_paths = all_image_paths['val'] + all_image_paths['test']
val_test_mesh_paths = all_mesh_paths['val'] + all_mesh_paths['test']

val_test_image_groomed_paths = []
val_test_transforms = []
for vt_image, vt_uid, vt_mesh_path in tqdm(zip(all_images['val'] + all_images['test'], all_uids['val'] + all_uids['test'], val_test_mesh_paths), total=val_size+test_size, desc='Validation/test transforms'):
    vt_image_groomed_path = val_test_images_directory + vt_uid + '.nrrd'
    val_test_image_groomed_paths.append(vt_image_groomed_path)
    
    translation = ref_center - vt_image.center()
    vt_image.setOrigin(vt_image.origin() + translation)
    transform = np.eye(4)
    transform[:3,-1] += translation
    
    vt_image.write(vt_image_groomed_path)
    itk_translation_transform = DeepSSMUtils.get_image_registration_transform(
        ref_image_file, 
        vt_image_groomed_path,
        transform_type='translation'
    )
    vt_image.applyTransform(
        itk_translation_transform,
        ref_image.origin(),  
        ref_image.dims(),
        ref_image.spacing(), 
        ref_image.coordsys(),
        sw.InterpolationType.NearestNeighbor, 
        meshTransform=False
    )
    vtk_translation_transform = sw.utils.getVTKtransform(itk_translation_transform)
    transform = np.matmul(vtk_translation_transform, transform)

    vt_image.write(vt_image_groomed_path)
    itk_rigid_transform = DeepSSMUtils.get_image_registration_transform(
        ref_image_file, 
        vt_image_groomed_path,
        transform_type='rigid'
    )
    vt_image.applyTransform(
        itk_rigid_transform,
        ref_image.origin(),  
        ref_image.dims(),
        ref_image.spacing(), 
        ref_image.coordsys(),
        sw.InterpolationType.NearestNeighbor, 
        meshTransform=False
    )
    vtk_rigid_transform = sw.utils.getVTKtransform(itk_rigid_transform)
    transform = np.matmul(vtk_rigid_transform, transform)
    
    vt_image.write(vt_image_groomed_path)
    itk_similarity_transform = DeepSSMUtils.get_image_registration_transform(
        ref_image_file, 
        vt_image_groomed_path, 
        transform_type='similarity'
    )
    vt_image.applyTransform(
        itk_similarity_transform,
        ref_image.origin(),  
        ref_image.dims(),
        ref_image.spacing(), 
        ref_image.coordsys(),
        sw.InterpolationType.NearestNeighbor, 
        meshTransform=False
    )
    vtk_similarity_transform = sw.utils.getVTKtransform(itk_similarity_transform)
    transform = np.matmul(vtk_similarity_transform, transform)
    
    vt_image.write(vt_image_groomed_path)
    val_test_transforms.append(transform)

val_image_groomed_paths = val_test_image_groomed_paths[:val_size]
val_transforms = val_test_transforms[:val_size]

test_image_groomed_paths = val_test_image_groomed_paths[val_size:]
test_transforms = val_test_transforms[val_size:]

print('\n' * 3)

# 8. Optimize Validation Particles with Fixed Domains ##############################################
print('Stage 8. Optimize Validation Particles with Fixed Domains')

print('Calculating mean shape ...')

mean_shape = sum([np.loadtxt(particles) for particles in train_world_particles]) / train_size
np.savetxt(f'{groom_directory}meanshape_world.particles', mean_shape)

subjects = []
for train_mesh_path, train_transform, train_local_particle_path in tqdm(zip(all_mesh_paths['train'], train_transforms, train_local_particles), total=train_size, desc='Creating train subjects'):
    subject = sw.Subject()
    subject.set_number_of_domains(1)
    
    rel_mesh_files = sw.utils.get_relative_paths([train_mesh_path], groom_directory)
    rel_groom_files = sw.utils.get_relative_paths([train_mesh_path], groom_directory)
    
    subject.set_original_filenames(rel_mesh_files)
    subject.set_groomed_filenames(rel_groom_files)
    transform = [train_transform.flatten()]
    
    subject.set_groomed_transforms(transform)
    subject.set_landmarks_filenames([train_local_particle_path])
    subject.set_extra_values({"fixed": "yes"})
    subjects.append(subject)

for val_mesh_path, val_transform, val_uid in tqdm(zip(all_mesh_paths['val'], val_transforms, all_uids['val']), total=val_size, desc='Creating validation subjects'):
    subject = sw.Subject()
    subject.set_number_of_domains(1)
    
    rel_mesh_files = sw.utils.get_relative_paths([val_mesh_path], groom_directory)
    rel_groom_files = sw.utils.get_relative_paths([val_mesh_path], groom_directory)
    
    initial_particles = sw.utils.transformParticles(mean_shape, val_transform, inverse=True)
    initial_particle_file = f'{val_paticle_directory}{val_uid}_local.particles'
    np.savetxt(initial_particle_file, initial_particles)
    
    rel_particle_files = sw.utils.get_relative_paths([initial_particle_file], groom_directory)
    subject.set_original_filenames(rel_mesh_files)
    subject.set_groomed_filenames(rel_groom_files)
    transform = [val_transform.flatten()]
    subject.set_groomed_transforms(transform)
    subject.set_landmarks_filenames(rel_particle_files)
    subject.set_extra_values({"fixed": "no"})
    subjects.append(subject)

project = sw.Project()
project.set_subjects(subjects)

print('Setting parameters ...')

parameters = sw.Parameters()

parameter_dictionary["multiscale"] = 0
parameter_dictionary["procrustes"] = 0
parameter_dictionary["procrustes_interval"] = 0
parameter_dictionary["procrustes_scaling"] = 0
parameter_dictionary["use_landmarks"] = 1
parameter_dictionary["use_fixed_subjects"] = 1
parameter_dictionary["narrow_band"] = 1e10
parameter_dictionary["fixed_subjects_column"] = "fixed"
parameter_dictionary["fixed_subjects_choice"] = "yes"
for key, value in parameter_dictionary.items():
    parameters.set(key, sw.Variant(value))
    
project.set_parameters("optimize", parameters)

# Set studio parameters
studio_dictionary = {
    "show_landmarks": 0,
    "tool_state": "analysis"
}
studio_parameters = sw.Parameters()
for key, value in studio_dictionary.items():
    studio_parameters.set(key, sw.Variant(value))
project.set_parameters("studio", studio_parameters)

spreadsheet_file = groom_directory + "validation.xlsx"
project.save(spreadsheet_file)

print('Optimizing validation particles ...')

optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
subprocess.check_call(optimize_cmd)

print('Analyze command:')
print('ShapeWorksStudio ' + spreadsheet_file)

val_world_particles = [f'{val_paticle_directory}{val_uid}_world.particles' for val_uid in all_uids['val']]

print('\n' * 3)

# 6. Augment Data ##############################################
print('Stage 6. Augment Data')

for explained_var in [99, 97, 95, 90, 80, 70]:
    print(f'Augmentation: {explained_var}% variance explained')
    aug_directory = f'{deepssm_directory}augmentation{explained_var}/'
    loader_directory = f'{deepssm_directory}torch_loaders{explained_var}/'
    for directory in [aug_directory, loader_directory]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    num_samples = 1
    percent_variability = explained_var / 100
    sampler_type = 'kde'
    processes = 8

    embedded_dim = DataAugmentationUtils.runDataAugmentation(
        out_dir=aug_directory,
        img_list=train_image_files, 
        local_point_list=sorted(glob(f'{groom_directory}train_particles/*_local.particles')), 
        num_samples=num_samples, 
        #num_dim,
        percent_variability=percent_variability, 
        sampler_type=sampler_type,
        #mixture_num,
        processes=processes, 
        world_point_list=sorted(glob(f'{groom_directory}train_particles/*_world.particles')),
    )
    aug_data_csv = aug_directory + 'TotalData.csv'

    print(f'PCA dimensions retained: {embedded_dim}')

    print('Getting partial world data ...')
    df = pd.read_csv(aug_data_csv, header=None)[:-num_samples]
    for i in tqdm(range(df.shape[0])):
        df.at[i, 1] = df.at[i, 1].replace('local', 'world')
    df.to_csv(
        aug_directory + 'PartialWorldData.csv', 
        header=False, 
        index=False
    )

    print('Getting partial world original data ...')
    for i in tqdm(range(df.shape[0])):
        df.at[i, 0] = df.at[i, 0].replace(f'_cleaned/{project_name}/groomed/train_images', '')
    df.to_csv(
        aug_directory + 'PartialWorldOrigData.csv', 
        header=False, 
        index=False
    )

print('Program Stage 1~8 Completed')
