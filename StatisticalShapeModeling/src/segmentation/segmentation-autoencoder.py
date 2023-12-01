import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import gc
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import json
import scipy.ndimage as ndimage
import nrrd
import torchio as tio
import monai
import nibabel as nib
import time

city = 'Beijing_Zang2'

modes = ['train', 'test']
total_size = 197
train_size, test_size = 158, 39

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

seed = 0
random_state = np.random.RandomState(seed=seed)
perm = random_state.permutation(total_size)
perm = {
    'train': perm[:train_size],
    'test': perm[train_size:train_size+test_size],
}

def get_subjects(mode):
    subjects = []
    image_paths = [sorted(glob(f'{image_dir}/*.nii.gz'))[i] for i in perm[mode]]
    for image_path in image_paths:
        subject = tio.Subject(segm=tio.LabelMap(image_path))
        subjects.append(subject)
    return subjects

def get_transform():
    resample = tio.Compose([
        tio.Resample(2),
        tio.CropOrPad((128,128,128)),
    ])
    spatial = tio.Compose([
        tio.RandomAffine(degrees=3, translation=0.1),
        # tio.RandomAffine(scales=(0.8, 1.2)),
        # tio.OneOf({
        #     tio.RandomAffine(degrees=(0, 0, 0, 0, 0, 0)): 1,
        #     tio.RandomAffine(degrees=(0, 0, 0, 0, 90, 90)): 1,
        #     tio.RandomAffine(degrees=(0, 0, 0, 0, 180, 180)): 1,
        #     tio.RandomAffine(degrees=(0, 0, 0, 0, 270, 270)): 1,
        # })
    ])
    remapping = tio.RemapLabels({
        i: (
            1 if i in {1,2,21,22,23,24}
            # 1 if i in {21, 23}                            
            # else 2 if i in {22, 24}  
            # else 3 if i == 1
            # else 4 if i == 2 
            else 0  
        ) for i in range(139)
    })

    # onehot = tio.OneHot(num_classes=1)
    transform = {
        'train': tio.Compose([
            resample,
            spatial,
            remapping,
            # onehot,
        ]),
        'test': tio.Compose([
            resample,
            remapping,
            # onehot,
        ]),
    }
    return transform

def get_dataloader(transform):
    dataloader = dict()
    for mode in modes:
        dataloader[mode] = torch.utils.data.DataLoader(
            tio.SubjectsDataset(
                subjects[mode], 
                transform=transform[mode]
            ),
            batch_size=4, 
            num_workers=os.cpu_count(),
            shuffle=(mode == 'train'),
        )
    return dataloader

def clean(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

image_dir = f'../dataset/{city}/Segmentation'
model_dir = f'../results/SCAE_vent_all_new/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

subjects = {mode: get_subjects(mode) for mode in modes}
transform = get_transform()
dataloaders = get_dataloader(transform)

def train(model, dataloaders, num_epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)
    loss_fn = monai.losses.DiceLoss(squared_pred=True).to(device)
    metric = monai.metrics.DiceMetric(reduction='mean_batch')

    t0 = time.time()
    best_val_dsc = 0
    tol = 0
    tol50 = 0
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        for mode in modes:
            if mode == 'train':
                model.train()
            else:
                model.eval()
            
            losses = []
            for subject in dataloaders[mode]:
                image = subject['segm'][tio.DATA].to(device).float()
                
                prob = model(image)
                loss = loss_fn(prob, image)
                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                losses.append(loss.item())
                metric((prob > 0.5).float(), image)
            
            mean_loss = np.mean(losses)
            print(f'{mode} loss: {mean_loss}')
            mean_dsc = metric.aggregate().tolist()[0]
            metric.reset()
            print(f'{mode} DSC: {mean_dsc}')

        if mean_dsc >= best_val_dsc:
            best_val_dsc = mean_dsc
            best_epoch = epoch
            torch.save(model.state_dict(), f'{model_dir}/best_autoencoder.torch')
            tol = 0
            tol50 = 0
        else:
            tol += 1
            tol50 += 1
        print(f'Best test DSC: {best_val_dsc}')
        
        if tol == 10:
            scheduler.step()
            print('Validation DSC stopped to improve for 10 epochs (LR /= 5).')
            tol = 0
        
        time_elapsed = time.time() - t0
        print(f'Time: {time_elapsed}\n')
        t0 = time.time()

        if tol50 == 50:
            print('Validation DSC stopped to improve for 50 epochs. Training terminated.')
            break

    print(f"Best model after epoch {best_epoch}. Best test DSC: {best_val_dsc}")

def init_model():
    model = torch.nn.Sequential(
        monai.networks.nets.AutoEncoder(
            spatial_dims=3, in_channels=1, out_channels=1, 
            channels=(16,32,64,96,128,196,256,512), strides=(1,2,2,2,2,2,2,2),
            norm=monai.networks.layers.Norm.BATCH,
            act=monai.networks.layers.Act.LEAKYRELU,
        ),
        torch.nn.Sigmoid(),
    )
    return model

if 'model' in globals(): clean(model)
model = init_model().to(device)
train(model=model, dataloaders=dataloaders, num_epochs=1000, learning_rate=5e-4)