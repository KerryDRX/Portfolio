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
import torchio as tio
import monai
import nibabel as nib
import time

city = 'Beijing_Zang2'

modes = ['train', 'test']
total_size = 197
train_size, test_size = 158, 39

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

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

def get_transform(GM_or_WM):
    resample = tio.Compose([
        tio.Resample(2),
        tio.CropOrPad((96,128,128)),
    ])
    spatial = tio.Compose([
        tio.RandomAffine(translation=1),
        # tio.RandomAffine(scales=(0.8, 1.2)),
        # tio.OneOf({
        #     tio.RandomAffine(degrees=(0, 0, 0, 0, 0, 0)): 1,
        #     tio.RandomAffine(degrees=(0, 0, 0, 0, 90, 90)): 1,
        #     tio.RandomAffine(degrees=(0, 0, 0, 0, 180, 180)): 1,
        #     tio.RandomAffine(degrees=(0, 0, 0, 0, 270, 270)): 1,
        # })
    ])
    remapping = dict()
    if GM_or_WM == 'GM':
        for i in range(139):
            remapping[i] = 1 if (3<=i<=11 or 19<=i<=20 or 25<=i<=32 or 35<=i) else 0
    elif GM_or_WM == 'WM':
        for i in range(139):
            remapping[i] = 1 if i in {12, 13, 16, 17} else 0
        # remapping[i] = 1 if (3<=i<=11 or 19<=i<=20 or 25<=i<=32 or 35<=i) else 2 if i in {12, 13, 16, 17} else 0
    
    remapping = tio.RemapLabels(remapping)
    transform = {
        'train': tio.Compose([
            resample,
            spatial,
            remapping,
        ]),
        'test': tio.Compose([
            resample,
            remapping,
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
            batch_size=1, 
            num_workers=os.cpu_count(),
            shuffle=(mode == 'train'),
        )
    return dataloader

def clean(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

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
                
                pred = model(image)
                loss = loss_fn(pred, image)
                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                losses.append(loss.item())
                metric((pred > 0.5).float(), image)

            print(f'{mode} loss: {np.mean(losses)}')
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

def convolution(in_channels, out_channels, stride):
    return torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)

def deconvolution(in_channels, out_channels, stride):
    return torch.nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1)

def normalization(channel):
    return torch.nn.BatchNorm3d(channel)

def activation():
    return torch.nn.PReLU()

def pooling(kernel_size):
    return torch.nn.MaxPool3d(kernel_size=kernel_size)

def upsampling(scale_factor):
    return torch.nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)

class Model(torch.nn.Module):
    def __init__(self, channels):
        super(Model, self).__init__()
        self.encoder = torch.nn.Sequential(
            convolution(in_channels=1, out_channels=channels[0], stride=2),
            normalization(channels[0]),
            activation(),
            
            convolution(in_channels=channels[0], out_channels=channels[1], stride=2),
            normalization(channels[1]),
            activation(),
            
            convolution(in_channels=channels[1], out_channels=channels[2], stride=2),
            normalization(channels[2]),
            activation(),

            convolution(in_channels=channels[2], out_channels=channels[3], stride=2),
            normalization(channels[3]),
            activation(),
        )
        self.decoder = torch.nn.Sequential(
            deconvolution(in_channels=channels[3], out_channels=channels[2], stride=2),
            normalization(channels[2]),
            activation(),
            
            deconvolution(in_channels=channels[2], out_channels=channels[1], stride=2),
            normalization(channels[1]),
            activation(),
            
            deconvolution(in_channels=channels[1], out_channels=channels[0], stride=2),
            normalization(channels[0]),
            activation(),
            
            deconvolution(in_channels=channels[0], out_channels=1, stride=2),
            normalization(1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

image_dir = f'../dataset/{city}/Segmentation'
model_dir = f'../results/SCAE_WM_temp2/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

subjects = {mode: get_subjects(mode) for mode in modes}
transform = get_transform('WM')
dataloaders = get_dataloader(transform)

model = Model(channels=[128,256,512,1024]).to(device)
train(model=model, dataloaders=dataloaders, num_epochs=1000, learning_rate=5e-4)