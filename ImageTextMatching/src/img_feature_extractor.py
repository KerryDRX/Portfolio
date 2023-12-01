# ====================================================================
# This script loads images and uses ResNet50 to extract image features, 
# and store the feature vectors and image ids into files.
# ====================================================================
import json
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import os
from random import sample
import PIL.Image as Image
from multiprocessing import Pool
# ====================================================================
# Step 1: Create paths if not exist. Specify files to store image ids.
# Directory to store training and test image features:
path_pic_feature_train = r'../dataset/pic_feature_train'
path_pic_feature_test = r'../dataset/pic_feature_test'
# JSON files to store training and test image ids:
path_img_id_train = r'../dataset/train_pic_id.json'
path_img_id_test = r'../dataset/test_pic_id.json'

if not os.path.exists(path_pic_feature_train):
    os.mkdir(path_pic_feature_train)
if not os.path.exists(path_pic_feature_test):
    os.mkdir(path_pic_feature_test)
# ====================================================================
# Step 2: Load image_info file and construct an image information list
# The list only keeps the image info dictionaries for images that exist
# in the dataset:
#   ../dataset/pics'
path_image_info = r'../dataset/image_info.json'
path_pic_folder = r'../dataset/pics'

print('Loading image_info.json ...')
with open(path_image_info, 'rb') as fp:
    buffer = fp.read()
print('File image_info.json loaded.')

print('Constructing image_list ...')
lines = str(buffer, encoding='utf8').split('\n')
image_info = []
# image_info is a list of image information
# Each entry in image_info is a dictionary containing pic_id, tags_term of an image.
for line in lines:
    try:
        image_info.append(json.loads(line))
    except:
        continue
print('Number of images: {}'.format(len(image_info)))

# Image info sifting: keep those image info dictionaries only if the image exists
# in the image dataset. The directory that stores all the images is
#   ../dataset/pics
image_info_sifted = [
    info_dict for info_dict in image_info
    if os.path.isfile(os.path.join(path_pic_folder, '{}.jpg'.format(info_dict['pic_id'])))
]
print('Number of sifted images:'.format(len(image_info_sifted)))
# ====================================================================
# Step 3: Train test split. Randomly select training and test samples
# from the sifted image info list.
size_train, size_test = 100000, 20000

size = size_train + size_test
image_info_train_test = sample(image_info_sifted, size)
image_info_train = image_info_train_test[0:size_train]
image_info_test = image_info_train_test[size_train:size]
print('Total sample size:'.format(len(image_info_train_test)))
print('Training sample size:'.format(len(image_info_train)))
print('Test sample size:'.format(len(image_info_test)))
# ====================================================================
# Step 4: Model construction.
# Specify device: GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format('GPU' if torch.cuda.is_available() else 'CPU'))

# Select pretrained resnet50 as the image feature extractor model
resnet50 = models.resnet50(pretrained=True)
resnet50 = resnet50.to(device)
resnet50.eval()
# Layers to extractor output feature vector
extract_list = ['avgpool']
# Image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
# ====================================================================


# ====================================================================
# This class defines an image feature extractor. The submodule refers
# to the model, and output_layers is the list of layers from which the
# output feature is extracted.
# ====================================================================
class ImageFeatureExtractor(nn.Module):
    def __init__(self, submodule, output_layers):
        super(ImageFeatureExtractor, self).__init__()
        self.submodule = submodule
        self.output_layers = output_layers

    def forward(self, x):
        x = x.to(device)
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == 'fc':
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.output_layers:
                outputs.append(x)
        return outputs


# ====================================================================
# This function uses ImageFeatureExtractor to extracts feature from
# the training image specified in info_dict, and stores the feature
# vector into the directory:
#   ../dataset/pic_feature_train/xxxxxxx.npy
# in which xxxxxxx is the picture id of that image.
# ====================================================================
def extract_picid_feature_train(info_dict):
    # Get picture id
    pic_id = info_dict['pic_id']
    path_img = os.path.join(path_pic_folder, '{}.jpg'.format(pic_id))
    path_pic_feature = os.path.join(path_pic_feature_train, '{}.npy'.format(pic_id))
    # Get image
    try:
        img = transform(Image.open(path_img))
    except:
        return None
    # Pass the image to the model and calculate the feature vector
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    extract_result = ImageFeatureExtractor(resnet50, extract_list)
    try:
        save_buffer = extract_result(x)[0].detach().numpy()
    except Exception:
        return None
    # Save the feature vector
    np.save(path_pic_feature, save_buffer)
    return pic_id


# ====================================================================
# This function uses ImageFeatureExtractor to extracts feature from
# the test image specified in info_dict, and stores the feature
# vector into the directory:
#   ../dataset/pic_feature_test/xxxxxxx.npy
# in which xxxxxxx is the picture id of that image.
# ====================================================================
def extract_picid_feature_test(info_dict):
    # Get picture id
    pic_id = info_dict['pic_id']
    path_img = os.path.join(path_pic_folder, '{}.jpg'.format(pic_id))
    path_pic_feature = os.path.join(path_pic_feature_test, '{}.npy'.format(pic_id))
    # Get image
    try:
        img = transform(Image.open(path_img))
    except:
        return None
    # Pass the image to the model and calculate the feature vector
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    extract_result = ImageFeatureExtractor(resnet50, extract_list)
    try:
        save_buffer = extract_result(x)[0].detach().numpy()
    except Exception:
        return None
    # Save the feature vector
    np.save(path_pic_feature, save_buffer)
    return pic_id


# ====================================================================
# The main function uses multiple processes to extract image features
# and stores them into target files:
#   ../dataset/train_pic_id.json
#   ../dataset/test_pic_id.json
# ====================================================================
def main():
    # Start multiple processes to extract image features
    print('Extracting image features ...')
    pool = Pool(processes=4)
    train_ids = pool.map(extract_picid_feature_train, image_info_train)
    test_ids = pool.map(extract_picid_feature_test, image_info_test)
    pool.close()
    pool.join()
    print('Feature extraction completed.')

    # Save the training and test image ids into target files
    train_ids = [train_id for train_id in train_ids if train_id is not None]
    test_ids = [test_id for test_id in test_ids if test_id is not None]
    with open(path_img_id_train, 'w', encoding='utf8') as fp:
        json.dump(train_ids, fp, ensure_ascii=False)
    with open(path_img_id_test, 'w', encoding='utf8') as fp:
        json.dump(test_ids, fp, ensure_ascii=False)
    print('Number of training images:'.format(len(train_ids)))
    print('Number of test images:'.format(len(test_ids)))


if __name__ == '__main__':
    main()
