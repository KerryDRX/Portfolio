# ====================================================================
# Two main tasks of this script:
#
# Given the paths to the json files which store the picture ids of
# training and test images, the subsets of these id lists are generated
# and saved to ensure all the images in the training and test sets
# have its tag term specified in word2vec.
#   Original training image id path: '../dataset/train_pic_id.json'
#   Original test image id path: '../dataset/test_pic_id.json'
#   Updated training image id path: '../dataset/train_id.json'
#   Updated test image id path: '../dataset/test_id.json'
#
# The tag term vectors are stored in npy files. Each npy file has
# its name as xxxxxxx.npy, in which xxxxxxx denotes the image id.
# The file content is an numpy array representing the word vector
# of the tag term of that image. Files are saved in the following
# two directories:
#   '../dataset/txt_feature_train'
#   '../dataset/txt_feature_test'
# ====================================================================
import numpy as np
import json
import os
# ====================================================================
# Step 1: Load word2vec file and construct word2vec dictionary.
path_word2vec = r'../dataset/word2vec.utf8'

# Load file
print('Loading word2vec.utf8 ...')
with open(path_word2vec, 'r', encoding='utf8') as fp:
    buffer = fp.read()
print('File word2vec.utf8 loaded.')

# Construct dictionary
print('Constructing word2vec dictionary ...')
lines = buffer.split('\n')
word2vec = {}
# word2vec maps each word in the dictionary to its word vector
for line in lines:
    try:
        word = line.split('\t')[0]  # Get word
        vec = line.split('\t')[2]  # Get vector
        vec = [float(val) for val in vec.split(' ')]  # Transform the vector from string to list
        word2vec[word] = vec  # Store the mapping into dictionary
    except:
        continue
print('Number of words in the dictionary: {}'.format(len(word2vec)))
# ====================================================================
# Step 2: Load image_info file and construct an image information list.
path_image_info = r'../dataset/image_info.json'

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
# ====================================================================
# Step 3: Load the training image ids and test image ids.
path_img_id_train = r'../dataset/train_pic_id.json'
path_img_id_test = r'../dataset/test_pic_id.json'

print('Loading img_id_train.json ...')
with open(path_img_id_train, 'rb') as fp:
    buffer = fp.read()
# img_id_train stores the picture ids of all the training images
img_id_train = json.loads(str(buffer, encoding='utf8'))
print('Number of training ids: {}'.format(len(img_id_train)))

print('Loading test_img_id.json ...')
with open(path_img_id_test, 'rb') as fp:
    buffer = fp.read()
# img_id_test stores the picture ids of all the test images
img_id_test = json.loads(str(buffer, encoding='utf8'))
print('Number of test ids: {}'.format(len(img_id_test)))
# ====================================================================
# Step 4: Save training/test image tag term vectors and ids.

pic_id_train = []  # Picture id of training images
pic_id_test = []  # Picture id of test images
path_txt_feature_train = r'../dataset/txt_feature_train'
path_txt_feature_test = r'../dataset/txt_feature_test'

# Create directories if not exist
if not os.path.exists(path_txt_feature_train):
    os.mkdir(path_txt_feature_train)
if not os.path.exists(path_txt_feature_test):
    os.mkdir(path_txt_feature_test)

# Save image tag term vectors
print('Saving image tag term vectors ...')
for info_dict in image_info:
    pic_id = info_dict['pic_id']  # Picture id of this image
    tag = info_dict['tags_term']  # Tag term of this image
    # Tag term has to be in the word2vec dictionary
    if tag not in word2vec:
        continue
    # Save training image tag vector
    if pic_id in img_id_train:
        try:
            buffer = word2vec[tag]
        except:
            print('Error saving image {} tag term vector.'.format(pic_id))
            continue
        save_path = os.path.join(path_txt_feature_train, '{}.npy'.format(pic_id))
        np.save(save_path, np.array(buffer))
        pic_id_train.append(pic_id)  # Add picture id to training image id list
    # Save test image tag vector
    if pic_id in img_id_test:
        try:
            buffer = word2vec[tag]
        except:
            print('Error saving image {} tag term vector.'.format(pic_id))
            continue
        save_path = os.path.join(path_txt_feature_test, '{}.npy'.format(pic_id))
        np.save(save_path, np.array(buffer))
        pic_id_test.append(pic_id)  # Add picture id to test image id list
print('Training sample size:', len(pic_id_train))
print('Test sample size:', len(pic_id_test))

# Save image ids
# These exclude pictures whose tag terms are not in word2vec
path_img_id_train = r'../dataset/train_id.json'
path_img_id_test = r'../dataset/test_id.json'

print('Saving training & test ids ...')
with open(path_img_id_train, 'w', encoding='utf8') as fp:
    json.dump(pic_id_train, fp, ensure_ascii=False)
with open(path_img_id_test, 'w', encoding='utf8') as fp:
    json.dump(pic_id_test, fp, ensure_ascii=False)
print('Image ids saved.')
