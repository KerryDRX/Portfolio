# ====================================================================
# This script runs the image search engine. The web app asks user for
# a tag term, and displays the images corresponds to the term given
# by the user. The images are displayed in an order such that the images
# with a higher similarity with the tag term will be put at the top.
# ====================================================================
import os
import json
import shutil
import PIL.Image as Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from flask import Flask, redirect, url_for, request
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
# Step 2: Load tag2picid dictionary (inverted index).
path_tag2picid = r'../dataset/tag2picid.json'

with open(path_tag2picid, 'rb') as f:
    buffer = f.read()
tag2picid = json.loads(str(buffer, encoding='utf8'))
# ====================================================================
# Step 3: Create directory for temporary picture storage.
pic_folder = r'../dataset/pics'
static_pic_folder = r'./static'

if not os.path.exists(static_pic_folder):
    os.mkdir(static_pic_folder)
# ====================================================================
# Step 4: Set up ResNet50 model for image feature extraction.
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == 'fc':
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

resnet50 = models.resnet50(pretrained=True)
resnet50.eval()
extract_list = ['avgpool']
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
# ====================================================================
# Step 5: Load the trained model for image text matching.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        pic_feature_size = 2048
        hidden_layer_size = 512
        txt_feature_size = 200
        embedding_size = 128
        self.pic_fc1 = nn.Linear(pic_feature_size, hidden_layer_size)
        self.pic_fc2 = nn.Linear(hidden_layer_size, embedding_size)
        self.txt_fc1 = nn.Linear(txt_feature_size, embedding_size)

    def forward(self, pic_feature, txt_feature):
        pic_embedding = self.pic_fc2(self.pic_fc1(pic_feature))
        txt_embedding = self.txt_fc1(txt_feature)
        return pic_embedding, txt_embedding

path_best_model = r'../results/best_model.pkl'
model = NeuralNetwork()
model = torch.load(path_best_model)
# ====================================================================
# Step 6: Build the image search engine.
app = Flask(__name__)

# Success: Images with the input tag term found.
# Now sort the images with respect to their similarity with the tag term.
# Images with higher similarity with the tag will be displayed first.
@app.route('/success/<tag>.html')
def success(tag):
    link = '<p align=center>HERE ARE THE PICTURES<br/><br/>'
    # List of picture ids that match the tag term
    pics = []
    # The dictionary to map picture id to the cosine similarity score
    # of that picture with the tag term
    picid2score = {}
    for pic_id in tag2picid[tag]:
        old_path = os.path.join(pic_folder, '{}.jpg'.format(pic_id))
        new_path = os.path.join(static_pic_folder, '{}.jpg'.format(pic_id))
        if not os.path.isfile(old_path):
            continue
        # Copy the image from database to static folder
        shutil.copyfile(old_path, new_path)
        try:
            img = transform(Image.open(new_path))
        except:
            continue
        # Get flattened image vector
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
        # Get image feature by pretrained ResNet50
        img_feature = FeatureExtractor(resnet50, extract_list)
        try:
            img_feature = img_feature(x)[0].detach().numpy()
        except Exception as err:
            print('Error in image feature extraction: {}'.format(err))
        img_feature = torch.tensor(img_feature).reshape(2048)
        # Get word vector of the tag term
        txt_feature = np.array(word2vec[tag])
        txt_feature = torch.tensor(txt_feature).float()
        # Calculate image and word embeddings from their features
        pic_embedding, txt_embedding = model(img_feature, txt_feature)
        score = nn.CosineSimilarity(dim=0)(pic_embedding, txt_embedding)
        picid2score[pic_id] = score
        pics.append(pic_id)
    # Sort the dictionary by descending order of the similarity score
    sorted_picid2score = sorted(picid2score, key=picid2score.__getitem__, reverse=True)
    # Display the images with a higher similarity with the tag term first
    for key in sorted_picid2score:
        img_str = '<img src=\'/static/{}.jpg\'/>'.format(key)
        link += img_str
    return link

# Failure: show the failure message.
@app.route('/fail/<tag>.html')
def fail(tag):
    return 'No picture matching your keyword: {}'.format(tag)

# Search: get the list of picture ids with the input tag term
@app.route('/search/', methods=['GET'])
def search_engine():
    # Get tag term to search
    tag_term = request.args.get('tag')
    if tag_term in tag2picid:
        # Image(s) found
        return redirect(url_for('success', tag=tag_term))
    else:
        # No matching images
        return redirect(url_for('fail', tag=tag_term))

# Run the web application
if __name__ == '__main__':
    app.run(debug = False)
