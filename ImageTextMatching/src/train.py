# ====================================================================
# The main task of this script is to train the model for image text
# matching.
#
# The model takes two vectors as input: an image feature
# vector and a tag term word vector. The model itself consists of
# two parts. The first part transforms the image vector into an
# embedding, and the second part transforms the word vector into
# another embedding. The two parts are independent of each other
# and do not interact during the training process.
#
# The output of the model is two embeddings, one for the image and
# one for the word. If the image and the word matches (i.e., the
# word is the tag term of that image), than the corresponding two
# embeddings should be similar (with a high cosine similarity).
#
# The best model will be saved to
#   ../results/best_model.pkl
# ====================================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Image ids
path_img_id_train = r'../dataset/train_pic_id.json'
path_img_id_test = r'../dataset/test_pic_id.json'
# Image feature vectors
path_pic_feature_train = r'../dataset/pic_feature_train'
path_pic_feature_test = r'../dataset/pic_feature_test'
# Image tag term vectors
path_txt_feature_train = r'../dataset/txt_feature_train'
path_txt_feature_test = r'../dataset/txt_feature_test'
# File to save the best model
path_best_model = r'../results/best_model.pkl'
# File to save the model training result plot
path_training_stats = r'../results/train.pkl'

# Specify device: GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format('GPU' if torch.cuda.is_available() else 'CPU'))

# Lists to store training and test image feature vectors
# and tag term feature vectors
train_pics, train_txts = [], []
test_pics, test_txts = [], []

# Load training and test image ids into train_ids and test_ids
with open(path_img_id_train, 'r') as fp:
    train_ids = json.load(fp)
print('Training sample size:', len(train_ids))
with open(path_img_id_test, 'r') as fp:
    test_ids = json.load(fp)
print('Test sample size:', len(test_ids))

# Store training and test image feature vectors and tag term vectors
# into train_pics, train_txts, test_pics, test_txts
for train_id in train_ids:
    path_pic = os.path.join(path_pic_feature_train, '{}.npy'.format(train_id))
    path_txt = os.path.join(path_txt_feature_train, '{}.npy'.format(train_id))
    pic = np.load(path_pic).reshape(2048)
    txt = np.load(path_txt)
    train_pics.append(torch.tensor(pic))
    train_txts.append(torch.tensor(txt).float())

for test_id in test_ids:
    path_pic = os.path.join(path_pic_feature_test, '{}.npy'.format(test_id))
    path_txt = os.path.join(path_txt_feature_test, '{}.npy'.format(test_id))
    pic = np.load(path_pic).reshape(2048)
    txt = np.load(path_txt)
    test_pics.append(torch.tensor(pic))
    test_txts.append(torch.tensor(txt).float())


# Dataset class for loading training data
class ImageTextDataset(Dataset):
    def __init__(self):
        self.x_data = train_pics
        self.y_data = train_txts
        self.length = len(train_ids)

    def __getitem__(self, index):
        # The right picture for the specified index
        right_pic = self.x_data[index]
        # The wrong picture for the specified index
        # (for loss calculation purpose)
        wrong_pic = self.x_data[(index + 3) % self.length]
        # The right tag for the specified index
        right_txt = self.y_data[index]
        return right_pic, wrong_pic, right_txt

    def __len__(self):
        return self.length


# Self-defined neural network model for image text matching
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


# Train the model and save the best model
def train(batch_size, learning_rate, num_epochs, margin):
    # Loss function: hinge loss
    # Margin specified in parameter of train()
    class HingeLoss(nn.Module):
        def __init__(self):
            super(HingeLoss, self).__init__()
            self.margin = torch.tensor(margin).to(device)

        def forward(self, y, y_hat):
            zero_tensor = Variable(torch.tensor(0.0), requires_grad=True).to(device)
            return max(zero_tensor, self.margin - y + y_hat)

    # Print training settings
    print('Batch Size: {}'.format(batch_size))
    print('Learning Rate: {}'.format(learning_rate))
    print('Number of Training Epochs: {}'.format(num_epochs))
    print('Margin of Hinge Loss Function: {}'.format(margin))

    # Load training data
    training_data = ImageTextDataset()
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # Define loss
    loss_function = HingeLoss()
    # Model initialization
    model = NeuralNetwork()
    model = model.to(device)
    # Model optimizer: SGD
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Best accuracy among all epochs
    best_acc = float('-inf')
    # Lists of epoch accuracy and loss
    accuracies = []
    train_losses = []
    test_losses = []
    for i in range(num_epochs):
        running_loss = torch.tensor(0.0)
        for index, data in enumerate(train_loader):
            # Load data from DataLoader
            # right_pic: the right feature vector of the image
            # wrong_pic: the wrong feature vector of the image
            # right_txt: the right feature vector of the image tag
            right_pic, wrong_pic, right_txt = data
            batch_size = right_pic.size(0)
            right_pic = Variable(right_pic, requires_grad=True)
            wrong_pic = Variable(wrong_pic, requires_grad=True)
            right_txt = Variable(right_txt, requires_grad=True)
            right_pic = right_pic.to(device)
            wrong_pic = wrong_pic.to(device)
            right_txt = right_txt.to(device)

            optimizer.zero_grad()
            # pic1, txt1: the image and tag embeddings of the right image & right tag
            # pic2, txt2: the image and tag embeddings of the wrong image & right tag
            pic1, txt1 = model(right_pic, right_txt)
            pic2, txt2 = model(wrong_pic, right_txt)
            # Calculate cosine similarities
            y = torch.cosine_similarity(pic1, txt1, dim=1)
            y_hat = torch.cosine_similarity(pic2, txt2, dim=1)
            # Loss = margin - (y - y_hat)
            # which means the larger the difference between y and y_hat
            # (i.e., the right match has a much higher cosine similarity
            # than the wrong match), the better the model performance is
            loss = loss_function(y.sum() / batch_size, y_hat.sum() / batch_size)
            loss.backward()
            # Accumulate the loss
            running_loss += loss * batch_size
            optimizer.step()
        training_loss = running_loss / len(train_ids)
        train_losses.append(training_loss)
        print('Epoch {} training loss: {:.6f}'.format(i + 1, training_loss))

        # Calculate test loss and accuracy
        running_acc = 0.0
        test_loss = torch.tensor(0.0)
        for j in range(len(test_pics)):
            # Retrieve the right and wrong pics and txt
            right_pic = test_pics[j].to(device)
            wrong_pic = test_pics[(j + 3) % len(test_pics)].to(device)
            right_txt = test_txts[j].to(device)
            # Calculate cosine similarities
            pic1, txt1 = model(right_pic, right_txt)
            pic2, txt2 = model(wrong_pic, right_txt)
            y = torch.cosine_similarity(pic1, txt1, dim=0)
            y_hat = torch.cosine_similarity(pic2, txt2, dim=0)
            loss = loss_function(y.sum() / batch_size, y_hat.sum() / batch_size)
            test_loss += loss * batch_size
            # Accurate if a right match results in a higher similarity than a wrong match
            if y > y_hat:
                running_acc += 1.0
        print('Epoch {} test loss: {:.6f}'.format(i + 1, test_loss / len(test_pics)))
        print('Epoch {} test accuracy: {:.6f}'.format(i + 1, running_acc / len(test_pics)))
        test_losses.append(test_loss / len(test_pics))
        accuracies.append(running_acc / len(test_pics))
        # Save the current best accuracy and best model
        if running_acc > best_acc:
            best_acc = running_acc
            torch.save(model, path_best_model)
    print('Best Accuracy: {}%'.format(100 * best_acc / len(test_pics)))

    return accuracies, train_losses, test_losses


# Specify parameters, initiate model training, and visualize the training statistics
def main():
    # Set model training parameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 20
    margin = 0.3
    # Train the model
    accuracies, train_losses, test_losses = train(batch_size, learning_rate, num_epochs, margin)
    # Draw the plot
    plt.xlabel('Training Epochs')
    plt.title('Model Accuracy')
    plt.plot(range(1, num_epochs+1), accuracies, label='Test Accuracy')
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(path_training_stats)


if __name__ == '__main__':
    main()
