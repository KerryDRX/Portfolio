<div align="center">

  # Image Search Engine

</div>

# Project Description
This project constructs an image search engine using deep learning. Millions of images and their corresponding tag terms are crawled from various websites for model training and testing. The image and text features are extracted using developed models, and are matched by a self-defined neural network. The ultimate model is able to sort the target images in descending order of their feature similarity with the tag term being searched. Finally, a web image search engine application is developed to allow users to search for figures using their browser.

# Data Source
A total of 1.7 million images are crawled from websites to train and evaluate our model. Each image is annotated with a tag term, using which the users can search for the image they want.

# Models:
- word2vec: for text feature extraction.
- ResNet50: for image feature extraction.
- A 2-layer Neural Network: for image-text feature matching.

# Procedures
<p align="center">
<img align="center" src="https://drive.google.com/uc?export=view&id=15i2ohtNJ0JNzBz_iEJDH_3ERe64Jxqj7" height="400"/>

## Step 1: Inverted Index
`inverted_index.py` aims to establish a mapping from the tag terms to image ids, speeding up the image search. A dictionary that maps each tag term to the list of images corresponding to that term is stored in a json file `tag2picid.json`.

## Step 2: Text Feature Extraction
`txt_feature_extractor.py` uses an off-the-shelf word2vec model to extract the word embeddings of all the tag terms. Each word is transformed into a word vector of dimension 200. The embeddings are stored in folders `txt_feature_train/` and `txt_feature_test/`.

## Step 3: Image Feature Extraction
`img_feature_extractor.py` leverages ResNet50 to extract image features from all the images in the dataset. Each image is transformed into a feature vector of dimension 2048. The image features are stored in folders `pic_feature_train/` and `pic_feature_test/`.

## Step 4: Image-Text Feature Matching
`train.py` constructs a 2-layer neural network model to transform the image feature vector and text feature vector into two vectors, both of which have dimension 128. The two transformed vectors should have high cosine similarity if the source image and word match (i.e., the word is the tag term of the image), and should have low similarity otherwise. The model is trained by minimizing the contrastive loss.

## Step 5: Similarity Ranking and Search Engine Server Construction
`server.py` builds a web application for image searching. Each time a user tries to search for images by giving a tag term, the server looks for all the images with the given tag term in the database constructed by inverted index. Now with all the matching images, their similarities with the tag term are computed using the model we trained in Step 4. The images are then ranked by their similarities, and show up in the browser according to this order, so that the pictures that the user sees first should be the ones that match the tag term the most.
