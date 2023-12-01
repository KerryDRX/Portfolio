<div align="center">

  # Document Clustering and Topic Modeling

</div>

## Overview
- This project leverages unsupervised models to cluster unlabeled e-commerce product reviews into groups for customer sentiment analysis.
    - Tokenization and stemming are firstly performed on the raw data for preprocessing.
    - The TF-IDF model is then applied to the documents for feature engineering.
    - Based on the TF-IDF matrix, models including K Means Clustering and Latent Dirichlet Allocation are applied to group customer reviews into clusters and identify the hidden topics of the corpus.

## Dataset
- The dataset is retrieved from [Amazon Customer Reviews Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html), a collection of customers' reviews on US watches.
