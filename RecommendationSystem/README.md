<div align="center">

  # Movie Recommendation System

</div>

## Introduction
This project retrieves movie rating data from [Netflix prize dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) and builds a recommendation system based on Alternating Least Square (ALS) Matrix Factorization. Spark ML is used for model implementation.

## Part 1: OLAP
Use Spark SQL to explore the movie rating dataset by looking into these aspects:

- About users
    - number of users
    - number of movies rated by each user
    - minimum and maximum number of movies rated by each user
    - user distribution with respect to the quantity of rated movies

- About movies
    - number of movies
    - number of movies in each year
    - number of users rating each movie
    - minimum and maximum number of users rating each movie
    - movie distribution with respect to the quantity of raters

- About ratings
    - distinct values of rating
    - average rating of each movie
    - movie with the highest and lowest average rating
    - movie distribution with respect to the average rating
    - average rating of movies in each year
    - best movie in each year

## Part 2: Model Training
- Use grid search and cross-validation to find the best set of hyperparameters. 
- Then adopt the best hyperparameter set to train the model again using the entire dataset. 
- Finally, save the fully trained model which will be used for giving recommendations.

## Part 3: Recommendation
- Find similar movies to a given movie by the cosine similarity of their feature vectors derived in ALS model.
- Recommend movies to an existing user by selecting the movies with highest predicted ratings given by the ALS model.
- Recommend movies to a new user, who has just watched a few movies. Include that user's viewing history in the training set and retrain the model. Then give recommendations correspondingly.