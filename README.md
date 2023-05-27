# LSTM_multisource
Note: Requires torch 0.10.0


This model effectively learns to predict sentiment (1/0 for 4/5 and 1/2 star reviews, respectively) for different types of reviews, including 1000 each Amazon, Yelp, and IMDB, from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences). 

This project arose from a curiosity about learning and training models to be more generally applicable outside a very specific task of interest. The model is an LSTM-based neural network simultaneously trained on multiple different (but related) data sources. Example is written for IMDB, Amazon, and Yelp reviews simultaneously. The key is the change of objective function (example 403-418 in train.py), which is the average of validation errors from all three sources (as opposed to a single source), and for every epoch, stepping down the gradient on each data set sequentially. By iteratively training on 3 different datasets trained on similar tasks (in this case, positive or negative review), we can identify the model with the highest generalizability that performs well on all three datasets simultaneously. 


The code runs a series of 100 random cross validation splits, and for each split, trains an LSTM model on each Amazon, IMDB, and Yelp reviews separately, as well as pairwise combinations and a model trained using all 3. The accuracies after training are collected. Analysis of the results indicate that training through using all 3 datasets simultaneously allows us to create generalizable models that are able to classify across different tasks. This can be useful to create generalized review models, that can classify positive/negative reviews across different platforms. Generalized models may perform better in cases where there is limited data from a task of interest, such as when a new platform is created.
