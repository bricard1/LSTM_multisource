# LSTM_multisource
Note: Requires torch 0.10.0
LSTM-based neural network trained on multiple different (but related) data sources. Example is written for IMDB, Amazon, and Yelp reviews simultaneously. The key is the change of objective function, which is the average of validation errors from all three sources (as opposed to a single source). By iteratively training on 3 different datasets trained on similar tasks (in this case, positve or negative review), we can identify the model with the highest generalizability that performs well on all thhree datasets simultanously. 

This model effectively learns sentiment (1/0 for 4/5 and 1/2 star reviews, respectively) for different types of reviews, including 1000 each Amazon, Yelp, and IMDB, from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences). 
