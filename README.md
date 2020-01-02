# BERT_multisource
BERT-based neural network trained on multiple different (but related) data sources. Example is written for IMDB, Amazon, and Yelp reviews simultaneously. The key is the change of objective function, which is the average of validation errors from all three sources (as opposed to a single source). . 

This model effectively learns sentiment (1/0 for 4/5 and 1/2 star reviews, respectively) for different types of reviews, including 1000 each Amazon, Yelp, and IMDB, from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences). 

This model was used for one of my PhD projects and involved one of the more complicated codes I needed to write. I needed to get 95% CI intervals, so I used a resampling measure to calculate the errors for each model, all 7 potential combinations of Yelp, Amazon, and/or IMDB data, and train a neural network for each. In practice this approach would take way too long for most data set sizes, but worked well for the 3,000 reviews here, with results that indicate that training on all three data sources retained good performance on each. 
