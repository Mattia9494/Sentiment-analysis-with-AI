# Sentiment-analysis-with-AI-Thesis
This repository contains my codes and experiments associated with my master thesis. I will develop a sentiment analysis investment strategy using natural language processing, thanks to machine learning techniques.
The databased comes from Kaggle: https://www.kaggle.com/kazanova/sentiment140 and is renamed "Sentiment140.csv" as input of the model.

I currently created a twitter analyzer with a multi-layer perceptron that uses Bidirectional GRU hidden layers.

I am using a GloVe word embedding coming from: https://nlp.stanford.edu/projects/glove/
in particular, the one using Wikipedia 2014 + Gigaword 5.
The files:
- glove.6B.50d.txt
- glove.6B.100d.txt
- glove.6B.200d.txt
- glove.6B.300d.txt

Should be put in the same folder/directory for use. The hyperparameter "embedding_dim" in twitter_sentiment_GRU.py defines which embeddings dimensions we want to use, therefore it can only take the values 50, 100, 200 or 300.
