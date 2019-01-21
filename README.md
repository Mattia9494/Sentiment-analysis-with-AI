# Sentiment-analysis-with-AI-Thesis
This repository contains the codes and experiments associated with my master thesis. The project consists in the development of an investment strategy using sentiment analysis on the Dow Jones through deep learning techniques.

The project has been implemented in two versions of a similar model, that are discerned by the use of two different databases:
- The first database is the "Combined_News_DJIA.csv". It comes from this Kaggle file: https://www.kaggle.com/aaron7sun/stocknews
and works as input for the DJIA_sentiment.py AI model. Let's first explore the composition of the DJIA dataset:
  1. A first column of dates from 08-08-2008 to 01-07-2016 for a total of 1989 lines.
  2. A second column of human-tagged labels, in which:
    (a) One: is expressed when the DJIA increases or stays the same on the same day.
    (b) Zero: in case of a daily decrease of the DJIA.
  3. The next 25 columns represent the most relevant news for that day in the form of
  strings. During the preprocessing phase these news strings will be merged together
  in a unique string for each row. By doing so we will obtain the general market
  signal that we will try to predict.
 
- The second database is the "Combined_News_DJIA_plusone.csv". This csv file has been derived from the first one, therefore it has the same structure, but with fewer rows. It answers to the question: What if the biggest market signal of today's news appears on tomorrow closing price instead? Therefore the labels in the second column are:
  1. One: is expressed when the close price of the DJIA tomorrow is higher (or equal) of today's opening price.
  2. Zero: in case of a decrease of the DJIA. 

- preprocessing.py contains all the formulas that have been used in both models for used to preprocess the current text.

- Applied_model.py is used to test the validity of the model with some days characterized by the presence of breaking news in the world due to main historical or political events, such as the Trump election.

All the other files are provisory at the moment since the work is still in progress.
