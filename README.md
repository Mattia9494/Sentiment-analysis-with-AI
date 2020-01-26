# Overview of the thesis and code content

This thesis aims at investigating if and to what point it is possible to trade on sentiment
and if Artificial Intelligence (AI) would be the best way to do so.
Despite the strict bindings of the Efficient Market Hypothesis (EMH) it appears that
some kind of trading opportunity might be available. It may not be a case that many of
those so-called "anomalies" appear to be, in reality, regular irrational behaviors done by
financial agents. In fact, behavioral finance rose a wide literature and multiple debates
supporting this statement. For this reason, I indicate an alternative to the EMH called
Adaptive Market Hypothesis (AMH) whose benefit is incorporating the best of both
worlds.

With this more appropriate theoretical basis we will take inspiration from previous
works, such as Sohangir 2018 or Ruiz-Martinez 2012, where they show that it
is possible to extract and use the market sentiment, specifically news, through computational
methods in the world dominated by Big Data we live in.

Among the various methods that can be used for such analysis, I decided to focus
on a form of AI Machine Learning technique named Deep Learning (DL) and to build
a model based on it. The reason for this choice is not simply driven by the curiosity
towards a model that has been recently hyped among the AI community, but by the
fact that it might indeed one of the best tools to face such problems in the proper way.
This because DL is built for specifically dealing with big amounts of data and perform
complex tasks where automatic learning is a necessity.

After having explained how AI and more specifically DL models are built, I will use
this tool for forecasting the market sentiment through the use of news headlines. The
prediction is based on the movement of the Dow Jones Industrial Average (DJIA) by
analyzing 25 daily news headlines available between 2008 and 2016. The result will be
the signal that can be theoretically used for developing an algorithmic trading strategy.
The analysis will be done on two specific situations (that I called A and B) that will be
pursued throughout five time-steps: today (T0), today+1 (T1), today+2 (T2), today+3
(T3), today+4 (T4).

On the one hand, Case A will use a DL system to predict, for instance, based on the
news headlines published on T0, if the adjusted closing price of T3 is higher than the
opening price on T3.

On the other hand, Case B uses a similar DL system to make an analogous prediction,
but this time only the intervals will be considered, instead of the single days. Using
the same example: It will be analyzed, based the news published in T0, whether the
adjusted closing price of T3 is higher than the opening price of T0.
In order to test the models' applicability in real case scenarios, both case A and B
will be implemented in two situations:

• Practical case 1: "Political election case" where we will explore how accurately the
model would have performed right after the expression of the popular vote in a
western country.

• Practical case 2: "Policies during Trump's presidency" where it will be investigated
to see how well the algorithm would predict the DJIA movements based on some
of Trump's most controversial policies during its presidency.

In the first chapter, we will start discovering what market sentiment is by examining
the weaknesses of the Efficient Market Hypothesis and some concepts of Behavioral Finance.
This will help us explain why and how it is reasonable to trade on news in a Big
Data environment using Deep Learning techniques.

An overview of linguistics will be delineated in the second chapter with a focus on
natural language processing that aims at analyzing many aspects of the written language.
A particular attention will be given to the concepts of ambiguity and preprocessing, since
they deal with all those procedures used for preparing the language that will be used as
input for our model.

In the third chapter, a general introduction to machine learning will be presented,
an idea that will be pushed to its limits in the fourth chapter when we will dive deeply
into deep learning algorithms. The fifth chapter talks about the vectorization of words
and the most popular forms of word embeddings nowadays.

Finally I will introduce the model itself, together with the analysis, the methods used
and the results obtained in its practical implementation. In addition, a couple of sections
will show the related work and the steps that could be done for improving the model's
accuracy in the future.

Our journey concludes with the presentation of the doubts and suggestion on how
deep learning techniques for systematic investment strategies could improve with further
research, but also the general impact that this new technology might have in the financial
markets.


This repository contains the codes and experiments associated with my master thesis. The project consists in the development of an investment strategy using sentiment analysis on the Dow Jones through deep learning techniques.

The project has been implemented in two versions of a similar model, that are discerned by the use of two different databases:
- The first database is the "Combined_News_DJIA.csv". It comes from this Kaggle file: https://www.kaggle.com/aaron7sun/stocknews
and works as input for the DJIA_sentiment.py AI model in the T0 case (both A and B). Let's first explore the composition of the DJIA dataset:
  1. A first column of dates from 08-08-2008 to 01-07-2016 for a total of 1989 lines.
  2. A second column of human-tagged labels, in which:
    - One: is expressed when the DJIA increases or stays the same on the same day.
    - Zero: in case of a daily decrease of the DJIA.
  3. The next 25 columns represent the most relevant news for that day in the form of
  strings. During the preprocessing phase these news strings will be merged together
  in a unique string for each row. By doing so we will obtain the general market
  signal that we will try to predict.
 
- The other databases have been derived from the first one, therefore it has the same structure, but with fewer rows. All those cases depend on case A and case B, respectively, in T1, T2, T3, T4.

- preprocessing.py contains all the formulas that have been used in both models for used to preprocess the current text.

- Applied_model.py is used to test the validity of the model with some days characterized by the presence of international breaking news due to main historical or political events in Case A and B.
