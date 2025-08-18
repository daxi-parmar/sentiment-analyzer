# sentiment-analyzer
A simple program that predicts whether a movie review is Positive or Negative using Python, NLTK, and scikit-learn.

## Overview
This project uses a pre-labeled dataset of 2000 movie reviews from NLTK.The reviews are converted into numerical features using TF-IDF,and a Naive Bayes classifier is used to predict the sentiment.Users can also enter their own reviews to test the model.

## Features
Loads dataset from NLTK,
Converts reviews into numbers using TF-IDF Vectorizer,
Trains a Naive Bayes model,
Predicts new user input as Positive or Negative,

## Tech Stack
Python,
NLTK,
Scikit-learn,
Pandas.

## How to run it
1. Install the libraries
pip install nltk scikit-learn pandas

2. Run the code in VS Code or terminal
python sentiment_analyzer.py

3. Type your own review, for example:
Enter a review(or type 'quit' to stop): It enjoyed reading the book!
Sentiment → Positive
