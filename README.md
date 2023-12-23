# NLP_SentimentAnalysys
# Sentiment Analysis Project

This repository contains the code and resources for a sentiment analysis project focused on analyzing customer reviews of restaurants on the Zomato platform. The project leverages Natural Language Processing (NLP) techniques and explores three different models: LSTM, Naive Bayes, and MaxEnt.

## Project Structure

- **`model/`**: This directory contains the trained models used for sentiment analysis.
  - `model_lstm.h5`: LSTM model file.
  - `model_maxent.pickle`: MaxEnt (Maximum Entropy) model file.
  - `model_naivebayes.pickle`: Naive Bayes model file.

- **`update/`**: This directory stores recent updates to the models.

- **`Ratings.csv`**: The dataset used for training and testing the sentiment analysis models.

- **`demo.py`**: The Python script for running a demo of the sentiment analysis models. If the demo doesn't run successfully, consider re-running all cells in the `nlp_sentiment.ipynb` notebook to reload the models and then run the demo again.

- **`nlp_sentiment.ipynb`**: Jupyter notebook containing the main codebase for the project. It covers data processing, model training, and evaluation.

- **`processing_data.py`**: Python script for data preprocessing.

## Instructions for Running Demo

If running the `demo.py` script is not successful, follow these steps:

1. Open the `nlp_sentiment.ipynb` notebook in a Jupyter environment.

2. Run all cells in the notebook to reload the models.

3. Execute the `demo.py` script again. This time, it should run successfully.

Feel free to explore the code, datasets, and models provided in this repository. If you have any questions or issues, please create a new GitHub issue.
