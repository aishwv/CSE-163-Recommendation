# Investor's Movie Recommendation System

## Description 

We analyse the revenue, director, rating and genre for each movie and make a detailed analysis as to which movies in the past have performed well and try to apply the same configuration for our machine learning model to ensure that the producer has the most profitable movie. We also accompany our analysis with visualizations that can help us better understand the data.

## Setup

This project is written completely in Python code, you can execute this project either on Ed CSE 163 workspace or other environments that can execute Python code such as Visual Studio Code or Jupiter Notebook. You can run the command `python main.py` on the terminal to run the main file of our project. Our code also uses **pandas**, **matplotlib**, **seaborn** installation and set up for these libraries must be ensured.

## Run
The `main.py` will first run the test of every functionality we made it for each research question. After passing all th unit tests. It will start to plot the figures of each research question. As for the **recommendation system**, it is rely on two dataset, including the main dataset `tomatoes_movies_csv` and the result from top 10 directors. We also make a example of searching for 'Avatar' in the `main` function. It will first look up the keyword by finding the most similary movie in the database, and finally print out all recommended movies.

## Machine Learning Library

To build the recommendation sytem, you will need extra library including **sklearn** and 
**fuzzywuzzy**.

## Datasets

The **TMDB dataset** is too large to be displayed on github but can be found here: https://www.kaggle.com/tmdb/tmdb-movie-metadata/ 
and the **rotten tomatoes dataset** can be found here: https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset?select=rotten_tomatoes_movies.csv 


