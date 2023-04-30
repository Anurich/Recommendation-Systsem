import pandas as pd 
import numpy as np 
from datetime import datetime


def readData():
    movie_data = pd.read_csv("ml_data/movies.csv")
    rating_data = pd.read_csv("ml_data/ratings.csv")

    return movie_data, rating_data


def preprocess(ratings):
    # let's keep only those users which has rated more than 10 movies 
    mask = ratings.groupby("userId").count()["movieId"]>10
    ratings = ratings[ratings.userId.isin(mask[mask].index)]
    # let's remove those ratings which are less than 3.5
    ratings = ratings[ratings["rating"] > 3.5]
    # let's reset the index 

    ratings.reset_index(inplace=True, drop=True)
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"]).dt.date
    ratings.set_index("timestamp", inplace=True)
    ratings = ratings.sort_index()
    return  ratings[:2000000]


