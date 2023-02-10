"""
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings.csv
"""

import numpy as np
from datetime import datetime

from sparse_array import NDSparseArray
from tensor_factorization import tensor_factorization, D, evaluate


def load_movies():
    arr = np.loadtxt(".\datasets\movies\\ratings_small.csv", delimiter=",", skiprows=1, dtype="i4,i4,f,i4")
    print("Loaded dataset")
    timestamps = [line[3] for line in arr]
    timestamps = [datetime.fromtimestamp(timestamp) for timestamp in timestamps]
    mini = min(timestamps).year
    timestamps = [(timestamp.year-mini)*12+timestamp.month for timestamp in timestamps]

    userIds = [line[0] for line in arr]
    movieIds = [line[1] for line in arr]
    ratings = [line[2] for line in arr]

    Y = NDSparseArray((max(userIds)+1, max(movieIds)+1, max(timestamps)+1))

    for i in range(len(userIds)):
        Y[userIds[i], movieIds[i], timestamps[i]] = ratings[i]

    return Y


def main():
    # LOAD DATA
    Y = load_movies()
    Y_test = Y  # load_movies(".\datasets\movies\\ratings_small.csv")

    # FACTORIZE
    print(Y.shape)
    U, M, C, S = tensor_factorization(Y, D(27, 76, 13))

    # VERIFY
    print("Done!")
    SE = 0
    n = len(Y_test.elements)
    for i, j, k in Y_test.indexes():
        rating = Y_test[i, j, k]
        evalRating = evaluate(U, M, C, S, i, j, k)
        error = abs(rating - evalRating)
        SE += error
    MAE = SE / n
    print(MAE)


main()
