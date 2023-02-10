"""
https://www.kaggle.com/datasets/cbhavik/music-taste-recommendation?resource=download
"""

import numpy as np
from datetime import datetime

from sparse_array import NDSparseArray
from tensor_factorization import tensor_factorization, D, evaluate


def load_songs():
    arr = np.loadtxt(".\datasets\songs\\piki_dataset.csv", delimiter=",", skiprows=1, dtype="i4,i4,i4,i4,i4,U20,i4")
    print("Loaded dataset")
    print(arr)
    timestamps = [line[5] for line in arr]
    for i in range(len(timestamps)):
        timestamp = timestamps[i]
        # hour = int(timestamp.split(" ", 1)[1].split(":", 1)[0])
        year = int(timestamp.split("-", 1)[0])
        month = int(timestamp.split("-", 2)[1])
        timestamp = (year-2019)*12+month
        timestamps[i] = timestamp

    userIds = [line[6] for line in arr]
    songIds = [line[3] for line in arr]
    ratings = [line[1] for line in arr]
    print(len(userIds))
    Y = NDSparseArray((max(userIds)+1, max(songIds)+1, max(timestamps)+1))

    for i in range(len(userIds)):
        Y[userIds[i], songIds[i], timestamps[i]] = ratings[i]

    return Y


def main():

    # LOAD DATA
    Y = load_songs()
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
