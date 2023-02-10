"""
https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews
"""

import numpy as np
from datetime import datetime

from sparse_array import NDSparseArray
from tensor_factorization import tensor_factorization, D, evaluate


def load_amazon():
    arr = np.loadtxt(".\datasets\\amazon\\ratings_electronics.csv", delimiter=",", skiprows=1, dtype="U16,U16,f,i4")
    print("Loaded dataset")
    arr = arr[:int(len(arr)*0.1)]

    timestamps = [line[3] for line in arr]
    timestamps = [datetime.fromtimestamp(timestamp) for timestamp in timestamps]
    mini = min(timestamps).year
    timestamps = [(timestamp.year - mini) * 12 + timestamp.month for timestamp in timestamps]

    userIds = [line[0] for line in arr]
    userIdMap = {}
    l = 0
    for userId in userIds:
        if userId not in userIdMap:
            userIdMap[userId] = l
            l += 1
    for i, userId in enumerate(userIds):
        userIds[i] = userIdMap[userId]

    productIds = [line[1] for line in arr]
    productIdMap = {}
    l = 0
    for productId in productIds:
        if productId not in productIdMap:
            productIdMap[productId] = l
            l += 1
    for i, productId in enumerate(productIds):
        productIds[i] = productIdMap[productId]

    ratings = [line[2] for line in arr]

    Y = NDSparseArray((max(userIds) + 1, max(productIds) + 1, max(timestamps) + 1))

    for i in range(len(userIds)):
        Y[userIds[i], productIds[i], timestamps[i]] = ratings[i]

    return Y


def main():
    # LOAD DATA
    Y = load_amazon()
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
