import pickle

import numpy as np

from sparse_array import NDSparseArray
from tensor_factorization import Lambda, fine_tune, evaluate


class FactorizationRatingsAprox:
    def __init__(self, U, M, C, S, Y_shape, new_user_id):
        self.U = U
        self.M = M
        self.C = C
        self.S = S
        self.Y_shape = Y_shape
        self.new_user_id = new_user_id

    @classmethod
    def from_file(cls, path_to_file):
        with open(path_to_file, 'rb') as file:
            loaded_obj = pickle.load(file)
        return loaded_obj

    def to_file(self, path_to_file):
        with open(path_to_file, 'wb+') as file:
            pickle.dump(self, file)

    def evaluate(self, movie_ratings):
        Y = NDSparseArray(self.Y_shape)
        timestamp = Y.shape[2]-1
        for rating in movie_ratings:
            Y[self.new_user_id, rating[0], timestamp] = rating[1]

        U = np.array(self.U, copy=True)
        M = np.array(self.M, copy=True)
        C = np.array(self.C, copy=True)
        S = np.array(self.S, copy=True)
        la = Lambda(0.000001, 0.0000001, 0.000001, 0.000001)

        for t in range(30, 35):
            U, M, C, S = fine_tune(U, M, C, S, Y, lambda s: 0.01 * 1 / (t ** 0.5), la)

        aproximate_ratings = []
        for movie_id in range(self.Y_shape[1]):
            rating = evaluate(U, M, C, S, self.new_user_id, movie_id, timestamp)
            aproximate_ratings.append([rating, movie_id])

        aproximate_ratings = [rating if 0 < rating[0] < 5.5 else [0, rating[1]] for rating in aproximate_ratings]
        aproximate_ratings.sort(reverse=True)
        aproximate_ratings = [[rating[1], float(rating[0])] for rating in aproximate_ratings]
        return aproximate_ratings[:100]
