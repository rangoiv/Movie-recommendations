import pickle

import numpy as np

from sparse_array import NDSparseArray
from tensor_factorization import Lambda, fine_tune, evaluate


class FactorizationRatingsAprox:
    def __init__(self, U, M, C, S, Y: NDSparseArray, new_user_id):
        self.U = U
        self.M = M
        self.C = C
        self.S = S
        self.Y_shape = Y.shape
        self.new_user_id = new_user_id

        average_ratings = {}
        for ind in Y.indexes():
            score = Y[ind]
            movie_id = ind[1]
            if not movie_id in average_ratings:
                average_ratings[movie_id] = [0, 0]
            average_ratings[movie_id][0] += score
            average_ratings[movie_id][1] += 1
        self.average_ratings = average_ratings


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
        average_ratings = {}
        for movie_id in self.average_ratings:
            average_ratings[movie_id] = self.average_ratings[movie_id][0] / self.average_ratings[movie_id][1]

        U = np.array(self.U, copy=True)
        M = np.array(self.M, copy=True)
        C = np.array(self.C, copy=True)
        S = np.array(self.S, copy=True)
        la = Lambda(0.000001, 0.0000001, 0.000001, 0.000001)

        for t in range(30, 35):
            U, M, C, S = fine_tune(U, M, C, S, Y, lambda s: 0.01 * 1 / (t ** 0.5), la, debug=False)

        aproximate_ratings = []
        for movie_id in range(self.Y_shape[1]):
            rating = evaluate(U, M, C, S, self.new_user_id, movie_id, timestamp)
            aproximate_ratings.append([rating, movie_id])

        new_aproximate_ratings = []
        for rating in aproximate_ratings:
            if 0 < rating[0] < 5.5 and rating[1] in average_ratings:
                new_aproximate_ratings.append([rating[0], rating[1]])
        aproximate_ratings = new_aproximate_ratings
        aproximate_ratings = [[float(rating[0]), rating[1]] for rating in aproximate_ratings]
        return aproximate_ratings
