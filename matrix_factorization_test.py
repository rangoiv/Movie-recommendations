import numpy

from matrix_factorization import matrix_factorization

R = [

    [5, 3, 0, 1],

    [4, 0, 0, 1],

    [1, 1, 0, 5],

    [1, 0, 0, 4],

    [0, 1, 5, 4],

    [2, 1, 3, 0],

]

R = numpy.array(R)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])
# Num of Features
K = 3

P = numpy.random.rand(N, K)
Q = numpy.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)

nR = numpy.dot(nP, nQ.T)

print(nR)

# %%
