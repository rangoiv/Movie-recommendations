import numpy

from matrix_factorization import matrix_factorization

import fold_unfold 

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

# print(nR)

# %%


T = numpy.array([[[1,4,7,10], [2,5,8,11], [3,6,9,12]],
                [[13,16,19,22], [14,17,20,23], [15,18,21,24]],
                [[25,26,27,28], [29,30,31,32], [33,34,35,36]]])


# T = numpy.array([[[1,13],[2,14],[3,15],[4,16]],[[5,17],[6,18],[7,19],[8,20]],[[9,21],[10,22],[11,23],[12,24]]])

k,m,n = numpy.shape(T)
print(T)
M = fold_unfold.unfold(T, 1)
print("1:\n",M)
T=fold_unfold.fold(M, 1, (k,m,n))
print(T)
M = fold_unfold.unfold(T, 2)
print("2:\n", M)
T=fold_unfold.fold(M, 2, [k,m,n])
print(T)
M = fold_unfold.unfold(T, 3)
print("3:\n",M)
T=fold_unfold.fold(M, 3, [k,m,n])
print(T)



