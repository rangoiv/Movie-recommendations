'''
Algorithm 1 Tensor Factorization

Input Y, d
Initialize U e R^nxdU, M e R^mxdM, C e R^cxdC and
S e R^dUxdMxdC with small random values
Set t = t0

while (i,j,k) in observations Y do:
  mi <- 1/sqrt(t) and t <- t+1
  Fijk = SxU Ui*xM

'''
import math
from random import shuffle

import numpy as np
import tensorly
import warnings

from sparse_array import NDSparseArray


class D:
    def __init__(self, U: int, M: int, C: int):
        self.U = U
        self.M = M
        self.C = C


class Lambda:
    def __init__(self, U: float, M: float, C: float, S: float):
        self.U = U
        self.M = M
        self.C = C
        self.S = S


def n_mode_product(x, u, n):
    return tensorly.tenalg.mode_dot(x, u, n)


def l(f, y):
    return (f - y) ** 2 / 2


def dl(f, y):
    return f - y


def dU(i, j, k, S, U, M, C):
    X = n_mode_product(S, M[j, :], 1)
    X = n_mode_product(X, C[k, :], 1)
    return X


def dM(i, j, k, S, U, M, C):
    X = n_mode_product(S, U[i, :], 0)
    X = n_mode_product(X, C[k, :], 1)
    return X


def dC(i, j, k, S, U, M, C):
    X = n_mode_product(S, U[i, :].T, 0)
    X = n_mode_product(X, M[j, :].T, 0)
    return X


def kron(A, B):
    dim1 = np.concatenate([np.array(A.shape), np.array(B.shape) * 0 + 1])
    dim2 = np.concatenate([np.array(A.shape) * 0 + 1, np.array(B.shape)])
    return A.reshape(dim1) @ B.reshape(dim2)


def dS(i, j, k, S, U, M, C):
    X = kron(U[i, :], M[j, :])
    X = kron(X, C[k, :])
    return X


def evaluate(U, M, C, S, i, j, k):
    f = n_mode_product(S, U[i, :].T, 0)
    f = n_mode_product(f, M[j, :].T, 0)
    f = n_mode_product(f, C[k, :].T, 0)
    return f


def evaldev(U, M, C, S, y, i, j, k):
    f = evaluate(U, M, C, S, i, j, k)
    df = dl(f, y)
    if math.isnan(f) or math.isnan(df) or math.isinf(f) or math.isinf(df):
        raise RuntimeWarning
    return f, df


def factorization_cycle(U, M, C, S, Y,
                        coef=lambda s: 0.01 * 1 / ((s + 30) ** 0.5),
                        la=Lambda(0.00001, 0.0001, 0.0001, 0.0001)):
    SE = 0
    X = list(Y.indexes())
    shuffle(X)
    warnings.filterwarnings("error")
    for t, (i, j, k) in enumerate(X):
        m = coef(t)
        y = Y[i, j, k]
        try:
            f, df = evaldev(U, M, C, S, y, i, j, k)
            DU = dU(i, j, k, S, U, M, C)
            DM = dM(i, j, k, S, U, M, C)
            DC = dC(i, j, k, S, U, M, C)
            DS = dS(i, j, k, S, U, M, C)
            U[i, :] = U[i, :] - (m * df) * DU - m * la.U * U[i, :]
            M[j, :] = M[j, :] - (m * df) * DM - m * la.M * M[j, :]
            C[k, :] = C[k, :] - (m * df) * DC - m * la.C * C[k, :]
            S = S - (m * df) * DS - m * la.S * S

            SE += abs(df)
            print(f"\r{SE / (t + 1)} {(t + 1)}/{len(Y.elements)}     ", end='')
        except RuntimeWarning:
            U[np.isinf(U)] = 0
            M[np.isinf(M)] = 0
            S[np.isinf(S)] = 0
            C[np.isinf(C)] = 0
            print("\rWarning          ", end='')
    print()
    return U, M, C, S


def initialize_factorization(Y: NDSparseArray, d: D, la: Lambda = Lambda(0.01, 0.01, 0.01, 0.01)):
    n, m, c = Y.shape
    U = np.random.rand(n, d.U) * la.U
    M = np.random.rand(m, d.M) * la.M
    C = np.random.rand(c, d.C) * la.C
    S = np.random.rand(d.U, d.M, d.C) * la.S

    return U, M, C, S


def tensor_factorization(Y: NDSparseArray, d: D, t0=30):
    print("Running tensor factorization")
    U, M, C, S = initialize_factorization(Y, d)
    for t in range(t0, t0 + 1):
        U, M, C, S = factorization_cycle(U, M, C, S, Y)
    return U, M, C, S
