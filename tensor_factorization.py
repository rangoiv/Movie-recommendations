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
    # n = int(n)
    # # We need one letter per dimension
    # # (maybe you could find a workaround for this limitation)
    # if n > 26:
    #     raise ValueError('n is too large.')
    # ind = ''.join(chr(ord('a') + i) for i in range(n))
    # exp = f'{ind}K...,JK->{ind}J...'
    # return np.einsum(exp, x, u)
    return tensorly.tenalg.mode_dot(x, u, n)


def l(f, y):
    return (f - y) ** 2 / 2


def dl(f, y):
    return f - y


def dU(F, Y, i, j, k, S, U, M, C):
    x = dl(F[i, j, k], Y[i, j, k])
    X = n_mode_product(S, M[j, :], 1)
    X = n_mode_product(X, C[k, :], 1)
    return x*X


def dM(F, Y, i, j, k, S, U, M, C):
    x = dl(F[i, j, k], Y[i, j, k])
    X = n_mode_product(S, U[i, :], 0)
    X = n_mode_product(X, C[k, :], 1)
    return x*X


def dC(F, Y, i, j, k, S, U, M, C):
    x = dl(F[i, j, k], Y[i, j, k])
    X = n_mode_product(S, U[i, :].T, 0)
    X = n_mode_product(X, M[j, :].T, 0)
    return x*X


def dS(F, Y, i, j, k, S, U, M, C):
    x = dl(F[i, j, k], Y[i, j, k])
    X = np.kron(U[i, :], M[j, :])
    X = np.kron(X, C[k, :])
    return x*X.reshape(S.shape)


def tensor_factorization(Y: NDSparseArray, d: D):
    t0 = 10
    la = Lambda(0.001, 0.001, 0.001, 0.001)

    n, m, c = Y.shape
    U = np.random.rand(n, d.U) * 0.1**5
    M = np.random.rand(m, d.M) * 0.1**5
    C = np.random.rand(c, d.C) * 0.1**5
    S = np.random.rand(d.U, d.M, d.C) * 0.1**7

    F = NDSparseArray(Y.shape)

    t = t0
    warnings.filterwarnings("error")
    for ind, (i, j, k) in enumerate(Y.indexes()):
        print(f"{ind}/{len(Y.elements)}", end='\r')
        m = 1 / t ** 0.5
        t = t + 1
        S1 = n_mode_product(S, U[i, :], 0)
        S1 = n_mode_product(S1, M[j, :], 0)
        S1 = n_mode_product(S1, C[k, :], 0)
        F[i, j, k] = S1
        U[i, :] = U[i, :] - m * la.U * U[i, :] - m * dU(F, Y, i, j, k, S, U, M, C)
        M[j, :] = M[j, :] - m * la.M * M[j, :] - m * dM(F, Y, i, j, k, S, U, M, C)
        C[k, :] = C[k, :] - m * la.C * C[k, :] - m * dC(F, Y, i, j, k, S, U, M, C)
        S = S - m * la.S * S - m * dS(F, Y, i, j, k, S, U, M, C)
    return U, M, C, S
