import numpy as np
from datetime import datetime

from sparse_array import NDSparseArray
from tensor_factorization import tensor_factorization, D


# LOAD DATA

arr = np.loadtxt(".\datasets\movies\\ratings_small.csv",
                 delimiter=",", skiprows=1, dtype="i4,i4,f,i4")

timestamps = [line[3] for line in arr]
mint = min(timestamps)
timestamps = timestamps - mint + 1
timestamps = (timestamps / 10000).astype(int)
maxt = max(timestamps)

userIds = [line[0] for line in arr]
movieIds = [line[1] for line in arr]
ratings = [line[2] for line in arr]

Y = NDSparseArray((max(userIds)+1, max(movieIds)+1, maxt+1))

for i in range(len(arr)):
    Y[userIds[i], movieIds[i], timestamps[i]] = ratings[i]

# FACTORIZE
U, M, C, S = tensor_factorization(Y, D(8, 10, 12))

# VERIFY
print("Done!")
print(U, M, C, S)
