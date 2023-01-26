class NDSparseArray:
    def __init__(self, shape: tuple, dtype=int):
        self.elements = {}
        self.shape = shape

    def __setitem__(self, key, value):
        self.elements[key] = value
        if value == 0:
            del self.elements[key]

    def __getitem__(self, item):
        try:
            value = self.elements[item]
        except KeyError:
            value = 0
        return value

    def indexes(self):
        return self.elements.keys()

# def sparse_mult(sparse, other_sparse):
#
#     out = NDSparseArray()
#
#     for key, value in sparse.elements.items():
#         i, j, k = key
#         # note, here you must define your own rank-3 multiplication rule, which
#         # is, in general, nontrivial, especially if LxMxN tensor...
#
#         # loop over a dummy variable (or two) and perform some summation
#         # (example indices shown):
#         out.setValue(key) = out.readValue(key) +
#         other_sparse.readValue((i,j,k+1)) * sparse((i-3,j,k))
#
# return out