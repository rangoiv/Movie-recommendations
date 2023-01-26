import numpy

def unfold(T, mode):
    if ( mode == 1) :
        M=numpy.array([numpy.concatenate((numpy.squeeze(T[:,0,:])),axis=0)])
        for i in range(1, numpy.shape(T)[1]):
            K =numpy.array([numpy.concatenate((numpy.squeeze(T[:,i,:])),axis=0)])
            M = numpy.append(M, K, axis=0)
            
    elif (mode == 2):
        M=numpy.array([numpy.concatenate((numpy.squeeze(T[:,:,0])),axis=0)])
        for i in range(1, numpy.shape(T)[2]):
            K = numpy.array([numpy.concatenate((numpy.squeeze(T[:,:,i])), axis=0)])
            M = numpy.append(M, K, axis=0)
    
    elif (mode == 3):
        M = numpy.array([numpy.concatenate(numpy.moveaxis(T[0,:,:], 0,-1), axis=None)])
        for i in range(1, numpy.shape(T)[0]):
            K = numpy.array([numpy.concatenate(numpy.moveaxis(T[i,:,:], 0,-1), axis=None)])
            M = numpy.append(M, K, axis=0)

    return M



def fold(M, mode, dimT):
    k,m,n = dimT

    T = numpy.zeros((k,m,n))

    if mode == 1:
        for i in range(0, k):
            T[i,:,:] = M[:, i*n: (i+1)*n]
    elif mode == 2:
        for i in range(0, k):
            T[i,:,:] = numpy.transpose(M[:, i*m :(i+1)*m])
    elif mode == 3:
        for i in range(0, n):
            T[:,:,i] = M[:, i*m :(i+1)*m]
    
    return T