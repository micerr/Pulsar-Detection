import numpy

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def logpdf_GAU_ND(X, mu, C):
    M = mu.shape[0]  # dimensions
    pi = numpy.pi
    Precision = numpy.linalg.inv(C)

    first = -M/2*numpy.log(2*pi)
    second = -0.5*numpy.linalg.slogdet(C)[1]
    XC = X - mu  # center the values
    third = -0.5*numpy.dot(numpy.dot(XC.T, Precision), XC)
    # take only the rows (i,i)
    return numpy.diagonal(first+second+third)
