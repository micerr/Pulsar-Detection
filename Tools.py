import numpy

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def logpdf_GAU_ND(X, mu, C):
    M = mu.shape[0]  # dimensions
    pi = numpy.pi
    Precision = numpy.linalg.inv(C)

    first = -M / 2 * numpy.log(2 * pi)
    second = -0.5 * numpy.linalg.slogdet(C)[1]
    XC = X - mcol(mu)  # center the values
    third_1 = numpy.dot(XC.T, Precision)
    third = -0.5 * (third_1.T * XC).sum(0)
    return first + second + third
