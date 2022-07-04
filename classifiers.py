import numpy

from Pipeline import PipelineStage
from Tools import mcol
from models import MVGModel, TiedMVGModel


class MVG(PipelineStage):

    def __init__(self):
        super().__init__()
        self.u = None
        self.C = None
        return

    def compute(self, model, D, L):
        dim = D.shape[0]
        K = L.max()+1
        nSamples = D.shape[1]

        u = numpy.zeros((dim, K))  # array of means vectors by class [MATRIX (dim, K)]
        C = numpy.zeros((K, dim, dim))  # array of covariance matrices by class

        for i in range(K):
            ## ESTIMATION OF MODEL
            # only the class i
            DCl = D[:, L == i]  # Matrix of samples of class i (dim, Nc)
            # compute the mean
            u[:, i:i+1] = mcol(DCl.mean(1))  # vector of means of dimensions of class i (dim, 1)
            # center the points
            DClC = DCl - u[:, i:i+1]
            # compute the covariance matrix
            C[i] = numpy.dot(DClC, DClC.T) / nSamples

        self.u = u
        self.C = C

        return MVGModel(u, C), D, L

    def __str__(self):
        return 'MVG\n%s\n%s\n' % (self.u, self.C)

class NaiveBayesMVG(PipelineStage):

    def __init__(self):
        super().__init__()
        self.u = None
        self.C = None
        return

    def compute(self, model, D, L):
        dim = D.shape[0]
        K = L.max()+1
        nSamples = D.shape[1]

        u = numpy.zeros((dim, K))  # array of means vectors by class [MATRIX (dim, K)]
        C = numpy.zeros((K, dim, dim))  # array of covariance matrices by class

        for i in range(K):
            ## ESTIMATION OF MODEL
            # only the class i
            DCl = D[:, L == i]  # Matrix of samples of class i (dim, Nc)
            # compute the mean
            u[:, i:i+1] = mcol(DCl.mean(1))  # vector of means of dimensions of class i (dim, 1)
            # center the points
            DClC = DCl - u[:, i:i+1]
            # compute the covariance matrix
            C[i] = numpy.diag((DClC**2).sum(1))/nSamples

        self.u = u
        self.C = C

        return MVGModel(u, C), D, L

    def __str__(self):
        return 'NaiveBayesMVG\n%s\n%s\n' % (self.u, self.C)

class TiedMVG(PipelineStage):

    def __init__(self):
        super().__init__()
        self.u = None
        self.C = None
        return

    def compute(self, model, D, L):
        dim = D.shape[0]
        K = L.max()+1
        nSamples = D.shape[1]

        u = numpy.zeros((dim, K))  # array of means vectors by class [MATRIX (dim, K)]
        C = numpy.zeros((K, dim, dim))  # array of covariance matrices by class

        for i in range(K):
            ## ESTIMATION OF MODEL
            # only the class i
            DCl = D[:, L == i]  # Matrix of samples of class i (dim, Nc)
            # compute the mean
            u[:, i:i+1] = mcol(DCl.mean(1))  # vector of means of dimensions of class i (dim, 1)
            # center the points
            DClC = DCl - u[:, i:i+1]
            # compute the partial covariance matrix and add it to within-class
            C += numpy.dot(DClC, DClC.T)
        # divide the partial covariance by the number of samples
        C /= nSamples

        self.u = u
        self.C = C

        return TiedMVGModel(u, C), D, L

    def __str__(self):
        return 'TiedMVG\n%s\n%s\n' % (self.u, self.C)

class TiedNaiveBayesMVG(PipelineStage):

    def __init__(self):
        super().__init__()
        self.u = None
        self.C = None
        return

    def compute(self, model, D, L):
        dim = D.shape[0]
        K = L.max()+1
        nSamples = D.shape[1]

        u = numpy.zeros((dim, K))  # array of means vectors by class [MATRIX (dim, K)]
        C = numpy.zeros((K, dim, dim))  # array of covariance matrices by class

        for i in range(K):
            ## ESTIMATION OF MODEL
            # only the class i
            DCl = D[:, L == i]  # Matrix of samples of class i (dim, Nc)
            # compute the mean
            u[:, i:i+1] = mcol(DCl.mean(1))  # vector of means of dimensions of class i (dim, 1)
            # center the points
            DClC = DCl - u[:, i:i+1]
            # compute the partial covariance matrix
            C += numpy.diag((DClC**2).sum(1))
        # divide the partial covariance by the number of samples
        C /= nSamples

        self.u = u
        self.C = C

        return TiedMVGModel(u, C), D, L

    def __str__(self):
        return 'TiedNaiveBayesMVG\n%s\n%s\n' % (self.u, self.C)
