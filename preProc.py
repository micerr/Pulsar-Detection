import numpy
import scipy

from Pipeline import PipelineStage
from Tools import mcol, center_data, cov_mat, within_cov_mat


class PCA(PipelineStage):

    def __init__(self):
        super().__init__()
        self.m = None
        self.mean = None
        self.C = None

    def compute(self, model, D, L):
        nSamples = D.shape[1]
        dim = D.shape[0]
        m = self.m if self.m <= dim else dim

        # Center the data
        self.mean = mcol(D.mean(1))
        DC = D - self.mean
        # Covariance matrix
        self.C = numpy.dot(DC, DC.T)/nSamples
        # Eigenvalues and Eigenvectors
        s, U = numpy.linalg.eigh(self.C)
        # Eigenvectors with higher variance eigenvalues
        P = U[:, ::-1][:, 0:m]
        # Project data
        DP = numpy.dot(P.T, D)

        return model, DP, L

    def setDimension(self, m):
        self.m = m

    def __str__(self):
        return 'PCA\nmu = %s\nC = %s\n' % (self.mean, self.C)


class LDA(PipelineStage):

    def __init__(self):
        super().__init__()
        self.Sb = None
        self.Sw = None
        self.m = None

    def compute(self, model, D, L):
        nSamples = D.shape[1]
        K = L.max()+1
        m = self.m if self.m <= K-1 else K-1

        Sw = 0
        Sb = 0
        # Dataset mean
        meanD = mcol(D.mean(1))
        for i in range(K):
            DCl = D[:, L == i]  # take only samples of the class-i
            meanClass = mcol(DCl.mean(1))  # compute the mean of the class data
            DClC = DCl - meanClass  # center the class data
            mC = meanClass - meanD  # center the mean of class, respect the global mean
            ## COMPUTING ELEMENT-I OF THE SUMMARY OF Sb
            Sb += DClC.shape[1] * numpy.dot(mC, mC.T)
            ## COMPUTING ELEMENT-I OF THE SUMMARY OF Sw
            Sw += numpy.dot(DClC, DClC.T)
        self.Sw = Sw = Sw / nSamples
        self.Sb = Sb = Sb / nSamples

        ## COMPUTING THE EIG VALUES OF THE GENERALIZED EIGENVALUE PROBLEM FOR HERMITIAN MATRICES
        s, U = scipy.linalg.eigh(Sb, Sw)  # numpy here don't work, numpy don't solve the generalized problem
        P = U[:, ::-1][:, 0:m]  # take the  dimension

        ## PROJECTING DATA ON NEW BASE ##
        DP = numpy.dot(P.T, D)

        return model, DP, L

    def setDimension(self, m):
        self.m = m

    def __str__(self):
        return "LDA\nSw = %s\nSb = %s" % (self.Sw, self.Sb)

class Whiten(PipelineStage):
    def __init__(self):
        super().__init__()
        self.isWithin = True

    def setWithinCov(self, withinCov):
        self.isWithin = withinCov

    def compute(self, model, D, L):
        C = cov_mat(D) if not self.isWithin else within_cov_mat(D, L)
        C = C ** 0.5
        return model, numpy.dot(C, D), L

    def __str__(self):
        return "Whitening"

def ZNorm_f(D):
    DC = center_data(D)
    C = numpy.dot(DC, DC.T) / D.shape[1]
    stdDev = mcol(numpy.diag(C)) ** 0.5
    return DC / stdDev

class ZNorm(PipelineStage):
    def __init__(self):
        super().__init__()

    def compute(self, model, D, L):
        return model, ZNorm_f(D), L

    def __str__(self):
        return "Z Normalization"

class L2Norm(PipelineStage):
    def __init__(self):
        super().__init__()

    def compute(self, model, D, L):
        DC = center_data(D)
        DN = DC / numpy.linalg.norm(DC, axis=0)
        return model, DN, L

    def __str__(self):
        return "L2 Normalization"
