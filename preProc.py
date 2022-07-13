import numpy
import scipy.special

from Pipeline import PipelineStage
from Tools import mcol, center_data, cov_mat, within_cov_mat


class PCA(PipelineStage):

    def __init__(self):
        super().__init__()
        self.m = 0
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

        model.addPreproc(PCAEval(P))

        return model, DP, L

    def setDimension(self, m):
        self.m = m

    def __str__(self):
        return 'PCA (m = %d)' % self.m

class PCAEval(PipelineStage):
    def __init__(self, P):
        super().__init__()
        self.P = P

    def compute(self, model, D, L):
        return model, numpy.dot(self.P.T, D), L

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

        model.addPreproc(LDAEval(P))

        return model, DP, L

    def setDimension(self, m):
        self.m = m

    def __str__(self):
        return "LDA"

class LDAEval(PipelineStage):
    def __init__(self, P):
        super().__init__()
        self.P = P

    def compute(self, model, D, L):
        return model, numpy.dot(self.P.T, D), L

class ZNorm(PipelineStage):
    def __init__(self):
        super().__init__()

    def compute(self, model, D, L):
        mu = mcol(D.mean(1))
        DC = D - mu
        C = numpy.dot(DC, DC.T) / D.shape[1]
        stdDev = mcol(numpy.diag(C)) ** 0.5

        model.addPreproc(ZNormEval(mu, stdDev))
        return model, DC / stdDev, L

    def __str__(self):
        return "Z Normalization"

class ZNormEval(PipelineStage):
    def __init__(self, mu, stdDev):
        super().__init__()
        self.mu = mu
        self.stdDev = stdDev

    def compute(self, model, D, L):
        return model, (D - self.mu)/self.stdDev, L

class L2Norm(PipelineStage):
    def __init__(self):
        super().__init__()

    def compute(self, model, D, L):
        mu = mcol(D.mean(1))
        DC = D - mu
        DN = DC / numpy.linalg.norm(DC, axis=0)

        model.addPreproc(L2NormEval(mu))
        return model, DN, L

    def __str__(self):
        return "L2 Normalization"

class L2NormEval(PipelineStage):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def compute(self, model, D, L):
        DC = D - self.mu
        return model, DC/numpy.linalg.norm(DC, axis=0), L

def cdf_GAU_STD(X):
    return 0.5*(1 + scipy.special.erf(X/numpy.sqrt(2)))

def inv_cdf_GAU_STD(X):
    return numpy.sqrt(2)*scipy.special.erfinv(2*X-1)

def empirical_cdf(X):
    N = X.shape[1]
    sort_feature = numpy.sort(X, axis=1)
    R = numpy.zeros(X.shape)
    for i in range(N):
        count = mcol(numpy.sum(sort_feature <= X[:, i:i+1], 1))
        R[:, i:i+1] = (count + 1) / (N + 2)
    return R

class Gaussianization(PipelineStage):
    """
    Gaussianization automatically implement also ZNormalization
    """

    def __init__(self):
        super().__init__()

    def compute(self, model, D, L):
        """
        Same thing can be done with rankdata() of scipy library
        """
        N = D.shape[1]
        sort_feature = numpy.sort(D, axis=1)
        R = numpy.zeros(D.shape)
        for i in range(N):
            count = mcol(numpy.sum(sort_feature <= D[:, i:i + 1], 1))
            R[:, i:i + 1] = (count + 1) / (N + 2)
        model.addPreproc(GaussEval(sort_feature, N))
        return model, inv_cdf_GAU_STD(R), L

    def __str__(self):
        return "Gaussianization"

class GaussEval(PipelineStage):
    def __init__(self, s, N):
        super().__init__()
        self.sort_feature = s
        self.N = N

    def compute(self, model, D, L):
        R = numpy.zeros(D.shape)
        for i in range(D.shape[1]):
            count = mcol(numpy.sum(self.sort_feature <= D[:, i:i + 1], 1))
            R[:, i:i + 1] = (count + 1) / (self.N + 2)
        return model, inv_cdf_GAU_STD(R), L
