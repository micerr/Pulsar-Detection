import numpy
import scipy

from Pipeline import PipelineStage
from Tools import mcol

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
        return 'PCA\nu = %s\nC = %s\n' % (self.mean, self.C)


class LDA(PipelineStage):

    def __init__(self):
        super().__init__()
        self.Sb = None
        self.Sw = None
        self.m = None

    def compute(self, model, D, L):
        K = L.max()+1
        m = self.m if self.m <= K-1 else K-1

        Sw = 0
        Sb = 0
        for i in range(K):
            DCl = D[:, L == i]  # take only samples of the class-i
            DClC = DCl - mcol(DCl.mean(1))  # center the data
            MC = mcol(DCl.mean(1)) - mcol(D.mean(1))  # center the mean of class, respect the global mean
            ## COMPUTING ELEMENT-I OF THE SUMMATORY OF Sb
            Sb += DClC.shape[1] * numpy.dot(MC, MC.T)
            ## COMPUTING ELEMENT-I OF THE SUMMATORY OF Sw
            Sw += numpy.dot(DClC, DClC.T)
        self.Sw = Sw = Sw / D.shape[1]
        self.Sb = Sb = Sb / D.shape[1]


        ## COMPUTING THE EIG VALUES OF THE GENERALIZED EIGENVALUE PROBLEM FOR HERMITIAN MATRICIES
        s, U = scipy.linalg.eigh(Sb, Sw)  # numpy here don't work, numpy don't solve the generalized problem
        P = U[:, ::-1][:, 0:m]  # take the  dimension

        ## PROJECTING DATA ON NEW BASE ##
        DP = numpy.dot(P.T, D)

        return model, DP, L

    def setDimension(self, m):
        self.m = m

    def __str__(self):
        return "LDA\nSw = %s\nSb = %s" % (self.Sw, self.Sb)
