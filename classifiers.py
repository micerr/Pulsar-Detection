import numpy
import scipy.optimize

from Pipeline import PipelineStage
from Tools import mcol, vec
from models import MVGModel, TiedMVGModel, LogRegModel


class MVG(PipelineStage):

    def __init__(self):
        super().__init__()
        self.u = None
        self.C = None
        return

    def compute(self, model, D, L):
        dim = D.shape[0]
        K = L.max()+1

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
            C[i] = numpy.dot(DClC, DClC.T) / DCl.shape[1]

        self.u = u
        self.C = C

        return MVGModel(K, u, C), D, L

    def __str__(self):
        return 'MVG'

class NaiveBayesMVG(PipelineStage):

    def __init__(self):
        super().__init__()
        self.u = None
        self.C = None
        return

    def compute(self, model, D, L):
        dim = D.shape[0]
        K = L.max()+1

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
            C[i] = numpy.diag((DClC**2).sum(1))/DCl.shape[1]

        self.u = u
        self.C = C

        return MVGModel(K, u, C), D, L

    def __str__(self):
        return 'NaiveBayesMVG'

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
        C = numpy.zeros((dim, dim))  # array of covariance matrices by class

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

        return TiedMVGModel(K, u, C), D, L

    def __str__(self):
        return 'TiedMVG'

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
        C = numpy.zeros((dim, dim))  # array of covariance matrices by class

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

        return TiedMVGModel(K, u, C), D, L

    def __str__(self):
        return 'TiedNaiveBayesMVG'

class LogisticRegression(PipelineStage):

    def __init__(self):
        super().__init__()
        self.DF = None  # Data labeled as False, in case of unbalanced data
        self.DT = None  # Data labeled as True, in case of unbalanced data
        self.min = None
        self.b = None
        self.w = None
        self.lmbd = None
        self.Z = None
        self.D = None
        self.dim = None
        self.piT = None  # balancer prior prob
        self.isExpanded = False
        
    def J(self, x):
        # J(w, b), in v there are D+1 elements, D element of array w, and the cost b
        """
        Compute and return the objective function value using DTR,
        LTR, l
        """
        w, b = x[0:(self.dim if not self.isExpanded else (self.dim**2 + self.dim))], x[-1]
        w = mcol(w)  # (D, 1) D := Dimensions
        """
        Here we are computing 1/n * sum (log (1 + exp(-z1*(w.T * xi + b)) ))
        The computation of logarithm can lead to numerical issues, so we use numpy.logaddexp
        the function compute log(exp(a) + exp(b)) 
            1 => exp(0)
        """
        if self.piT is None:
            S = numpy.dot(w.T, self.D) + b  # (1, N) The second operand of  exp(...)
            # Computing the log AND doing 1/n * sum(..)
            obj = numpy.logaddexp(0, -self.Z * S).mean()
            # The norm^2 can be calculated also with:  (w**2).sum()
            return self.lmbd * 0.5 * numpy.linalg.norm(w) ** 2 + obj
        else:
            """
            Z isn't used because we know that inside the sums there are only either z_i = 1
            or z_i = -1, for this reason I did -ST for the True case and +SF for the False case
            """
            ST = numpy.dot(w.T, self.DT) + b  # (1, N) The second operand of  exp(...)
            # Computing the log AND doing 1/nT * sum(..)
            objT = numpy.logaddexp(0, -ST).mean()

            SF = numpy.dot(w.T, self.DF) + b
            # Computing the log AND doing 1/nT * sum(..)
            objF = numpy.logaddexp(0, +SF).mean()

            # The norm^2 can be calculated also with:  (w**2).sum()
            return self.lmbd * 0.5 * numpy.linalg.norm(w) ** 2 + self.piT * objT + (1 - self.piT) * objF
    
    def setLambda(self, lmbd):
        self.lmbd = lmbd

    def setPiT(self, pi):
        self.piT = pi

    def setExpanded(self, isExpanded):
        self.isExpanded = isExpanded

    def compute(self, model, D, L):
        self.D = D
        self.DT = D[:, L == 1]
        self.DF = D[:, L == 0]
        self.dim = D.shape[0]
        self.Z = (L * 2.0) - 1.0

        if self.lmbd is None:
            print("Error LogReg: no Lambda is selected for the regularizer")
            return None, D, L

        if self.isExpanded:
            nSamples = D.shape[1]
            phix = numpy.zeros((self.dim ** 2 + self.dim, nSamples))

            for i in range(nSamples):
                x = D[:, i:i + 1]
                phix[:, i:i + 1] = numpy.vstack((vec(numpy.dot(x, x.T)), x))

            self.D = phix

        # find the minimum of function J(w, b)
        bestParam, minimum, d = scipy.optimize.fmin_l_bfgs_b(
            self.J,
            numpy.zeros((self.dim + 1) if not self.isExpanded else (self.dim**2 + self.dim + 1)),
            # the starting point is not important because the function is convex
            approx_grad=True
        )

        wBest = mcol(bestParam[0:(self.dim if not self.isExpanded else (self.dim**2 + self.dim))])  # (D, 1)
        bBest = bestParam[-1]  # scalar
        
        self.w = wBest
        self.b = bBest
        self.min = minimum
        
        return LogRegModel(wBest, bBest, self.isExpanded), D, L

    def __str__(self):
        return 'LogReg'
