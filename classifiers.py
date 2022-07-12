import numpy
import scipy.optimize
import matplotlib.pyplot as plt

from Pipeline import PipelineStage
from Tools import mcol, vec, mrow, LBG_x2_Cluster, logpdf_GMM
from models import MVGModel, TiedMVGModel, LogRegModel, SVMModel, GMMModel


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

        newModel = MVGModel(K, u, C)
        newModel.setPreproc(model.preproc)

        return newModel, D, L

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

        newModel = MVGModel(K, u, C)
        newModel.setPreproc(model.preproc)

        return newModel, D, L

    def __str__(self):
        return 'NaiveBayesMVG'

class TiedMVG(PipelineStage):

    def __init__(self):
        super().__init__()
        self.piT = 0.5
        self.u = None
        self.C = None
        return
    
    def setPiT(self, pi):
        self.piT = pi
        return self

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
            C += numpy.dot(DClC, DClC.T) * (self.piT if i == 1 else (1-self.piT))
        # divide the partial covariance by the number of samples
        C /= nSamples

        self.u = u
        self.C = C

        newModel = TiedMVGModel(K, u, C)
        newModel.setPreproc(model.preproc)

        return newModel, D, L

    def __str__(self):
        return 'TiedMVG piT= %.1f' % self.piT

class TiedNaiveBayesMVG(PipelineStage):

    def __init__(self):
        super().__init__()
        self.piT = 0.5
        self.u = None
        self.C = None
        return
    
    def setPiT(self, pi):
        self.piT = pi
        return self

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
            C += numpy.diag((DClC**2).sum(1)) * (self.piT if i == 1 else (1-self.piT))
        # divide the partial covariance by the number of samples
        C /= nSamples

        self.u = u
        self.C = C

        newModel = TiedMVGModel(K, u, C)
        newModel.setPreproc(model.preproc)

        return newModel, D, L

    def __str__(self):
        return 'TiedNaiveBayesMVG piT= %.1f' % self.piT

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
        return self

    def setPiT(self, pi):
        self.piT = pi
        return self

    def setExpanded(self, isExpanded):
        self.isExpanded = isExpanded
        return self

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
            self.DT = phix[:, L == 1]
            self.DF = phix[:, L == 0]

        # find the minimum of function J(w, b)
        bestParam, minimum, d = scipy.optimize.fmin_l_bfgs_b(
            self.J,
            numpy.zeros((self.dim + 1) if not self.isExpanded else (self.dim**2 + self.dim + 1)),
            # the starting point is not important because the function is convex
            approx_grad=True,
            factr=1.0  # Improve the precision
        )

        # print("J(w,b): ", minimum)

        wBest = mcol(bestParam[0:(self.dim if not self.isExpanded else (self.dim**2 + self.dim))])  # (D, 1)
        bBest = bestParam[-1]  # scalar
        
        self.w = wBest
        self.b = bBest
        self.min = minimum

        newModel = LogRegModel(wBest, bBest, self.isExpanded)
        newModel.setPreproc(model.preproc)
        
        return newModel, D, L

    def __str__(self):
        return 'LogReg %s piT= %.1f' % (("Quadratic" if self.isExpanded else "Linear"), self.piT)
    
class SVM(PipelineStage):

    polyKernel = (lambda X_i, X_j: numpy.dot(X_i.T, X_j))
    RBFKernel = (lambda X_i, X_j: numpy.dot(X_i.T, X_j))

    def __init__(self):
        super().__init__()
        self.piT = 0.5
        self.C = 0  # C = 0 => Hard-Margin SVM
        self.H = None  # (N, N)
        self.Z = None  # (1, N)
        self.D = None  # (M, N)
        # As K becomes larger, the effects of regularizing b become weaker. However, as K becomes larger, the
        # dual problem also becomes harder to solve (i.e. the algorithm may require many additional iterations).
        self.K = 1  # Default 1
        # default kernel is just a dot product
        # this is a formula in order to work also with matrices
        # x_i.T * x_j
        self.kernel = (lambda X_i, X_j: numpy.dot(X_i.T, X_j))
        self.isNoKern = True
        
    def setK(self, K):
        self.K = K

    def setC(self, C):
        self.C = C

    def setPiT(self, pi):
        self.piT = pi

    def setPolyKernel(self, c, d):
        """
        polynomial kernel of degree d
        k(x_1, x_2) = (x_1.T * x_2 + c)^d
        """
        self.isNoKern = False
        self.kernel = (lambda X_i, X_j: numpy.power(numpy.dot(X_i.T, X_j) + c, d) + self.K**2)

    def setRBFKernel(self, g):
        """
        Radial basis function kernel
        K(x_1, x_2) = exp(-g*||x_1-x_2||^2)
        """
        self.isNoKern = False

        def RBFKernel(X_i, X_j):
            """
            The principal problem here is that X_i  and X_j don't have the same shape
            So the difference has some problems
            Here we do a trick in order to do it without loops
            ...Solution taken from Internet
            """
            a = numpy.repeat(X_i, X_j.shape[1], axis=1)
            b = numpy.tile(X_j, X_i.shape[1])
            c = (numpy.linalg.norm(a - b, axis=0) ** 2).reshape((X_i.shape[1], X_j.shape[1]))
            return numpy.exp(-g * c) + self.K**2

        self.kernel = (lambda X_i, X_j: RBFKernel(X_i, X_j))

    def setNoKern(self):
        """
        default kernel is just a dot product
        this is a formula in order to work also with matrices
        x_i.T * x_j
        """
        self.isNoKern = True
        self.kernel = (lambda X_i, X_j: numpy.dot(X_i.T, X_j))

    def compute_H(self):
        """
        H_ij = z_i * z_j * x_i.T * x_j.T
        This function compute this calculus for all elements i,j inside D, by BROADCASTING
        numpy.dot(Z.T, Z) return the a matrix (N, N) in which elements are: H_ij = z_i * z_j
        Use the kernel passed thought setKernel or the default to compute k(x_i, x_j)
        """
        X = self.D
        if not self.isNoKern:
            """
            If we are doing a non linear SVM, the kernel MUST be computed with the normal Dataset
            so here I'm removing the additional K feature
            N.B.
                Internally of the non linear kernels i.e. Poly or RBF, is added a constant eps = K^2 
                in order to regularize the bias that here we are removing  
            """
            X = X[:-1, :]
        return numpy.dot(self.Z.T, self.Z) * self.kernel(X, X)

    def L_Dual(self, a):
        """
        Just following the formula of the dual problem in Matrix form
        Here is used the modified version
        D = [[D], [K]]
        w = [[w], [b]]
        Returns also the gradient for best performance
        """
        a = mcol(a)
        """
        Compute the formula for the lagrangian
        J^D(a) = - 1/2*a.T*H*a + a.T*1
        return a lagrangian and gradient
        """
        Ha = numpy.dot(self.H, a)
        ones = numpy.ones(a.shape)
        res = 0.5 * numpy.dot(a.T, Ha) - numpy.dot(a.T, ones)
        '''
        Compute the gradient of Lagrangian
        Gradient = - H*a + 1        
        '''
        gradient = Ha - ones
        return res, gradient.ravel()

    def getWeightsFromDual(self, a):
        """
        Reconstruct w (Primal Solution) from a (Dual solution)
        We do not need to reconstruct b, because we are using the modified version
        D = [[D], [K]]
        w = [[w], [b]]

        Here I'm doing sum(i = 1 to n) of a_i * z_i * x_i
        Everything is done by broadcasting, so we do for all sample in a single shot
        Z must be Transposed (colum vector)
        """
        return mcol(numpy.sum(a * self.Z * self.D, 1))

    def L_primal(self, w):
        """
        This function is used only for checks
        We want that the dual solution to be the most as close as possible to the primal
        Here I'm using the modified version, so b is inside w
        D = [[D], [K]]
        w = [[w], [b]]
        """
        loss = 1 - mrow(self.Z) * numpy.dot(w.T, self.D)  # (1, N)
        loss[loss <= 0] = 0  # max(0, hinge loss)
        return 0.5 * numpy.linalg.norm(w) ** 2 + self.C * numpy.sum(loss, 1)

    def compute(self, model, D, L):
        """
        this is a solution for the modified problem
        D = [[D], [K]]
        w = [[w], [b]]
        """
        N = D.shape[1]  # Number of samples
        self.D = numpy.vstack((D, numpy.full((1, N), self.K)))
        self.Z = mrow((L * 2.0) - 1.0)  # 0 => -1 ; 1 => 1
        self.H = self.compute_H()
        piTEmp = numpy.sum(L == 1) / N
        bounds = [(0, (self.C*self.piT/piTEmp if cl == 1 else self.C*(1-self.piT)/(1-piTEmp)) if self.C != 0 else None) for cl in L]

        aBest, minimum, d = scipy.optimize.fmin_l_bfgs_b(
            self.L_Dual,
            numpy.zeros(N),
            # The gradient is directly computed inside the L_Dual funct
            bounds=bounds,
            # Those are the bounds for the soft-margin solution
            factr=1.0,
            # improve the precision
            maxiter=10**9, maxfun=10**9  # increase the max number of iterations
            # iprint=1
        )

        # CHECK can be removed to save time, is used to control if everything is done well
        # if self.isNoKern:
        #     wBest = self.getWeightsFromDual(aBest)  # wBest = [[w], [b]]
        #     print("Duality GAP: ", self.L_primal(wBest) + self.L_Dual(aBest)[0])
        # print("Dual loss :", -self.L_Dual(aBest)[0])

        newModel = SVMModel(aBest, self.K, self.kernel, self.isNoKern, self.D, L)
        newModel.setPreproc(model.preproc)

        return newModel, D, L

    def __str__(self):
        return "SVM %s piT=%.1f" % (("hard-margin" if self.C == 0 else "soft-margin"), self.piT)

class GMM(PipelineStage):

    def __init__(self):
        super().__init__()
        self.psi = 0.01
        self.alpha = 0.1
        self.i = 2
        self.isTied = False
        self.isDiagonal = False

    def setDiagonal(self, diag):
        self.isDiagonal = diag

    def setTied(self, tied):
        self.isTied = tied

    def setIterationLBG(self, i):
        self.i = i

    def setAlpha(self, alpha):
        self.alpha = alpha

    def setPsi(self, psi):
        self.psi = psi

    def compute(self, model, D, L):
        K = L.max() + 1

        GMMs = []
        for i in range(K):
            # print("Class ", i)
            DCl = D[:, L == i]

            # Start gmm
            mu = numpy.mean(DCl, axis=1)
            XC = DCl - mcol(mu)
            C = numpy.dot(XC, XC.T) / DCl.shape[1]
            gmm = [(1.0, mcol(mu), C)]

            GMMs.append(LBG_x2_Cluster(DCl, gmm, self.alpha, self.i, self.psi, self.isDiagonal, self.isTied))

        newModel = GMMModel(K, GMMs)
        newModel.setPreproc(model.preproc)

        return newModel, D, L

    def __str__(self):
        desc = ""
        if self.isDiagonal and self.isTied:
            desc += "GMM Diagonal-Tied Covariance "
        elif self.isDiagonal:
            desc += "GMM Diagonal Covariance "
        elif self.isTied:
            desc += "GMM Tied Covariance "
        else:
            desc += "GMM Full Covariance "

        return desc + "comps=%s alpha=%s psi=%s" % (2**self.i, self.alpha, self.psi)
