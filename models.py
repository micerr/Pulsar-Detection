import numpy
import scipy
import scipy.special

from Pipeline import Model
from Tools import logpdf_GAU_ND, mrow, mcol, vec, logpdf_GMM


class GenerativeModel(Model):

    def __init__(self, K, mu, C):
        super().__init__()
        self.mu = mu
        self.C = C
        self.K = K  # num Classes
        self.prior = None
        return

    def setPrior(self, prior):
        self.prior = mcol(numpy.array(prior))
        return

    def logLikelihood(self, ll, D, mu, C):
        pass

    def transform(self, D, L):
        D, L = super().transform(D, L)
        K = self.K
        nSamples = D.shape[1]
        ll = numpy.zeros((K, nSamples))  # array of log likelihoods vectors
        if self.prior is None:
            self.prior = mcol(numpy.ones(K) / float(K))

        # Compute log-likelihood
        ll = self.logLikelihood(ll, D, self.mu, self.C)  # (K, N)

        if K == 2:
            # Binary
            llr = ll[1] - ll[0]
            return llr  # (1, N)

        logSJoint = ll + numpy.log(self.prior)
        # Can we skip the below part in order to get time, see slide 36 of GenerativeLinearQuadratic
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = numpy.exp(logSPost)
        return ll

class MVGModel(GenerativeModel):

    def __init__(self, K, mu, C):
        super().__init__(K, mu, C)

    def setPrior(self, prior):
        super().setPrior(prior)

    def logLikelihood(self, ll, D, mu, C):
        # Compute log-likelihood
        for i in range(self.K):
            ll[i:i + 1, :] = mrow(logpdf_GAU_ND(D, self.mu[:, i], self.C[i]))
        return ll

    def transform(self, D, L):
        return super().transform(D, L)

class TiedMVGModel(GenerativeModel):

    def __init__(self, K, mu, C):
        super().__init__(K, mu, C)

    def setPrior(self, prior):
        super().setPrior(prior)

    def logLikelihood(self, ll, D, mu, C):
        # Compute log-likelihood
        for i in range(self.K):
            ll[i:i + 1, :] = mrow(logpdf_GAU_ND(D, self.mu[:, i], self.C))
        return ll

    def transform(self, D, L):
        return super().transform(D, L)

class LogRegModel(Model):

    def __init__(self, w, b, isExpanded):
        super().__init__()
        self.w = w
        self.b = b
        self.isExpanded = isExpanded

    def transform(self, D, L):
        D, L = super().transform(D, L)
        """
        Here we compute the score vector, is a vector with NumSample elements
        it's computed by :      s(xt) = w.T * xt + b
        """
        if self.isExpanded:
            nSamples = D.shape[1]
            dim = D.shape[0]
            phix = numpy.zeros((dim ** 2 + dim, nSamples))

            for i in range(nSamples):
                x = D[:, i:i + 1]
                phix[:, i:i + 1] = numpy.vstack((vec(numpy.dot(x, x.T)), x))

            D = phix

        llrPost = numpy.dot(self.w.T, D) + self.b  # (1, NumSamples)
        return llrPost

class SVMModel(Model):

    def __init__(self, a, K, kernel, isNoKern, DTR, LTR):
        super().__init__()
        self.a = mcol(a)
        self.K = K
        self.kernel = kernel
        self.isNoKern = isNoKern
        self.DTR = DTR
        self.LTR = LTR

    def transform(self, D, L):
        D, L = super().transform(D, L)
        N = D.shape[1]
        """
        If we are using a non-linear SVM kernel the modified version is implicitly done inside the
        calculus of the kernel, adding +K^2 factor, so we don't need to redo it again
        Here I add the additional K feature only if the kernel is linear
        """
        D = numpy.vstack((D, numpy.full((1, N), self.K))) if self.isNoKern else D
        Z = mcol((self.LTR * 2.0) - 1.0)  # 0 => -1 ; 1 => 1
        ## TODO
        """
        TODO
        The SVM decision rule is to assign a pattern to class HT if the score is greater than 0, and to
        HF otherwise. However, SVM decisions are not probabilistic, and are not able to account for
        different class priors and mis-classification costs. Bayes decisions thus require either a score post-
        processing step, i.e. score calibration, or cross-validation to select the optimal threshold for a
        specific application. Below we simply use threshold 0 and compute the corresponding accuracy.
        """
        X = self.DTR
        if not self.isNoKern:
            """
            If we are doing a non linear SVM, the kernel MUST be computed with the normal Dataset
            so here I'm removing the additional K feature
            N.B.
                Internally of the non linear kernels i.e. Poly or RBF, is added a constant eps = K^2 
                in order to regularize the bias that here we are removing  
            """
            X = X[:-1, :]
        """
        Here I compute the score with the following formula
        s(xt) = sum(from 1 to n) of[ a_i * z_i * k(x_i, x_t)]
        I do everything with broadcasting
        """
        S = numpy.dot((self.a * Z).T, self.kernel(X, D))
        return S

class GMMModel(Model):
    def __init__(self, K, GMMs):
        super().__init__()
        self.prior = None
        self.K = K
        self.GMMs = GMMs

    def setPrior(self, prior):
        self.prior = mcol(numpy.array(prior))
        return

    def transform(self, D, L):
        D, L = super().transform(D, L)
        K = self.K
        nSamples = D.shape[1]
        ll = numpy.zeros((K, nSamples))  # array of log likelihoods vectors
        if self.prior is None:
            self.prior = mcol(numpy.ones(K) / float(K))

        # Compute log-likelihood
        for c in range(K):
            ll[c:c+1, :] = logpdf_GMM(D, self.GMMs[c])

        if K == 2:
            # Binary
            llr = ll[1] - ll[0]
            return llr

        logSJoint = ll + numpy.log(self.prior)
        # Can we skip the below part in order to get time, see slide 36 of GenerativeLinearQuadratic
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = numpy.exp(logSPost)
        return ll
