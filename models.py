import numpy
import scipy
import scipy.special

from Pipeline import Model
from Tools import logpdf_GAU_ND, mrow, mcol, vec


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
        K = self.K
        nSamples = D.shape[1]
        ll = numpy.zeros((K, nSamples))  # array of log likelihoods vectors
        if self.prior is None:
            self.prior = mcol(numpy.ones(K) / float(K))

        # Compute log-likelihood
        ll = self.logLikelihood(ll, D, self.mu, self.C)

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
