import numpy
import scipy
import scipy.special

from Pipeline import Model
from Tools import logpdf_GAU_ND, mrow, mcol


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

        # if K == 2:
        #     # Binary
        #     prior = self.prior[0]
        #     llr = ll[0] - ll[1]
        #     threshold = - numpy.log(prior/(1-prior))
        #     return llr > threshold
        # else:
        #     # Multiclass

        logSJoint = ll + numpy.log(self.prior)
        # Can we skip the below part in order to get time ? see slide 36 of GenerativeLinearQuadratic
        logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = numpy.exp(logSPost)
        return SPost

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
