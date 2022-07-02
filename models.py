import numpy
import scipy
import scipy.special

from Pipeline import Model
from Tools import logpdf_GAU_ND, mrow, mcol


class MVGModel(Model):

    def __init__(self, mu, C):
        super().__init__()
        self.mu = mu
        self.C = C
        self.prior = None
        return

    def setPrior(self, prior):
        self.prior = mcol(numpy.array(prior))
        return

    def transform(self, D, L):
        K = L.max() + 1
        nSamples = D.shape[1]
        ll = numpy.zeros((K, nSamples))  # array of log likelihoods vectors
        if self.prior is None:
            self.prior = mcol(numpy.ones(K) / float(K))

        # Compute log-likelihood
        for i in range(K):
            ll[i:i + 1, :] = mrow(logpdf_GAU_ND(D, self.mu[:, i], self.C[i]))

        if K == 2:
            # Binary
            prior = self.prior[0]
            llr = ll[0] - ll[1]
            threshold = - numpy.log(prior/(1-prior))
            
            pass
        else:
            # Multiclass
            logSJoint = ll + self.prior
            # Can we skip the below part in order to get time ? see slide 36 of GenerativeLinearQuadratic
            logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
            logSPost = logSJoint - logSMarginal
            SPost = numpy.exp(logSPost)
            return SPost

    def __str__(self):
        return

