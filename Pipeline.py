import numpy

from Tools import assign_label_bin, accuracy, DCF_norm_bin, DCF_min


class Pipeline:

    def __init__(self):
        self.stages = []
        return

    def setStages(self, stages):
        self.stages = stages

    def addStages(self, stages):
        self.stages += stages

    def fit(self, D, L, verbose=False):
        model = Model()
        for stage in self.stages:
            model, D, L = stage.compute(model, D, L)
            if verbose:
                print(stage)
        return model

class PipelineStage:

    def __init__(self):
        pass

    def compute(self, model, D, L):
        pass

    def __str__(self):
        pass

class VoidStage(PipelineStage):

    def __init__(self):
        super().__init__()

    def compute(self, model, D, L):
        return model, D, L

    def __str__(self):
        return "Raw"

class Model:

    def __init__(self):
        self.P = None

    def transform(self, D, L):
        pass

    def setP(self, P):
        self.P = P

class CrossValidator:

    def __init__(self):
        self.Cfn = 1
        self.Cfp = 1
        self.pi = 0.5
        self.k = None
        self.pipeline = None

    def setEstimator(self, pipeline):
        self.pipeline = pipeline
        return

    def setEstimatorParams(self, pi, Cfn, Cfp):
        self.pi = pi
        self.Cfn = Cfn
        self.Cfp = Cfp

    def setNumFolds(self, k):
        self.k = k
        if k < 2:
            self.k = 2

    def fit(self, D, L):

        K = L.max() + 1
        nSamples = D.shape[1]
        if nSamples % self.k != 0:
            sizeFold = nSamples//self.k + 1
        else:
            sizeFold = nSamples / self.k
        sizeFold = int(sizeFold)

        numpy.random.seed(nSamples*K)
        idx = numpy.random.permutation(nSamples)

        llr = numpy.zeros((1, nSamples))
        for i in range(self.k):
            # divide the random numbers in Keff-fold parts
            idxTest = idx[(i * sizeFold):((i + 1) * sizeFold)]
            idxTrain = numpy.append(idx[:(i * sizeFold)], idx[((i + 1) * sizeFold):])

            DTR = D[:, idxTrain]
            LTR = L[idxTrain]
            DTE = D[:, idxTest]
            LTE = L[idxTest]

            model = self.pipeline.fit(DTR, LTR)
            if model.P is not None:
                DTE = numpy.dot(model.P.T, DTE)
            llr[:, idxTest] = model.transform(DTE, LTE)

        # pred = assign_label_bin(llr, self.pi, self.Cfn, self.Cfp)
        # acc = accuracy(pred, L)
        # print("Error:\t",  (1-acc)*100, "%")
        # bCost = DCF_norm_bin(llr, L, self.pi, self.Cfn, self.Cfp)
        # minCost = DCF_min(llr, L, self.pi, self.Cfn, self.Cfp)
        # print("DCF norm:\t", bCost, "\nDCF min:\t", minCost, "\n")

        return llr
