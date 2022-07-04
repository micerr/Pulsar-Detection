import numpy

class Pipeline:

    def __init__(self):
        self.stages = []
        return

    def addStages(self, stages):
        self.stages += stages

    def fit(self, D, L, verbose=False):
        model = None
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

class Model:

    def __init__(self):
        pass

    def transform(self, D, L):
        pass

class CrossValidator:

    def __init__(self):
        self.k = None
        self.pipeline = None
        return

    def setEstimator(self, pipeline):
        self.pipeline = pipeline
        return

    def setEstimatorParams(self, params):
        # Implement
        return

    def setNumFolds(self, k):
        self.k = k
        if k < 2:
            self.k = 2
        return

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

        SPost = numpy.zeros((K, nSamples))
        pred = numpy.zeros((1, nSamples))
        for i in range(self.k):
            # divide the random numbers in Keff-fold parts
            idxTest = idx[(i * sizeFold):((i + 1) * sizeFold)]
            idxTrain = numpy.append(idx[:(i * sizeFold)], idx[((i + 1) * sizeFold):])

            DTR = D[:, idxTrain]
            LTR = L[idxTrain]
            DTE = D[:, idxTest]
            LTE = L[idxTest]

            model = self.pipeline.fit(DTR, LTR)
            SPost[:, idxTest], pred[:, idxTest] = model.transform(DTE, LTE)

        ## --------- EVALUATION ------------
        err = (pred != L).sum() / nSamples  # calculate the error of model
        print(err)
