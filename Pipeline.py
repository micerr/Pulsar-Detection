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
