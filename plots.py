import matplotlib.pyplot as plt
from Pipeline import PipelineStage


class Histogram(PipelineStage):

    def __init__(self):
        super().__init__()
        self.labels = []
        self.dimensions = []

    def setLabels(self, labels):
        self.labels = labels

    def setDimensions(self, dimensions):
        self.dimensions = dimensions

    def compute(self, model, D, L):
        K = L.max() + 1
        dims = D.shape[0]
        for dim in range(dims):
            plt.figure()
            plt.xlabel(dim if len(self.dimensions) == 0 else self.dimensions[dim])
            for k in range(K):
                plt.hist(D[dim, L == k], bins=10, density=True, alpha=0.4, label=(k if len(self.labels) == 0 else self.labels[k]))
            plt.legend()
        plt.show()
        return model, D, L

    def __str__(self):
        return "Histogram\n"

class Scatter(PipelineStage):

    def __init__(self):
        super().__init__()
        self.labels = []
        self.dimensions = []

    def setLabels(self, labels):
        self.labels = labels

    def setDimensions(self, dimensions):
        self.dimensions = dimensions

    def compute(self, model, D, L):
        K = L.max() + 1
        dims = D.shape[0]
        for attri in range(dims):
            for attrj in range(dims):
                if attri >= attrj:
                    continue
                plt.figure()
                plt.xlabel(attri if len(self.dimensions) == 0 else self.dimensions[attri])
                plt.ylabel(attrj if len(self.dimensions) == 0 else self.dimensions[attrj])
                for k in range(K):
                    plt.scatter(D[attri, L == k], D[attrj, L == k], label=(k if len(self.labels) == 0 else self.labels[k]))
                plt.legend()
            plt.show()
        return model, D, L

    def __str__(self):
        return "Scatter\n"
