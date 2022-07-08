import matplotlib.pyplot as plt
import seaborn as sns
import numpy

from Pipeline import PipelineStage
from Tools import confusion_matrix, DCF_norm_bin, DCF_min, pearson_correlation_mat


class Histogram(PipelineStage):

    def __init__(self):
        super().__init__()
        self.dpi = None
        self.name = None
        self.save = False
        self.title = ""
        self.perBin = 15
        self.labels = []
        self.dimensions = []

    def setLabels(self, labels):
        self.labels = labels

    def setDimensions(self, dimensions):
        self.dimensions = dimensions

    def setElemPerBin(self, n):
        self.perBin = n

    def setTitle(self, title):
        self.title = title

    def setSaveDirectoryDPI(self, directory, name, ext, dpi):
        self.save = True
        self.name = directory + "/" + name + "." + ext
        self.dpi = dpi

    def compute(self, model, D, L):
        K = L.max() + 1
        dims = D.shape[0]
        N = D.shape[1]
        for dim in range(dims):
            plt.figure()
            plt.title(self.title)
            plt.xlabel(dim if len(self.dimensions) == 0 else self.dimensions[dim])
            for k in range(K):
                plt.hist(D[dim, L == k], bins=round(N/self.perBin), density=True, alpha=0.4, label=(k if len(self.labels) == 0 else self.labels[k]))
            plt.legend()
            if self.save:
                plt.savefig(self.name, dpi=self.dpi)
            plt.show()
        return model, D, L

    def __str__(self):
        return "Histogram\n"

class Scatter(PipelineStage):

    def __init__(self):
        super().__init__()
        self.save = False
        self.dpi = None
        self.name = None
        self.labels = []
        self.dimensions = []
        self.title = ""

    def setLabels(self, labels):
        self.labels = labels

    def setDimensions(self, dimensions):
        self.dimensions = dimensions

    def setTitle(self, title):
        self.title = title

    def setSaveDirectoryDPI(self, directory, name, ext, dpi):
        self.save = True
        self.name = directory+"/"+name+"."+ext
        self.dpi = dpi

    def compute(self, model, D, L):
        K = L.max() + 1
        dims = D.shape[0]
        for attri in range(dims):
            for attrj in range(dims):
                if attri >= attrj:
                    continue
                plt.figure()
                plt.title(self.title)
                plt.xlabel(attri if len(self.dimensions) == 0 else self.dimensions[attri])
                plt.ylabel(attrj if len(self.dimensions) == 0 else self.dimensions[attrj])
                for k in range(K):
                    plt.scatter(D[attri, L == k], D[attrj, L == k], label=(k if len(self.labels) == 0 else self.labels[k]))
                plt.legend()
                if self.save:
                    plt.savefig(self.name, dpi=self.dpi)
                plt.show()
        return model, D, L

    def __str__(self):
        return "Scatter\n"

def print_ROCs(llrs, L, titles):
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    for i in range(len(llrs)):
        ts = numpy.array(llrs[i])
        ts.sort()
        ts = numpy.concatenate((numpy.array([-numpy.inf]), ts, numpy.array([+numpy.inf])))
        x = []
        y = []
        for t in ts:
            P = (llrs[i] > t) + 0
            M = confusion_matrix(P, L)
            TN, FN, FP, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
            FPR = FP/(TN+FP)
            FNR = FN/(FN+TP)
            TPR = 1 - FNR
            x.append(FPR)
            y.append(TPR)
        plt.scatter(x, y, s=4, label=titles[i])
    plt.legend()
    plt.show()

def bayes_error_addToPlot(llr, L, title):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    mindcf = []
    dcf = []
    for effPriorLogOdd in effPriorLogOdds:
        effPrior = 1/(1+numpy.exp(-effPriorLogOdd))
        dcf.append(DCF_norm_bin(llr, L, effPrior, 1, 1))  # actually
        mindcf.append(DCF_min(llr, L, effPrior, 1, 1))  # minimum

    plt.plot(effPriorLogOdds, dcf, label="DCF "+title.__str__())
    plt.plot(effPriorLogOdds, mindcf, label="min DCF "+title.__str__())

def print_DCFs(llrs, L, descriptions):
    plt.figure()
    plt.xlabel("log-odds")
    plt.ylabel("DCF value")
    for i in range(len(llrs)):
        bayes_error_addToPlot(llrs[i], L, descriptions[i])
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.show()

def print_pearson_correlation_mat(D, title):
    dim = D.shape[0]
    plt.figure(figsize=(dim, dim))
    corrMatr = pearson_correlation_mat(D)
    heatmap = sns.heatmap(corrMatr, vmin=-1, vmax=1, annot=True, cmap='Reds')
    heatmap.set_title('Pearson correlation matrix '+title, fontdict={'fontsize': 12}, pad=12)
    plt.show()

def print_pearson_correlation_matrices(D, L):
    K = L.max() + 1
    print_pearson_correlation_mat(D, "whole data")
    for k in range(K):
        print_pearson_correlation_mat(D[:, L == k], str(k))
