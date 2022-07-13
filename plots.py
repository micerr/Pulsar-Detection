import matplotlib.pyplot as plt
import seaborn as sns
import numpy

from Pipeline import PipelineStage
from Tools import confusion_matrix, DCF_norm_bin, DCF_min, pearson_correlation_mat
from preProc import inv_cdf_GAU_STD


class Histogram(PipelineStage):

    def __init__(self):
        super().__init__()
        self.ext = None
        self.dpi = None
        self.name = None
        self.save = False
        self.title = ""
        self.perBin = 15
        self.labels = []
        self.dimensions = []

    def setLabels(self, labels):
        self.labels = labels
        return self

    def setDimensions(self, dimensions):
        self.dimensions = dimensions
        return self

    def setSizeBin(self, n):
        self.perBin = n
        return self

    def setTitle(self, title):
        self.title = title
        return self

    def setSaveDirectoryDPI(self, directory, name, ext, dpi):
        self.save = True
        self.name = directory + "/" + name
        self.ext = "." + ext
        self.dpi = dpi
        return self

    def compute(self, model, D, L):
        K = L.max() + 1
        dims = D.shape[0] if len(self.dimensions) == 0 else len(self.dimensions)
        N = D.shape[1]
        for dim in range(dims):
            plt.figure()
            plt.title(self.title)
            plt.xlabel(dim if len(self.dimensions) == 0 else self.dimensions[dim])
            for k in range(K):
                plt.hist(D[dim, L == k], bins=round(N/self.perBin), density=True, alpha=0.4, label=(k if len(self.labels) == 0 else self.labels[k]))
            plt.legend()
            if self.save:
                plt.savefig(self.name+"_"+self.title+"_"+self.dimensions[dim]+self.ext, dpi=self.dpi)
            plt.show()
        return model, D, L

    def __str__(self):
        return "Histogram\n"

class Scatter(PipelineStage):

    def __init__(self):
        super().__init__()
        self.ext = None
        self.save = False
        self.dpi = None
        self.name = None
        self.labels = []
        self.dimensions = []
        self.title = ""

    def setLabels(self, labels):
        self.labels = labels
        return self

    def setDimensions(self, dimensions):
        self.dimensions = dimensions
        return self

    def setTitle(self, title):
        self.title = title
        return self

    def setSaveDirectoryDPI(self, directory, name, ext, dpi):
        self.save = True
        self.name = directory+"/"+name
        self.ext = "." + ext
        self.dpi = dpi
        return self

    def compute(self, model, D, L):
        K = L.max() + 1
        dims = D.shape[0] if len(self.dimensions) == 0 else len(self.dimensions)
        for attri in range(dims):
            for attrj in range(dims):
                if attri >= attrj:
                    continue
                plt.figure()
                plt.title(self.title)
                plt.xlabel(attri if len(self.dimensions) == 0 else self.dimensions[attri])
                plt.ylabel(attrj if len(self.dimensions) == 0 else self.dimensions[attrj])
                for k in range(K):
                    plt.scatter(D[attri, L == k], D[attrj, L == k], alpha=0.1, label=(k if len(self.labels) == 0 else self.labels[k]))
                plt.legend()
                if self.save:
                    plt.savefig(self.name+"_"+self.title+"_"+self.dimensions[attri]+"_"+self.dimensions[attrj]+self.ext, dpi=self.dpi)
                plt.show()
        return model, D, L

    def __str__(self):
        return "Scatter\n"

def print_ROCs(llrs, L, titles, name):
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    for i in range(len(llrs)):
        ts = numpy.array(llrs[i])
        ts.sort()
        ts = numpy.concatenate((numpy.array([-numpy.inf]), ts.ravel(), numpy.array([+numpy.inf])))
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
        plt.plot(x, y, label=titles[i])
    plt.legend()
    plt.savefig("./plots/ROCDET/ROC"+name+".png", dpi=600)
    plt.show()

def print_DETs(ss, L, titles, name):
    figure, ax = plt.subplots()
    ax.set(xlabel="FPR", ylabel="FNR")
    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locations = inv_cdf_GAU_STD(numpy.array(ticks))
    tick_labels = [
        "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
        for s in ticks
    ]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-3, 2)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(-3, 1)

    for i in range(len(ss)):
        ts = numpy.array(ss[i])
        ts.sort()
        ts = numpy.concatenate((numpy.array([-numpy.inf]), ts.ravel(), numpy.array([+numpy.inf])))
        x = []
        y = []
        for t in ts:
            P = (ss[i] > t) + 0
            M = confusion_matrix(P, L)
            TN, FN, FP, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
            FPR = FP / (TN + FP)
            FNR = FN / (FN + TP)
            x.append(FPR)
            y.append(FNR)
        ax.plot(inv_cdf_GAU_STD(numpy.array(x)), inv_cdf_GAU_STD(numpy.array(y)), label=titles[i])
    ax.legend()
    plt.savefig("./plots/ROCDET/DET"+name+".png", dpi=600)


def bayes_error_addToPlot(llr, L, title, colo):
    effPriorLogOdds = numpy.linspace(-4, 4, 25)
    mindcf = []
    dcf = []
    for effPriorLogOdd in effPriorLogOdds:
        effPrior = 1/(1+numpy.exp(-effPriorLogOdd))
        dcf.append(DCF_norm_bin(llr, L, effPrior, 1, 1))  # actually
        mindcf.append(DCF_min(llr, L, effPrior, 1, 1))  # minimum

    plt.plot(effPriorLogOdds, dcf, label="DCF "+title.__str__(), color=colo)
    plt.plot(effPriorLogOdds, mindcf, label="min DCF "+title.__str__(), linestyle="--", color= colo)

def print_DCFs(llrs, L, descriptions, nameFigure, title):
    colors = ["b", "r", "g", "y"]
    plt.figure()
    plt.title(title)
    plt.xlabel("log-odds")
    plt.ylabel("DCF value")
    for i in range(len(llrs)):
        bayes_error_addToPlot(llrs[i], L, descriptions[i], colors[i])
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.legend()
    plt.savefig("./plots/DCF/"+nameFigure+".png", dpi=300)
    plt.show()

def print_pearson_correlation_mat(D, title, directory=None):
    dim = D.shape[0]
    plt.figure(figsize=(dim, dim))
    corrMatr = pearson_correlation_mat(D)
    heatmap = sns.heatmap(corrMatr, vmin=0, vmax=1, annot=True, cmap='gist_gray_r')
    heatmap.set_title('Pearson correlation matrix '+title, fontdict={'fontsize': 12}, pad=12)
    if directory is not None:
        plt.savefig(directory+"/"+title+".png", dpi=300)
    plt.show()

def print_pearson_correlation_matrices(D, L, labels, directory=None):
    K = L.max() + 1
    print_pearson_correlation_mat(D, "Dataset", directory)
    for k in range(K):
        print_pearson_correlation_mat(D[:, L == k], labels[k], directory)
