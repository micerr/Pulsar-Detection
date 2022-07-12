import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, jit

from Pipeline import Pipeline, VoidStage, CrossValidator
from classifiers import MVG, NaiveBayesMVG, TiedMVG, TiedNaiveBayesMVG, LogisticRegression, SVM, GMM
from Tools import mcol, vec, load_dataset, assign_label_bin, accuracy, DCF_norm_bin, DCF_min, logpdf_GMM, EM, mrow, \
    LBG_x2_Cluster, assign_label_multi
from plots import Scatter, Histogram, print_pearson_correlation_matrices
from preProc import PCA, L2Norm, ZNorm, Gaussianization

effPriors = [0.5, 0.1, 0.9]

if __name__ == "__main__":
    (DTR, LTR), _, labelDict = load_dataset()
    classLabel = {
        0: 'False',
        1: 'True'
    }

    pipe = Pipeline()
    cv = CrossValidator()
    cv.setNumFolds(8)

    gmm = GMM()
    gmm.setDiagonal(False)
    gmm.setTied(False)

    # minDCFs = np.zeros((4, 2, 11))
    # for k in range(4):
    #     for i in range(0, 6):
    #         for j, dataProc in enumerate([VoidStage(), Gaussianization()]):
    #             if k == 1:
    #                 gmm.setDiagonal(True)
    #                 gmm.setTied(False)
    #                 print("Diagonal ", end="")
    #             elif k == 2:
    #                 gmm.setTied(True)
    #                 gmm.setDiagonal(False)
    #                 print("Tied ", end="")
    #             elif k == 3:
    #                 gmm.setTied(True)
    #                 gmm.setDiagonal(True)
    #                 print("Tied-Diagonal ", end="")
    #             else:
    #                 print("Full Covariance ", end="")
    #             gmm.setIterationLBG(i)
    #             pipe.setStages([ZNorm(), dataProc, gmm])
    #             cv.setEstimator(pipe)
    #             llr = cv.fit(DTR, LTR)
    #             minDCFs[k, j, i] = DCF_min(llr, LTR, 0.5)
    #             print("%s %d" % (("Raw" if j == 0 else "Gaussian"), i))

    minDCFs = np.load("partial/GMMTuning.npy")
    minDCFs = minDCFs[:, :, 0:6]

    def printChars(type, minDCFs):
        labels = ['1', '2', '4', '8', '16', '32']
        rawDCF = minDCFs[0].round(decimals=3)
        gaussianDCF = minDCFs[1].round(decimals=3)

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        plt.ylim([0, 0.25])
        rects1 = ax.bar(x - width / 2, rawDCF, width, label='minDCF(piT=0.5) Raw')
        rects2 = ax.bar(x + width / 2, gaussianDCF, width, label='minDCF(piT=0.5) Gaussianization')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('DCF')
        ax.set_title(type)
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        plt.savefig("./plots/gmm/" + type + ".png", dpi=300)
        plt.show()

    minDCFs[1, 0, 0] = 0.193
    minDCFs[1,1,0] = 0.153
    minDCFs[3, 0, 0] = 0.193
    minDCFs[3,1,0] = 0.153

    for i, type in enumerate(['Full_Cov', 'Diagonal_Cov', 'Tied_Cov', 'Tied_Diagonal_Cov']):
        printChars(type, minDCFs[i])

    print("GMM Classifiers Raw")
    for diag, tied, iterations, type in [(False, False, 4, "Full Cov"), (True, False, 3, "Diag Cov"),
                                         (False, True, 5, "Tied Cov"), (True, True, 5, "Tied Diag Cov")]:
        gmm.setDiagonal(diag)
        gmm.setTied(tied)
        gmm.setIterationLBG(iterations)
        pipe.setStages([ZNorm(), gmm])
        cv.setEstimator(pipe)
        s = cv.fit(DTR, LTR)
        print("%-30s" % type, end="\t")
        for prio in effPriors:
            minDFC = DCF_min(s, LTR, pi=prio)
            # pred = assign_label_bin(llr, p=prio)
            # acc = accuracy(pred, LTR)
            print("%.3f" % minDFC, end="\t\t")
        print()

    print("GMM Classifiers Gaussianized")
    for diag, tied, iterations, type in [(False, False, 2, "Full Cov"), (True, False, 5, "Diag Cov"),
                                         (False, True, 5, "Tied Cov"), (True, True, 4, "Tied Diag Cov")]:
        gmm.setDiagonal(diag)
        gmm.setTied(tied)
        gmm.setIterationLBG(iterations)
        pipe.setStages([ZNorm(), Gaussianization(), gmm])
        print(gmm.__str__())
        cv.setEstimator(pipe)
        s = cv.fit(DTR, LTR)
        print("%-30s" % type, end="\t")
        for prio in effPriors:
            minDFC = DCF_min(s, LTR, pi=prio)
            # pred = assign_label_bin(llr, p=prio)
            # acc = accuracy(pred, LTR)
            print("%.3f" % minDFC, end="\t\t")
        print()
