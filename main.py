import numpy as np
import matplotlib.pyplot as plt

from Pipeline import Pipeline, VoidStage, CrossValidator
from classifiers import MVG, NaiveBayesMVG, TiedMVG, TiedNaiveBayesMVG, LogisticRegression, SVM, GMM
from Tools import mcol, vec, load_dataset, assign_label_bin, accuracy, DCF_norm_bin, DCF_min, logpdf_GMM, EM, mrow, \
    LBG_x2_Cluster, assign_label_multi
from plots import Scatter, Histogram, print_pearson_correlation_matrices
from preProc import PCA, L2Norm, ZNorm, Gaussianization

effPriors = [0.5, 0.1, 0.9]
lambdas = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6]
C = [10**-3, 10**-2, 10**-1, 10**0, 10**1]
K = [10**0, 10**1]
G = [10**-3, 10**-2, 10**-1, 10**0]

if __name__ == "__main__":
    (DTR, LTR), _, labelDict = load_dataset()
    classLabel = {
        0: 'False',
        1: 'True'
    }

    pipe = Pipeline()

    # print scatters and histograms
    scatter = Scatter()\
        .setTitle("RAW")\
        .setDimensions(labelDict)\
        .setLabels(classLabel)\
        .setSaveDirectoryDPI("./plots/scatter/raw", "", "png", 300)
    hist = Histogram() \
        .setTitle("RAW") \
        .setDimensions(labelDict)\
        .setLabels(classLabel)\
        .setSizeBin(200)\
        .setSaveDirectoryDPI("./plots/histogram/raw", "", "png", 300)

    # pipe.setStages([scatter, hist])
    # pipe.fit(DTR, LTR)
    # print_pearson_correlation_matrices(DTR, LTR, classLabel, "./plots/correlation")
#
    # scatter.setSaveDirectoryDPI("./plots/scatter/Znorm", "", "png", 300).setTitle("Znorm")
    # hist.setSaveDirectoryDPI("./plots/histogram/Znorm", "", "png", 300).setTitle("Znorm")
    # pipe.setStages([ZNorm(), scatter, hist])
    # pipe.fit(DTR, LTR)
#
    # scatter.setSaveDirectoryDPI("./plots/scatter/L2norm", "", "png", 300).setTitle("L2norm")
    # hist.setSaveDirectoryDPI("./plots/histogram/L2norm", "", "png", 300).setTitle("L2norm")
    # pipe.setStages([L2Norm(), scatter, hist])
    # pipe.fit(DTR, LTR)
#
    # scatter.setSaveDirectoryDPI("./plots/scatter/Gauss", "", "png", 300).setTitle("Gaussianized")
    # hist.setSaveDirectoryDPI("./plots/histogram/Gauss", "", "png", 300).setTitle("Gaussianized")
    # pipe.setStages([Gaussianization(), scatter, hist])
    # pipe.fit(DTR, LTR)
#
    # scatter.setSaveDirectoryDPI("./plots/scatter/Znorm/Gauss", "", "png", 300).setTitle("Znorm-Gaussianized")
    # hist.setSaveDirectoryDPI("./plots/histogram/Znorm/Gauss", "", "png", 300).setTitle("Znorm-Gaussianized")
    # pipe.setStages([ZNorm(), Gaussianization(), scatter, hist])
    # pipe.fit(DTR, LTR)
#
    # scatter.setSaveDirectoryDPI("./plots/scatter/L2norm/Gauss", "", "png", 300).setTitle("L2norm-Gaussianized")
    # hist.setSaveDirectoryDPI("./plots/histogram/L2norm/Gauss", "", "png", 300).setTitle("L2norm-Gaussianized")
    # pipe.setStages([L2Norm(), Gaussianization(), scatter, hist])
    # pipe.fit(DTR, LTR)


    # Generative Models
    mvg = MVG()
    mvgNaive = NaiveBayesMVG()
    mvgTied = TiedMVG()
    mvgTied1 = TiedMVG()
    mvgTied1.setPiT(0.1)
    mvgTied9 = TiedMVG()
    mvgTied9.setPiT(0.9)
    mvgTiedNaive = TiedNaiveBayesMVG()
    mvgTiedNaive1 = TiedNaiveBayesMVG()
    mvgTiedNaive1.setPiT(0.1)
    mvgTiedNaive9 = TiedNaiveBayesMVG()
    mvgTiedNaive9.setPiT(0.9)

    cv = CrossValidator()
    cv.setNumFolds(8)

    def forEachGenerativeModel(dataPrec, featureExtr):
        for classificator in [mvg, mvgNaive, mvgTied, mvgTied1, mvgTied9, mvgTiedNaive, mvgTiedNaive1, mvgTiedNaive9]:
            print("%-30s" % classificator.__str__(), end="\t")
            pipe.setStages([ZNorm(), dataPrec, featureExtr, classificator])
            cv.setEstimator(pipe)
            llr = cv.fit(DTR, LTR)
            for prio in effPriors:
                minDFC = DCF_min(llr, LTR, pi=prio)
                # pred = assign_label_bin(llr, p=prio)
                # acc = accuracy(pred, LTR)
                print("%.3f" % minDFC, end="\t\t")
            print()

    # print("MVG Classifiers ")
    # print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    # for dataPrec in [VoidStage(), Gaussianization()]:
    #     for featureExtr in [VoidStage(), PCA()]:
    #         if type(featureExtr) is PCA:
    #             for i in range(5, 8)[::-1]:
    #                 featureExtr.setDimension(i)
    #                 print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #                 forEachGenerativeModel(dataPrec, featureExtr)
    #         else:
    #             print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #             forEachGenerativeModel(dataPrec, featureExtr)

    # Discriminant probabilistic models
    lr = LogisticRegression()

    def plotDCFByLambda(preProc, minDCFs, directory):
        plt.figure()
        plt.title(directory+" "+preProc.__str__())
        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel("DCF")
        x = np.linspace(lambdas[0], lambdas[-1], len(lambdas))
        plt.plot(x, minDCFs[0], label="minDCF(piT= 0.5)")
        plt.plot(x, minDCFs[1], label="minDCF(piT= 0.1)")
        plt.plot(x, minDCFs[2], label="minDCF(piT= 0.9)")
        plt.xlim([lambdas[0], lambdas[-1]])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig("./plots/LogReg/"+directory+"/"+preProc.__str__()+".png", dpi=300)
        plt.show()

    # find best lambda linear
    # lr.setPiT(0.5)
    # minDCFs = np.zeros((3, 13))
    # for preProc in [VoidStage(), L2Norm(), Gaussianization()]:
    #     for i, lambd in enumerate(lambdas):
    #         lr.setLambda(lambd)
    #         pipe.setStages([ZNorm(), preProc, lr])
    #         cv.setEstimator(pipe)
    #         llr = cv.fit(DTR, LTR)
    #         for j, prio in enumerate(effPriors):
    #             print(lambd, " ", prio)
    #             minDCFs[j, i] = DCF_min(llr, LTR, pi=prio)
    #     plotDCFByLambda(preProc, minDCFs, "linear")

    # find best lambda quad
    # for preProc in [VoidStage(), L2Norm(), Gaussianization()]:
    #     for i, lambd in enumerate(lambdas):
    #         lr.setLambda(lambd)
    #         lr.setExpanded(True)
    #         pipe.setStages([ZNorm(), preProc, lr])
    #         cv.setEstimator(pipe)
    #         llr = cv.fit(DTR, LTR)
    #         for j, prio in enumerate(effPriors):
    #             print(lambd, " ", prio)
    #             minDCFs[j, i] = DCF_min(llr, LTR, pi=prio)
    #     plotDCFByLambda(preProc, minDCFs, "quad")

    lr.setLambda(10**-6)
    lr.setExpanded(True)

    def forEachLogRegModel(dataPrec, featureExtr):
        for i, classificator in enumerate([lr, lr, lr]):
            lr.setPiT(effPriors[i])
            print("%-30s" % classificator.__str__(), end="\t")
            pipe.setStages([ZNorm(), dataPrec, featureExtr, classificator])
            cv.setEstimator(pipe)
            llr = cv.fit(DTR, LTR)
            for prio in effPriors:
                minDFC = DCF_min(llr, LTR, pi=prio)
                # pred = assign_label_bin(llr, p=prio)
                # acc = accuracy(pred, LTR)
                print("%.3f" % minDFC, end="\t\t")
            print()

    # print("LogReg linear Classifiers ")
    # print("LogReg Quadratic Classifiers ")
    # print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    # for dataPrec in [VoidStage(), L2Norm(), Gaussianization()]:
    #     for featureExtr in [VoidStage(), PCA()]:
    #         if type(featureExtr) is PCA:
    #             for i in range(5, 8)[::-1]:
    #                 featureExtr.setDimension(i)
    #                 print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #                 forEachLogRegModel(dataPrec, featureExtr)
    #         else:
    #             print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #             forEachLogRegModel(dataPrec, featureExtr)

    # Discriminant non-probabilistic model
    svm = SVM()

    def plotDCFByCK(preProc, minDCFs, directory):
        plt.figure()
        plt.title(directory+" "+preProc.__str__())
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("DCF")
        x = np.linspace(C[0], C[-1], len(C))
        for j, k in enumerate(K):
            plt.plot(x, minDCFs[0, j, :], label=("minDCF(piT= 0.5) K=%.1f" % k))
            plt.plot(x, minDCFs[1, j, :], label=("minDCF(piT= 0.1) K=%.1f" % k))
            plt.plot(x, minDCFs[2, j, :], label=("minDCF(piT= 0.9) K=%.1f" % k))
        plt.xlim([C[0], C[-1]])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig("./plots/svm/"+directory+"/"+preProc.__str__()+".png", dpi=300)
        plt.show()

    def plotDCFByCGamma(preProc, minDCFs, directory):
        plt.figure()
        plt.title(directory+" "+preProc.__str__())
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("DCF")
        x = np.linspace(C[0], C[-1], len(C))
        for j, g in enumerate(G):
            plt.plot(x, minDCFs[0, j, :], label=("log(g)=%d" % np.log10(g)))
        plt.xlim([C[0], C[-1]])
        plt.ylim([0, 0.5])
        plt.legend()
        plt.savefig("./plots/svm/"+directory+"/"+preProc.__str__()+".png", dpi=300)
        plt.show()

    def plotDCFByC(preProc, minDCFs, directory):
        plt.figure()
        plt.title(directory + " " + preProc.__str__())
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("DCF")
        x = np.linspace(C[0], C[-1], len(C))
        plt.plot(x, minDCFs[0], label="minDCF(piT= 0.5)")
        plt.plot(x, minDCFs[1], label="minDCF(piT= 0.1)")
        plt.plot(x, minDCFs[2], label="minDCF(piT= 0.9)")
        plt.xlim([C[0], C[-1]])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig("./plots/svm/" + directory + "/" + preProc.__str__() + ".png", dpi=300)
        plt.show()

    # find best C
    # svm.setNoKern()
    # svm.setK(1)
    # print("finding best C for noKern")
    # minDCFs = np.zeros((3, len(C)))
    # for preProc in [VoidStage(), Gaussianization()]:
    #     for i, c in enumerate(C):
    #         svm.setC(c)
    #         pipe.setStages([ZNorm(), preProc, svm])
    #         cv.setEstimator(pipe)
    #         llr = cv.fit(DTR, LTR)
    #         for j, prio in enumerate(effPriors):
    #             print(c, " ", prio)
    #             minDCFs[j, i] = DCF_min(llr, LTR, pi=prio)
    #     plotDCFByC(preProc, minDCFs, "noKern")

    def forEachSVMModel(dataPrec, featureExtr):
        for i, classificator in enumerate([svm, svm, svm]):
            svm.setPiT(effPriors[i])
            print("%-30s" % classificator.__str__(), end="\t")
            pipe.setStages([ZNorm(), dataPrec, featureExtr, classificator])
            cv.setEstimator(pipe)
            llr = cv.fit(DTR, LTR)
            for prio in effPriors:
                minDFC = DCF_min(llr, LTR, pi=prio)
                print("%.3f" % minDFC, end="\t\t")
            print()

    # svm.setK(1)
    # svm.setC(10**-1)
    # svm.setNoKern()
    #
    # print("SVM Classifiers No Kernel ")
    # print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    # for dataPrec in [VoidStage(), Gaussianization()]:
    #     for featureExtr in [VoidStage(), PCA()]:
    #         if type(featureExtr) is PCA:
    #             continue
    #             # for i in range(5, 8)[::-1]:
    #             #     featureExtr.setDimension(i)
    #             #     print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #             #     forEachSVMModel(dataPrec, featureExtr)
    #         else:
    #             print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #             forEachSVMModel(dataPrec, featureExtr)

    # SVM Polynomial kernel find the best C
    #
    # print("finding best C for polynomial")
    # svm.setPiT(0.5)
    # svm.setK(1)
    # svm.setPolyKernel(1, 2)
    # minDCFs = np.zeros((3, len(C)))
    # for preProc in [VoidStage()]:
    #     for i, c in enumerate(C):
    #         svm.setC(c)
    #         pipe.setStages([ZNorm(), preProc, svm])
    #         cv.setEstimator(pipe)
    #         llr = cv.fit(DTR, LTR)
    #         np.save("./svmPoly"+preProc.__str__()+str(c), llr)
    #         for j, prio in enumerate(effPriors):
    #             print(c, " ", prio)
    #             minDCFs[j, i] = DCF_min(llr, LTR, pi=prio)
    #     plotDCFByC(preProc, minDCFs, "poly")
    #     np.save("./svmPoly"+preProc.__str__(), minDCFs)
    #
    # svm.setK(1)
    # svm.setC(1)
    # svm.setPolyKernel(1, 2)
    #
    # print("SVM Classifiers Poly Kernel ")
    # print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    # for dataPrec in [VoidStage()]:
    #     for featureExtr in [VoidStage()]:
    #         if type(featureExtr) is PCA:
    #             for i in range(5, 8)[::-1]:
    #                 featureExtr.setDimension(i)
    #                 print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #                 forEachSVMModel(dataPrec, featureExtr)
    #         else:
    #             print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #             forEachSVMModel(dataPrec, featureExtr)

    # svm.setK(1)
    # svm.setPiT(0.5)
    # print("find best C and gamma for RBF")
    # minDCFs = np.zeros((3, len(G), len(C)))
    # for preProc in [VoidStage()]:
    #     for i, c in enumerate(C):
    #         for n, g in enumerate(G):
    #             svm.setC(c)
    #             svm.setRBFKernel(g)
    #             pipe.setStages([ZNorm(), preProc, svm])
    #             cv.setEstimator(pipe)
    #             llr = cv.fit(DTR, LTR)
    #             for j, prio in enumerate(effPriors):
    #                 print(c, " ", g, " ", prio)
    #                 minDCFs[j, n, i] = DCF_min(llr, LTR, pi=prio)
    #     plotDCFByCGamma(preProc, minDCFs, "RBF")

    # svm.setC(10)
    # svm.setK(1)
    # svm.setRBFKernel(10**-2)

    # print("SVM Classifiers RBF Kernel ")
    # print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    # for dataPrec in [VoidStage()]:
    #     for featureExtr in [VoidStage()]:
    #         if type(featureExtr) is PCA:
    #             continue
    #             # for i in range(5, 8)[::-1]:
    #             #     featureExtr.setDimension(i)
    #             #     print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #             #     forEachSVMModel(dataPrec, featureExtr)
    #         else:
    #             print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
    #             forEachSVMModel(dataPrec, featureExtr)

    gmm = GMM()
    gmm.setDiagonal(False)
    gmm.setTied(False)

    # minDCFs = np.zeros((4, 2, 6))
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


    for i, type in enumerate(['Full_Cov', 'Diagonal_Cov', 'Tied_Cov', 'Tied_Diagonal_Cov']):
        printChars(type, minDCFs[i])

    # print("GMM Classifiers Raw")
    # for diag, tied, iterations, type in [(False, False, 4, "Full Cov"), (True, False, 3, "Diag Cov"),
    #                                      (False, True, 5, "Tied Cov"), (True, True, 5, "Tied Diag Cov")]:
    #     gmm.setDiagonal(diag)
    #     gmm.setTied(tied)
    #     gmm.setIterationLBG(iterations)
    #     pipe.setStages([ZNorm(), gmm])
    #     cv.setEstimator(pipe)
    #     s = cv.fit(DTR, LTR)
    #     print("%-30s" % type, end="\t")
    #     for prio in effPriors:
    #         minDFC = DCF_min(s, LTR, pi=prio)
    #         # pred = assign_label_bin(llr, p=prio)
    #         # acc = accuracy(pred, LTR)
    #         print("%.3f" % minDFC, end="\t\t")
    #     print()

    # print("GMM Classifiers Gaussianized")
    # for diag, tied, iterations, type in [(False, False, 2, "Full Cov"), (True, False, 5, "Diag Cov"),
    #                                      (False, True, 5, "Tied Cov"), (True, True, 4, "Tied Diag Cov")]:
    #     gmm.setDiagonal(diag)
    #     gmm.setTied(tied)
    #     gmm.setIterationLBG(iterations)
    #     pipe.setStages([ZNorm(), Gaussianization(), gmm])
    #     print(gmm.__str__())
    #     cv.setEstimator(pipe)
    #     s = cv.fit(DTR, LTR)
    #     print("%-30s" % type, end="\t")
    #     for prio in effPriors:
    #         minDFC = DCF_min(s, LTR, pi=prio)
    #         # pred = assign_label_bin(llr, p=prio)
    #         # acc = accuracy(pred, LTR)
    #         print("%.3f" % minDFC, end="\t\t")
    #     print()
