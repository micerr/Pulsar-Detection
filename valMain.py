import numpy
import numpy as np
import matplotlib.pyplot as plt

from Pipeline import Pipeline, VoidStage, CrossValidator
from classifiers import MVG, NaiveBayesMVG, TiedMVG, TiedNaiveBayesMVG, LogisticRegression, SVM
from Tools import mcol, vec, load_dataset, assign_label_bin, accuracy, DCF_norm_bin, DCF_min, logpdf_GMM, EM, mrow, \
    LBG_x2_Cluster, assign_label_multi
from plots import Scatter, Histogram, print_pearson_correlation_matrices, print_DETs, print_ROCs, print_DCFs
from preProc import PCA, L2Norm, ZNorm, Gaussianization

effPriors = [0.5, 0.1, 0.9]
applications = [0.5, 0.1, 0.9]
lambdas = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10]

if __name__ == "__main__":
    (DTR, LTR), _, labelDict = load_dataset()
    classLabel = {
        0: 'False',
        1: 'True'
    }

    pipe = Pipeline()
    cv = CrossValidator()
    cv.setNumFolds(8)

    # lr = LogisticRegression()
    # lr.setLambda(10**-6)
    # lr.setExpanded(True)
    # lr.setPiT(0.5)
    #
    # pipe.setStages([ZNorm(), lr])
    # cv.setEstimator(pipe)
    # slr = cv.fit(DTR, LTR)
    # np.save("./results/quadLogReg-6_05", slr)
    # for effPrio in effPriors:
    #     minDFC = DCF_min(slr, LTR, pi=effPrio)
    #     print("%.3f" % minDFC, end="\t\t")
    # print()

    # svm = SVM()
    # svm.setC(10)
    # svm.setRBFKernel(10**-2)
    # svm.setK(1)
    # svm.setPiT(0.5)
    #
    # pipe.setStages([ZNorm(), svm])
    # cv.setEstimator(pipe)
    # ssvm = cv.fit(DTR, LTR)
    # np.save("./results/svmRBFRaw10_0.5", ssvm)
    # for effPrio in effPriors:
    #     minDFC = DCF_min(ssvm, LTR, pi=effPrio)
    #     print("%.3f" % minDFC, end="\t\t")
    # print()

    slr = np.load("./partial/quadLogReg-6_05.npy")
    ssvm = np.load("./partial/svmRBFRaw10_0.5.npy")

    # print_DETs([slr, ssvm], LTR, ["Quad Log Reg", "SVM RBF"],"")
    # print_ROCs([slr, ssvm], LTR, ["Quad Log Reg", "SVM RBF"],"")

    # print("Uncalibrated")
    # for i, s in enumerate([slr, ssvm]):
    #     print("%s" % ("Quad Log Reg" if i == 0 else "SVM RBF Kernel"), end="\n")
    #     print("minDCF", end="\t")
    #     for app in applications:
    #         minDCF = DCF_min(s, LTR, pi=app)
    #         print("%.3f" % minDCF, end="\t")
    #     print()
    #     print("actDCF", end="\t")
    #     for app in applications:
    #         actDCF = DCF_norm_bin(s, LTR, pi=app)
    #         print("%.3f" % actDCF, end="\t")
    #     print()
    #
    # print_DCFs([slr, ssvm], LTR, ["Quad Log Reg", "SVM RBF"], "quadLogReg_SVM_RBF_NoCalibr", "Uncalibrated")

    lr = LogisticRegression()
    lr.setLambda(10**-6)
    pipe.setStages([lr])
    cv.setEstimator(pipe)

    # scores = [slr, ssvm]
    #
    # print("Calibration")
    # for i in range(2):
    #     for effPrio in effPriors:
    #         lr.setPiT(effPrio)
    #         s = cv.fit(scores[i], LTR)
    #         s -= numpy.log(effPrio/(1-effPrio))
    #         print("%s,Log Reg t=%.1f" % (("Quad Log Reg" if i == 0 else "SVM RBF Kernel"), effPrio), end="\t")
    #         for app in applications:
    #             actDCF = DCF_norm_bin(s, LTR, app)
    #             print("%.3f" % actDCF, end="\t")
    #         print()

    # for effPrio in effPriors:
    #     lr.setPiT(effPrio)
    #     sclr = cv.fit(slr, LTR)
    #     scsvm = cv.fit(ssvm, LTR)
    #     sclr -= numpy.log(effPrio/(1-effPrio))
    #     scsvm -= numpy.log(effPrio/(1-effPrio))
    #     print_DCFs([sclr, scsvm], LTR, ["Quad Log Reg", "SVM RBF"], "quadLogReg_SVM_RBF_Calibr_%.2f" % effPrio, "Calibration for target=%.1f" % effPrio)

    lr.setPiT(0.5)
    sclr = cv.fit(slr, LTR)
    scsvm = cv.fit(ssvm, LTR)
    fusion = numpy.vstack((sclr, scsvm))

    # find best lambda linear
    # minDCFs = np.zeros((3, 13))
    # for i, lambd in enumerate(lambdas):
    #     lr.setLambda(lambd)
    #     pipe.setStages([lr])
    #     cv.setEstimator(pipe)
    #     s = cv.fit(fusion, LTR)
    #     for j, prio in enumerate(effPriors):
    #         print(lambd, " ", prio)
    #         minDCFs[j, i] = DCF_min(s, LTR, pi=prio)
    # print(minDCFs)

    sFusion = cv.fit(fusion, LTR)

    print("After Fusion")
    for i, s in enumerate([sclr, scsvm, sFusion]):
        if i == 0:
            print("Quad Log Reg")
        elif i == 1:
            print("SVM RBF Kernel")
        else:
            print("Fusion")

        print("minDCF", end="\t")
        for app in applications:
            minDCF = DCF_min(s, LTR, pi=app)
            print("%.3f" % minDCF, end="\t")
        print()
        print("actDCF", end="\t")
        for app in applications:
            actDCF = DCF_norm_bin(s, LTR, pi=app)
            print("%.3f" % actDCF, end="\t")
        print()

    print_DCFs([sclr, scsvm, sFusion], LTR, ["Quad Log Reg", "SVM RBF", "Fusion"], "fusion_Cost_Function_OnlySVM", "Bayes error plot for the fused system")

    print_DETs([sclr, scsvm, sFusion], LTR, ["Quad Log Reg", "SVM RBF", "Fusion"], "fusionOnlySVM")
    print_ROCs([sclr, scsvm, sFusion], LTR, ["Quad Log Reg", "SVM RBF", "Fusion"], "fusionOnlySVM")

