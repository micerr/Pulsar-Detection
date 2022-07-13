import numpy
import numpy as np
import matplotlib.pyplot as plt

from Pipeline import Pipeline, VoidStage, CrossValidator
from classifiers import MVG, NaiveBayesMVG, TiedMVG, TiedNaiveBayesMVG, LogisticRegression, SVM, GMM
from Tools import mcol, vec, load_dataset, assign_label_bin, accuracy, DCF_norm_bin, DCF_min, logpdf_GMM, EM, mrow, \
    LBG_x2_Cluster, assign_label_multi
from plots import Scatter, Histogram, print_pearson_correlation_matrices, print_DETs, print_ROCs, print_DCFs
from preProc import PCA, L2Norm, ZNorm, Gaussianization

applications = [0.5, 0.1, 0.9]
effPriors = [0.5, 0.1, 0.9]
lambdas = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6]
C = [10**-3, 10**-2, 10**-1, 10**0, 10**1]
K = [10**0, 10**1]
G = [10**-3, 10**-2, 10**-1, 10**0]

if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE), labelDict = load_dataset()
    classLabel = {
        0: 'False',
        1: 'True'
    }

    pipe = Pipeline()
    cv = CrossValidator()
    cv.setNumFolds(8)

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

    def forEachGenerativeModel(dataPrec, featureExtr):
        for classificator in [mvg, mvgNaive, mvgTied, mvgTied1, mvgTied9, mvgTiedNaive, mvgTiedNaive1, mvgTiedNaive9]:
            print("%-30s" % classificator.__str__(), end="\t")
            pipe.setStages([ZNorm(), dataPrec, featureExtr, classificator])
            models = pipe.fit(DTR, LTR)
            llr = models.transform(DTE, LTE)
            for app in applications:
                minDFC = DCF_min(llr, LTE, pi=app)
                print("%.3f" % minDFC, end="\t\t")
            print()

    print("MVG Classifiers ")
    print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    for dataPrec in [VoidStage(), Gaussianization()]:
        for featureExtr in [VoidStage(), PCA()]:
            if type(featureExtr) is PCA:
                for i in range(6, 8)[::-1]:
                    featureExtr.setDimension(i)
                    print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
                    forEachGenerativeModel(dataPrec, featureExtr)
            else:
                print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
                forEachGenerativeModel(dataPrec, featureExtr)

    lr = LogisticRegression()

    def forEachLogRegModel(dataPrec, featureExtr):
        for i, classificator in enumerate([lr, lr, lr]):
            lr.setPiT(applications[i])
            print("%-30s" % classificator.__str__(), end="\t")
            pipe.setStages([ZNorm(), dataPrec, featureExtr, classificator])
            models = pipe.fit(DTR, LTR)
            lpostr = models.transform(DTE, LTE)
            for app in applications:
                minDFC = DCF_min(lpostr, LTE, pi=app)
                print("%.3f" % minDFC, end="\t\t")
            print()

    def plotDCFByLambda(preProc, minDCFs, directory, name):
        plt.figure()
        plt.title(directory+" "+preProc.__str__())
        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel("DCF")
        x = np.linspace(lambdas[0], lambdas[-1], len(lambdas))
        for i, lin in enumerate(["-", "--"]):
            for j, pi in enumerate(["0.5", "0.1", "0.9"]):
                plt.plot(x, minDCFs[i][j], label="minDCF(piT= " + pi + ") " + ("Eval" if i == 0 else "Train"),
                         linestyle=lin)
        plt.xlim([lambdas[0], lambdas[-1]])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig("./plots/LogReg/"+directory+"/"+name+preProc.__str__()+".png", dpi=300)
        plt.show()

    def findBestLambda(lr, type):
        lr.setPiT(0.5)

        scoresEval = np.zeros((3, len(lambdas), DTE.shape[1]))
        for j, preProc in enumerate([VoidStage(), Gaussianization()]):
            for i, lambd in enumerate(lambdas):
                lr.setLambda(lambd)
                pipe.setStages([ZNorm(), preProc, lr])
                model = pipe.fit(DTR, LTR)
                scoresEval[j, i] = model.transform(DTE, None)

        scoresTrain = np.zeros((3, len(lambdas), DTR.shape[1]))
        for j, preProc in enumerate([VoidStage(), Gaussianization()]):
            for i, lambd in enumerate(lambdas):
                lr.setLambda(lambd)
                pipe.setStages([ZNorm(), preProc, lr])
                cv.setEstimator(pipe)
                scoresTrain[j, i] = cv.fit(DTR, LTR)

        minDCFEval = numpy.zeros((3, len(lambdas)))
        minDCFTrain = numpy.zeros((3, len(lambdas)))
        for i, preProc in enumerate([VoidStage(), Gaussianization()]):
            for j in range(len(lambdas)):
                for k, app in enumerate(applications):
                    minDCFEval[k, j] = DCF_min(scoresEval[i, j], LTE, pi=app)
                    minDCFTrain[k, j] = DCF_min(scoresTrain[i, j], LTR, pi=app)
            plotDCFByLambda(preProc, [minDCFEval, minDCFTrain], type, "EvalTrainC_" + preProc.__str__())

    lr.setLambda(10**-6)
    lr.setExpanded(False)

    findBestLambda(lr, "linear")

    print("LogReg linear Classifiers ")
    print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    for dataPrec in [VoidStage(), Gaussianization()]:
        print("%s -- %s" % (dataPrec.__str__(), VoidStage().__str__()))
        forEachLogRegModel(dataPrec, VoidStage())

    lr.setExpanded(True)

    findBestLambda(lr, "quad")

    print("LogReg Quadratic Classifiers ")
    print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    for dataPrec in [VoidStage(), Gaussianization()]:
        print("%s -- %s" % (dataPrec.__str__(), VoidStage().__str__()))
        forEachLogRegModel(dataPrec, VoidStage())

    def forEachSVMModel(dataPrec, featureExtr):
        for i, classificator in enumerate([svm, svm, svm]):
            svm.setPiT(applications[i])
            print("%-30s" % classificator.__str__(), end="\t")
            pipe.setStages([ZNorm(), dataPrec, featureExtr, classificator])
            models = pipe.fit(DTR, LTR)
            s = models.transform(DTE, LTE)
            for app in applications:
                minDFC = DCF_min(s, LTE, pi=app)
                print("%.3f" % minDFC, end="\t\t")
            print()

    def plotDCFByCEvalTrain(preProc, minDCFs, directory, filename):
        plt.figure()
        plt.title(directory + " " + preProc.__str__())
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("DCF")
        x = np.linspace(C[0], C[-1], len(C))
        for i, lin in enumerate(["-", "--"]):
            for j, pi in enumerate(["0.5", "0.1", "0.9"]):
                plt.plot(x, minDCFs[i][j], label="minDCF(piT= "+pi+") "+("Eval" if i == 0 else "Train"), linestyle=lin)
        plt.xlim([C[0], C[-1]])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig("./plots/svm/" + directory + "/" + filename + ".png", dpi=300)
        plt.show()

    def plotDCFByCGamma(preProc, minDCFs, directory):
        plt.figure()
        plt.title(directory+" "+preProc.__str__())
        plt.xscale("log")
        plt.xlabel("C")
        plt.ylabel("DCF")
        x = np.linspace(C[0], C[-1], len(C))
        for i, lin in enumerate(["-", "--"]):
            for j, g in enumerate(G):
                plt.plot(x, minDCFs[i, j, :], label=("log(g)=%d" % np.log10(g)))
        plt.xlim([C[0], C[-1]])
        plt.ylim([0, 0.5])
        plt.legend()
        plt.savefig("./plots/svm/"+directory+"/Eval"+preProc.__str__()+".png", dpi=300)
        plt.show()

    def findBestC(svm, type):
        print("finding best C for "+type+" Eval")
        scoresEval = np.zeros((2, len(C), LTE.size))
        for j, preProc in enumerate([VoidStage(), Gaussianization()]):
            for i, c in enumerate(C):
                svm.setC(c)
                pipe.setStages([ZNorm(), preProc, svm])
                model = pipe.fit(DTR, LTR)
                s = model.transform(DTE, LTE)
                scoresEval[j, i] = s
        np.save("./partial/SVM"+type+"BestEval", scoresEval)

        print("finding best C for "+type+" Train")
        scoresTrain = np.zeros((2, len(C), LTR.size))
        for j, preProc in enumerate([VoidStage(), Gaussianization()]):
            for i, c in enumerate(C):
                svm.setC(c)
                pipe.setStages([ZNorm(), preProc, svm])
                cv.setEstimator(pipe)
                s = cv.fit(DTR, LTR)
                scoresTrain[j, i] = s
        np.save("./partial/SVM"+type+"BestTrain", scoresTrain)

        scoresEval = np.load("./partial/SVM"+type+"BestEval.npy")
        scoresTrain = np.load("./partial/SVM"+type+"BestTrain.npy")

        minDCFEval = numpy.zeros((3, len(C)))
        minDCFTrain = numpy.zeros((3, len(C)))
        for i, preProc in enumerate([VoidStage(), Gaussianization()]):
            for j in range(len(C)):
                for k, app in enumerate(applications):
                    minDCFEval[k, j] = DCF_min(scoresEval[i, j], LTE, pi=app)
                    minDCFTrain[k, j] = DCF_min(scoresTrain[i, j], LTR, pi=app)
            plotDCFByCEvalTrain(preProc, [minDCFEval, minDCFTrain], type, "EvalTrainC_"+preProc.__str__())

    def findBestGamma(svm, type):
        print("finding best Gamma for " + type + " Eval")
        scoresEval = np.zeros((len(C), len(G), LTE.size))
        for i, c in enumerate(C):
            for n, g in enumerate(G):
                svm.setC(c)
                svm.setRBFKernel(g)
                pipe.setStages([ZNorm(), svm])
                model = pipe.fit(DTR, LTR)
                llr = model.transform(DTE, None)
                scoresEval[i, n] = llr
        np.save("./partial/SVM" + type + "BestEval", scoresEval)

        print("finding best Gamma for " + type + " Train")
        scoresTrain = np.zeros((len(C), len(G), LTR.size))
        for i, c in enumerate(C):
            for n, g in enumerate(G):
                svm.setC(c)
                svm.setRBFKernel(g)
                pipe.setStages([ZNorm(), svm])
                llr = cv.fit(DTR, LTR)
                scoresTrain[i, n] = llr
        np.save("./partial/SVM" + type + "BestTrain", scoresTrain)

        scoresEval = np.load("./partial/SVM" + type + "BestEval.npy")
        scoresTrain = np.load("./partial/SVM" + type + "BestTrain.npy")

        minDCFEval = numpy.zeros((len(G), len(C)))
        minDCFTrain = numpy.zeros((len(G), len(C)))
        for i in range(len(C)):
            for j in range(len(G)):
                minDCFEval[j, i] = DCF_min(scoresEval[i, j], LTE, pi=0.5)
                minDCFTrain[j, i] = DCF_min(scoresTrain[i, j], LTR, pi=app)
        plotDCFByCGamma(VoidStage(), [minDCFEval, minDCFTrain], type)

    svm = SVM()
    svm.setNoKern()
    svm.setK(1)

    findBestC(svm, "noKern")

    svm.setK(1)
    svm.setNoKern()

    print("SVM Classifiers No Kernel ")
    print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    for dataPrec in [VoidStage(), Gaussianization()]:
        if type(dataPrec) is VoidStage():
            svm.setC(1)
        else:
            svm.setC(10**-1)
        print("%s -- %s" % (dataPrec.__str__(), VoidStage().__str__()))
        forEachSVMModel(dataPrec, VoidStage())

    svm.setK(1)
    svm.setPolyKernel(1, 2)

    findBestC(svm, "poly")

    svm.setK(1)
    svm.setC(1)
    svm.setPolyKernel(1, 2)

    print("SVM Classifiers Poly Kernel ")
    print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    print("%s -- %s" % (VoidStage().__str__(), VoidStage().__str__()))
    forEachSVMModel(VoidStage(), VoidStage())

    svm.setC(10)
    svm.setK(1)
    svm.setRBFKernel(10**-2)

    print("SVM RBF")
    findBestGamma(svm, "RBF")

    print("SVM Classifiers RBF Kernel ")
    print("%-30s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    print("%s -- %s" % (VoidStage().__str__(), VoidStage().__str__()))
    forEachSVMModel(VoidStage(), VoidStage())

    gmm = GMM()
    gmm.setDiagonal(False)
    gmm.setTied(False)

    def printChars(type, minDCFs, minDCFsTrain):
        labels = ['1', '2', '4', '8', '16', '32']
        rawDCF = minDCFs[0].round(decimals=3)
        gaussianDCF = minDCFs[1].round(decimals=3)
        rawDCFTrain = minDCFsTrain[0].round(decimals=3)
        gaussianDCFTrain = minDCFsTrain[1].round(decimals=3)

        x = np.arange(len(labels))  # the label locations
        width = 0.20  # the width of the bars

        fig, ax = plt.subplots()
        plt.ylim([0, 0.25])
        rects1 = ax.bar(x - width / 2 - width, rawDCFTrain, width, label='minDCF(piT=0.5) Raw Train', alpha=0.5)
        rects2 = ax.bar(x - width / 2, rawDCF, width, label='minDCF(piT=0.5) Raw Eval')
        rects3 = ax.bar(x + width / 2, gaussianDCFTrain, width, label='minDCF(piT=0.5) Gaussianization Train', alpha=0.5)
        rects4 = ax.bar(x + width / 2 + width, gaussianDCF, width, label='minDCF(piT=0.5) Gaussianization Eval')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('DCF')
        ax.set_title(type)
        ax.set_xticks(x, labels)
        ax.legend()

        plt.savefig("./plots/gmm/Eval" + type + ".png", dpi=300)
        plt.show()

    # Find best number of components
    minDCFs = np.zeros((4, 2, 6))
    for k in range(4):
        for i in range(0, 6):
            for j, dataProc in enumerate([VoidStage(), Gaussianization()]):
                if k == 1:
                    gmm.setDiagonal(True)
                    gmm.setTied(False)
                    print("Diagonal ", end="")
                elif k == 2:
                    gmm.setTied(True)
                    gmm.setDiagonal(False)
                    print("Tied ", end="")
                elif k == 3:
                    gmm.setTied(True)
                    gmm.setDiagonal(True)
                    print("Tied-Diagonal ", end="")
                else:
                    print("Full Covariance ", end="")
                gmm.setIterationLBG(i)
                pipe.setStages([ZNorm(), dataProc, gmm])
                model = pipe.fit(DTR, LTR)
                llr = model.transform(DTE, LTE)
                minDCFs[k, j, i] = DCF_min(llr, LTE, 0.5)
                print("%s %d" % (("Raw" if j == 0 else "Gaussian"), i))
    np.save("./partial/GMMTuningEval.npy", minDCFs)

    minDCFsTrain = numpy.load("../partial/GMMTuning.npy")[:, :, 0:6]
    minDCFs = numpy.load("../partial/GMMTuningEval.npy")

    print("GMM chars")
    for i, type in enumerate(['Full_Cov', 'Diagonal_Cov', 'Tied_Cov', 'Tied_Diagonal_Cov']):
        printChars(type, minDCFs[i], minDCFsTrain[i])

    print("GMM Classifiers Raw")
    for diag, tied, iterations, type in [(False, False, 4, "Full Cov"), (True, False, 3, "Diag Cov"),
                                         (False, True, 5, "Tied Cov"), (True, True, 5, "Tied Diag Cov")]:
        gmm.setDiagonal(diag)
        gmm.setTied(tied)
        gmm.setIterationLBG(iterations)
        pipe.setStages([ZNorm(), gmm])
        model = pipe.fit(DTR, LTR)
        s = model.transform(DTE, LTE)
        print("%-30s" % type, end="\t")
        for app in applications:
            minDFC = DCF_min(s, LTE, pi=app)
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
        model = pipe.fit(DTR, LTR)
        s = model.transform(DTE, LTE)
        print("%-30s" % type, end="\t")
        for app in applications:
            minDFC = DCF_min(s, LTE, pi=app)
            print("%.3f" % minDFC, end="\t\t")
        print()

    ##################################################################################################################
    ## Evaluation chosen Models
    ## TRAINING
    lr = LogisticRegression()
    lr.setLambda(10**-6)
    lr.setExpanded(True)
    lr.setPiT(0.5)

    pipe.setStages([ZNorm(), lr])
    model = pipe.fit(DTR, LTR)
    cv.setEstimator(pipe)
    score = cv.fit(DTR, LTR)

    lrCal = LogisticRegression()
    lrCal.setLambda(10**-6)
    lrCal.setPiT(0.5)

    pipe.setStages([lrCal])
    modelCalibrator = pipe.fit(score, LTR)
    cv.setEstimator(pipe)
    scoreCalibratedQLR = cv.fit(score, LTR)

    ## INFERENCE
    slr = model.transform(DTE, None)
    sclr = modelCalibrator.transform(slr, None)

    np.save("./partial/Eval_quadLogReg-6_05", slr)
    np.save("./partial/EvalCal_quadLogReg-6_05", sclr)
    print()

    ## TRAINING
    svm = SVM()
    svm.setC(10)
    svm.setRBFKernel(10**-2)
    svm.setK(1)
    svm.setPiT(0.5)

    pipe.setStages([ZNorm(), svm])
    model = pipe.fit(DTR, LTR)
    cv.setEstimator(pipe)
    score = cv.fit(DTR, LTR)

    lrCal = LogisticRegression()
    lrCal.setLambda(10 ** -6)
    lrCal.setPiT(0.5)

    pipe.setStages([lrCal])
    modelCalibrator = pipe.fit(score, LTR)
    cv.setEstimator(pipe)
    scoreCalibratedSVM = cv.fit(score, LTR)

    ## INFERENCE
    ssvm = model.transform(DTE, None)
    scsvm = modelCalibrator.transform(ssvm, None)

    np.save("./partial/Eval_svmRBFRaw10_0.5", ssvm)
    np.save("./partial/EvalCal_svmRBFRaw10_0.5", scsvm)
    print()

    ## TRAINING FUSION
    lrFusion = LogisticRegression()
    lrFusion.setPiT(0.5)
    lrFusion.setLambda(10**-6)
    pipe.setStages([lrFusion])
    modelFusion = pipe.fit(numpy.vstack((scoreCalibratedQLR, scoreCalibratedSVM)), LTR)

    ## INFERENCE
    sFusion = modelFusion.transform(numpy.vstack((sclr, scsvm)), None)
    np.save("./partial/Eval_fusion", sFusion)

    sFusion = np.load("../partial/Eval_fusion.npy")
    slr = np.load("../partial/Eval_quadLogReg-6_05.npy")
    sclr = np.load("../partial/EvalCal_quadLogReg-6_05.npy")
    ssvm = np.load("../partial/Eval_svmRBFRaw10_0.5.npy")
    scsvm = np.load("../partial/EvalCal_svmRBFRaw10_0.5.npy")

    print("Uncalibrated")
    for i, s in enumerate([slr, ssvm]):
        print("%s" % ("Quad Log Reg" if i == 0 else "SVM RBF Kernel"), end="\n")
        print("minDCF", end="\t")
        for app in applications:
            minDCF = DCF_min(s, LTE, pi=app)
            print("%.3f" % minDCF, end="\t")
        print()
        print("actDCF", end="\t")
        for app in applications:
            actDCF = DCF_norm_bin(s, LTE, pi=app)
            print("%.3f" % actDCF, end="\t")
        print()

    print_DCFs([slr, ssvm], LTE, ["Quad Log Reg", "SVM RBF"], "Eval_quadLogReg_SVM_RBF_NoCalibr", "Uncalibrated")

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
            minDCF = DCF_min(s.ravel(), LTE, pi=app)
            print("%.3f" % minDCF, end="\t")
        print()
        print("actDCF", end="\t")
        for app in applications:
            actDCF = DCF_norm_bin(s.ravel(), LTE, pi=app)
            print("%.3f" % actDCF, end="\t")
        print()

    print_DCFs([sclr, scsvm, sFusion], LTE, ["Quad Log Reg", "SVM RBF", "Fusion"], "Eval_fusion_Cost_Function", "Bayes error plot for the fused system")

    print_DETs([sclr, scsvm, sFusion], LTE, ["Quad Log Reg", "SVM RBF", "Fusion"], "Eval_fusionOnlySVM")
    print_ROCs([sclr, scsvm, sFusion], LTE, ["Quad Log Reg", "SVM RBF", "Fusion"], "Eval_fusionOnlySVM")
