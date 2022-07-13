import numpy as np
import numpy
from Tools import load_dataset, DCF_min, DCF_norm_bin
from classifiers import LogisticRegression, SVM
from Pipeline import Pipeline, CrossValidator
from plots import print_DCFs, print_DETs, print_ROCs

from preProc import ZNorm
applications = [0.5, 0.1, 0.9]

if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE), labelDict = load_dataset()

    pipe = Pipeline()
    cv = CrossValidator()

    ## TRAINING QUADRATIC LOGISTIC REGRESSION
    lr = LogisticRegression()
    lr.setLambda(10**-6)
    lr.setExpanded(True)
    lr.setPiT(0.5)

    pipe.setStages([ZNorm(), lr])
    model = pipe.fit(DTR, LTR)  # Find the best model for the all Training set
    cv.setEstimator(pipe)
    score = cv.fit(DTR, LTR)  # Cross validation score on DTR, used for training the model Calibrator (Logistic Regression)

    # setup log reg for calibration
    lrCal = LogisticRegression()
    lrCal.setLambda(10**-6)
    lrCal.setPiT(0.5)

    pipe.setStages([lrCal])
    modelCalibrator = pipe.fit(score, LTR)  # train the modelCalibrator with scores of the DTR cross validation
    cv.setEstimator(pipe)
    scoreCalibratedQLR = cv.fit(score, LTR)  # same process as before, calibrate the scores through cross validation
    # and use them for the fusion

    ## INFERENCE QUADRATIC LOGISTIC REGRESSION
    # the two models have parameters only trained on DTR and LTR, so no bias is added
    slr = model.transform(DTE, None)
    sclr = modelCalibrator.transform(slr, None)  # calibrate scores

    np.save("./partial/Eval_quadLogReg-6_05", slr)
    np.save("./partial/EvalCal_quadLogReg-6_05", sclr)
    print()

    ## TRAINING SVM
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

    ## INFERENCE SVM
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
    # same procedure as before, scoreCalibrated come from CrossValidation on DTR, LTR
    modelFusion = pipe.fit(numpy.vstack((scoreCalibratedQLR, scoreCalibratedSVM)), LTR)

    ## INFERENCE
    # the model has parameters only trained on DTR and LTR, so no bias is added
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

    print_DETs([sclr, scsvm, sFusion], LTE, ["Quad Log Reg", "SVM RBF", "Fusion"], "Eval_fusion")
    print_ROCs([sclr, scsvm, sFusion], LTE, ["Quad Log Reg", "SVM RBF", "Fusion"], "Eval_fusion")