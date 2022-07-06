import numpy

from Pipeline import Pipeline, CrossValidator
from Tools import mcol, vec, load_dataset, assign_label_bin, accuracy, DCF_norm_bin, DCF_min
from featureExtraction import PCA, LDA
from classifiers import MVG, NaiveBayesMVG, TiedNaiveBayesMVG, TiedMVG, LogisticRegression, SVM
from plots import Histogram, Scatter, print_DCFs, print_ROCs

def load_iris_binary():
    import sklearn.datasets
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    # set the seed
    numpy.random.seed(seed)
    # create a vector (,1) of random number no repetitions
    idx = numpy.random.permutation(D.shape[1])
    # divide the random numbers in 2 parts
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    # get only the samples of that random number
    DTR = D[:, idxTrain]
    LTR = L[idxTrain]
    DTE = D[:, idxTest]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)

if __name__ == "__main__":

    # (DTR, LTR), (DTE, LTE), labelDict = load_dataset()

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    print("Normal Kernel")
    for K in [1, 10]:
        for C in [0.1, 1.0, 10.0]:
            svm = SVM()
            pipe = Pipeline()
            svm.setC(C)
            svm.setK(K)
            pipe.addStages([svm])
            model = pipe.fit(DTR, LTR)
            s = model.transform(DTE, LTE)
            pred = assign_label_bin(s)
            acc = accuracy(pred, LTE)
            print("Error:\t", (1 - acc) * 100, "%")

    print("Poly kernel")
    for c in [0, 1]:
        for K in [0, 1]:
            svm = SVM()
            pipe = Pipeline()
            svm.setC(1.0)
            svm.setK(K)
            svm.setPolyKernel(c, 2)
            pipe.addStages([svm])
            model = pipe.fit(DTR, LTR)
            s = model.transform(DTE, LTE)
            pred = assign_label_bin(s)
            acc = accuracy(pred, LTE)
            print("Error:\t", (1 - acc) * 100, "%")

    print("RBF Kernel")
    for K in [0, 1]:
        for g in [1, 10]:
            svm = SVM()
            pipe = Pipeline()
            svm.setC(1.0)
            svm.setK(K)
            svm.setRBFKernel(g)
            pipe.addStages([svm])
            model = pipe.fit(DTR, LTR)
            s = model.transform(DTE, LTE)
            pred = assign_label_bin(s)
            acc = accuracy(pred, LTE)
            print("Error:\t", (1 - acc) * 100, "%")

    lrLinear = LogisticRegression()
    lrLinear.setLambda(0.1)
    lrQuadratic = LogisticRegression()
    lrQuadratic.setExpanded(True)
    lrQuadratic.setLambda(0.1)
    models = [MVG(), NaiveBayesMVG(), TiedMVG(), TiedNaiveBayesMVG()]

    llrs = []

    for model in models:
        pipe = Pipeline()
        pipe.addStages([model])
        cv = CrossValidator()
        cv.setNumFolds(10)
        cv.setEstimator(pipe)
        print(model)
        cv.fit(DTR, LTR)

    pipe = Pipeline()
    pipe.addStages([TiedMVG()])
    model = pipe.fit(DTR, LTR)
    llr = model.transform(DTE, LTE)

    pred = assign_label_bin(llr)
    acc = accuracy(pred, LTE)
    print("Error:\t", (1 - acc) * 100, "%")
    bCost = DCF_norm_bin(llr, LTE)
    minCost = DCF_min(llr, LTE)
    print("DCF norm:\t", bCost, "\nDCF min:\t", minCost, "\n")

