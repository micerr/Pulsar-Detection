import numpy
import matplotlib.pyplot as plt

# from Data.GMM_load import load_gmm
from Pipeline import Pipeline, CrossValidator
from Tools import mcol, vec, load_dataset, assign_label_bin, accuracy, DCF_norm_bin, DCF_min, logpdf_GMM, EM, mrow, \
    LBG_x2_Cluster, assign_label_multi
from preProc import PCA, LDA, Center, Whiten, L2Norm
from classifiers import MVG, NaiveBayesMVG, TiedNaiveBayesMVG, TiedMVG, LogisticRegression, SVM, GMM
from plots import Histogram, Scatter, print_DCFs, print_ROCs, print_pearson_correlation_matrices

def load_iris_binary():
    import sklearn.datasets
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L

def load_iris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

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
    """

    (DTR, LTR), (DTE, LTE), labelDict = load_dataset()
    print_pearson_correlation_matrices(DTR, LTR)
    pipe = Pipeline()
    pca = PCA()
    pca.setDimension(5)
    pipe.addStages([Histogram(), pca, Histogram()])
    pipe.fit(DTR, LTR, True)

    pipe = Pipeline()
    lda = LDA()
    lda.setDimension(1)
    pipe.addStages([pca, lda, Histogram()])
    pipe.fit(DTR, LTR, True)
    """

    D, L = load_iris()
    pipe = Pipeline()
    scatter = Scatter()
    scatter.setLabels(["Setosa", "Versicolor", "Virginica"])
    scatter.setDimensions(["Sepal lenght", "Sepal width", "Petal length", "Petal width"])
    histo = Histogram()
    histo.setSizeBin(15)
    histo.setLabels(["Setosa", "Versicolor", "Virginica"])
    histo.setDimensions(["Sepal lenght", "Sepal width", "Petal length", "Petal width"])
    white = Whiten()
    white.setWithinCov(True)
    pca = PCA()
    pca.setDimension(2)
    scatterPCA = Scatter()
    scatterPCA.setLabels(["Setosa", "Versicolor", "Virginica"])
    scatterPCA.setDimensions(["PC-1", "PC-2"])
    histoPCA = Histogram()
    histoPCA.setLabels(["Setosa", "Versicolor", "Virginica"])
    histoPCA.setDimensions(["PC-1", "PC-2"])
    pipe.setStages([scatter, histo, Center(), L2Norm(), scatter, histo])
    pipe.fit(D, L)

    pca = PCA()
    pca.setDimension(2)
    lda = LDA()
    lda.setDimension(2)
    scatter = Scatter()
    scatter.setLabels(["Setosa", "Versicolor", "Virginica"])
    scatter.setDimensions(["PC-1", "PC-2"])
    pipe = Pipeline()
    scatter.setTitle("PCA")
    pipe.setStages([pca, scatter])
    pipe.fit(D, L)
    scatter.setTitle("LDA")
    pipe.setStages([lda, scatter])
    pipe.fit(D, L)
    scatter.setTitle("PCA+LDA")
    # scatter.setSaveDirectoryDPI(".", "pca_lda", "png", 1000)
    pipe.setStages([pca, lda, scatter])
    pipe.fit(D, L)

    # cv = CrossValidator()
    # pipe = Pipeline()
    # cv.setNumFolds(D.shape[1])
    # for genModel in [MVG(), NaiveBayesMVG(), TiedMVG(), TiedNaiveBayesMVG()]:
    #     pipe.setStages([genModel])
    #     cv.setEstimator(pipe)
    #     postP = cv.fit(D, L)
    #     pred = assign_label_multi(postP)
    #     acc = accuracy(pred, L)
    #     print("Error:\t", (1 - acc) * 100, "%")

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    print("Logist Regression")
    for lambd in [10**-6, 10**-3, 10**-1, 10**0]:
        lr = LogisticRegression()
        pipe = Pipeline()
        lr.setLambda(lambd)
        pipe.addStages([lr])
        model = pipe.fit(DTR, LTR)
        s = model.transform(DTE, LTE)
        pred = assign_label_bin(s)
        acc = accuracy(pred, LTE)
        print("Error:\t", (1 - acc) * 100, "%")

    print("Logist Regression Expanded")
    for lambd in [10 ** -6, 10 ** -3, 10 ** -1, 10 ** 0]:
        lr = LogisticRegression()
        pipe = Pipeline()
        lr.setLambda(lambd)
        lr.setExpanded(True)
        pipe.addStages([lr])
        model = pipe.fit(DTR, LTR)
        s = model.transform(DTE, LTE)
        pred = assign_label_bin(s)
        acc = accuracy(pred, LTE)
        print("Error:\t", (1 - acc) * 100, "%")

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

    X = numpy.load("./Data/GMM_data_4D.npy")
    gmm = load_gmm("./Data/GMM_4D_3G_init.json")
    logDens = numpy.load("./Data/GMM_4D_3G_init_ll.npy")
    myRes = logpdf_GMM(X, gmm)
    print("Check result: ", numpy.sum(myRes-logDens))

    # X = numpy.load("./Data/GMM_data_1D.npy")
    # gmm = load_gmm("./Data/GMM_1D_3G_init.json")
    # logDens = numpy.load("./Data/GMM_1D_3G_init_ll.npy")
    # myRes = logpdf_GMM(X, gmm)
    # print("Check result: ", numpy.sum(myRes - logDens))

    bestDensity = load_gmm("./Data/GMM_4D_3G_EM.json")
    myBest = EM(X, gmm)
    avg1 = numpy.mean(logpdf_GMM(X, bestDensity))
    avg2 = numpy.mean(logpdf_GMM(X, myBest))
    print("Error avg log-likelihood: ", avg1, avg2, avg1-avg2)
    errorW = 0
    errorMU = 0
    errorC = 0
    for g in range(len(bestDensity)):
        errorW += bestDensity[g][0] - myBest[g][0]
        errorMU += numpy.sum(bestDensity[g][1] - myBest[g][1])
        errorC += numpy.sum(bestDensity[g][2] - myBest[g][2])
    print("Error weights: ", errorW, "\nError mean: ", errorMU, "\nError Cov: ", errorC)

    # plt.figure()
    # plt.hist(X.ravel(), bins=50, density=True)
    # XPlot = numpy.linspace(-8, 12, 1000)
    # plt.plot(XPlot.ravel(), numpy.exp(logpdf_GMM(mrow(XPlot), myBest)))
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.hist(X.ravel(), bins=50, density=True)
    # XPlot = numpy.linspace(-8, 12, 1000)
    # plt.plot(XPlot.ravel(), numpy.exp(logpdf_GMM(mrow(XPlot), bestDensity)))
    # plt.legend()
    # plt.show()

    X = numpy.load("./Data/GMM_data_4D.npy")
    mu = numpy.mean(X, axis=1)
    XC = X - mcol(mu)
    C = numpy.dot(XC, XC.T)/X.shape[1]
    GMM_1 = [(1.0, mu, C)]

    myBest = LBG_x2_Cluster(X, GMM_1, 0.1, 2)
    bestDensity = load_gmm("./Data/GMM_4D_4G_EM_LBG.json")
    avg1 = numpy.mean(logpdf_GMM(X, bestDensity))
    avg2 = numpy.mean(logpdf_GMM(X, myBest))
    print("Error avg log-likelihood: ", avg1, avg2, avg1 - avg2)
    errorW = 0
    errorMU = 0
    errorC = 0
    for g in range(len(bestDensity)):
        errorW += bestDensity[g][0] - myBest[g][0]
        errorMU += numpy.sum(bestDensity[g][1] - myBest[g][1])
        errorC += numpy.sum(bestDensity[g][2] - myBest[g][2])
    print("Error weights: ", errorW, "\nError mean: ", errorMU, "\nError Cov: ", errorC)

    # plt.figure()
    # plt.hist(X.ravel(), bins=50, density=True)
    # XPlot = numpy.linspace(-8, 12, 1000)
    # plt.plot(XPlot.ravel(), numpy.exp(logpdf_GMM(mrow(XPlot), myBest)))
    # plt.plot(XPlot.ravel(), numpy.exp(logpdf_GMM(mrow(XPlot), bestDensity)))
    # plt.show()
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    pipe = Pipeline()
    gmm = GMM()
    gmm.setPsi(0.01)
    gmm.setAlpha(0.1)
    for components in [1, 2, 4, 8, 16]:
        iterations = round(numpy.log2(components))
        gmm.setIterationLBG(iterations)
        for t in ["full", "diagonal", "tied"]:
            if t == "diagonal":
                gmm.setDiagonal(True)
                gmm.setTied(False)
            if t == "tied":
                gmm.setTied(True)
                gmm.setDiagonal(False)
            if t == "full":
                gmm.setTied(False)
                gmm.setDiagonal(False)
            pipe.setStages([gmm])
            model = pipe.fit(DTR, LTR, verbose=True)
            post = model.transform(DTE, LTE)
            pred = assign_label_multi(post)
            acc = accuracy(pred, LTE)
            print("Error:\t", (1 - acc) * 100, "%")

    """
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
    """
