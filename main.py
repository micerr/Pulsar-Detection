import numpy

from Pipeline import Pipeline, CrossValidator
from Tools import mcol, vec
from featureExtraction import PCA, LDA
from classifiers import MVG, NaiveBayesMVG, TiedNaiveBayesMVG, TiedMVG, LogisticRegression
from plots import Histogram, Scatter, print_DCFs, print_ROCs


def load_iris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"]

def split_db_2to1(D, L, seed=0) :
    nTrain = int(D.shape[1]*2.0/3.0);
    # set the seed
    numpy.random.seed(seed);
    # create a vector (,1) of random number no repetitions
    idx = numpy.random.permutation(D.shape[1]);
    # divide the random numbers in 2 parts
    idxTrain = idx[0:nTrain];
    idxTest = idx[nTrain:];
    # get only the samples of that random number
    DTR = D[:,idxTrain];
    LTR = L[idxTrain];
    DTE = D[:,idxTest];
    LTE = L[idxTest];

    return (DTR,LTR), (DTE,LTE);

def load_iris_binary():
    import sklearn.datasets
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


if __name__ == "__main__":
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L, 0)

    labels = ["Setosa", "Versicolor", "Virginica"]
    dimensions = ["sepal length", "sepal width", "petal lenght", "petal width"]

    BD, BL = load_iris_binary()
    (BDTR, BLTR), (BDTE, BLTE) = split_db_2to1(BD, BL)

    lrLinear = LogisticRegression()
    lrLinear.setLambda(0.1)
    lrQuadratic = LogisticRegression()
    lrQuadratic.setExpanded(True)
    lrQuadratic.setLambda(0.1)
    models = [MVG(), NaiveBayesMVG(), TiedMVG(), TiedNaiveBayesMVG(), lrLinear, lrQuadratic]

    llrs = []

    for model in models:
        pipe = Pipeline()
        pipe.addStages([model])
        cv = CrossValidator()
        cv.setNumFolds(BD.shape[1])
        cv.setEstimator(pipe)
        print(model)
        cv.fit(BD, BL)
