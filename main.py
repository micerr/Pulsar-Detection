import numpy

from Pipeline import Pipeline, CrossValidator
from featureExtraction import PCA, LDA
from classifiers import MVG, NaiveBayesMVG, TiedNaiveBayesMVG, TiedMVG
from plots import Histogram, Scatter


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


if __name__ == "__main__":
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L, 0)

    labels = ["Setosa", "Versicolor", "Virginica"]
    dimensions = ["sepal length", "sepal width", "petal lenght", "petal width"]

    for model in [MVG(), NaiveBayesMVG(), TiedMVG(), TiedNaiveBayesMVG()]:
        pipe = Pipeline()
        pipe.addStages([model])

        cv = CrossValidator()
        cv.setNumFolds(D.shape[1])
        cv.setEstimator(pipe)
        cv.fit(D, L)

