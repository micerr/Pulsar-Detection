from Pipeline import Pipeline
from featureExtraction import PCA, LDA
from plots import Histogram, Scatter


def load_iris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"]


if __name__ == "__main__":
    D, L = load_iris()
    labels = ["Setosa", "Versicolor", "Virginica"]
    dimensions = ["sepal length", "sepal width", "petal lenght", "petal width"]

    histPre = Histogram()
    histPre.setLabels(labels)
    histPre.setDimensions(dimensions)

    scatterPre = Scatter()
    scatterPre.setLabels(labels)
    scatterPre.setDimensions(dimensions)

    pca = PCA()
    pca.setDimension(2)

    lda = LDA()
    lda.setDimension(2)

    scatterPost = Scatter()
    scatterPost.setLabels(labels)
    histPost = Histogram()

    pipe = Pipeline()
    pipe.addStages([pca, scatterPost, histPost])
    pipe.fit(D, L, verbose=True)

    del pipe
    pipe = Pipeline()
    pipe.addStages([lda, scatterPost, histPost])
    pipe.fit(D, L, verbose=True)

    del pipe
    pipe = Pipeline()
    pipe.addStages([pca, lda, scatterPost, histPost])
    pipe.fit(D, L, verbose=True)


