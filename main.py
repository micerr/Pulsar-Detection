import numpy as np

from Pipeline import Pipeline
from Tools import load_dataset
from plots import Scatter, Histogram, print_pearson_correlation_matrices
from preProc import PCA, L2Norm, Center, StandardizeVariance

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
        .setSaveDirectoryDPI("./plots/scatter/raw", "", "png", 600)
    hist = Histogram() \
        .setTitle("RAW") \
        .setDimensions(labelDict)\
        .setLabels(classLabel)\
        .setElemPerBin(200)\
        .setSaveDirectoryDPI("./plots/histogram/raw", "", "png", 600)
#
    # pipe.setStages([scatter, hist])
    # pipe.fit(DTR, LTR)
    # print_pearson_correlation_matrices(DTR, LTR, classLabel, "./plots/correlation")
#
    # # print scatters and histograms for PCA
    # pca = PCA()
    # for i in range(2, 8)[::-1]:
    #     labelDim = []
    #     for j in range(i):
    #         labelDim.append("PCA-"+str(j))
    #     scatter\
    #         .setTitle("PCA-"+str(i))\
    #         .setDimensions(labelDim) \
    #         .setLabels(classLabel) \
    #         .setSaveDirectoryDPI("./plots/scatter/pca"+str(i), "", "png", 600)
    #     hist\
    #         .setTitle("PCA-" + str(i)) \
    #         .setDimensions(labelDim) \
    #         .setLabels(classLabel) \
    #         .setElemPerBin(200) \
    #         .setSaveDirectoryDPI("./plots/histogram/pca"+str(i), "", "png", 600)
    #     pca.setDimension(i)
    #     pipe.setStages([pca, scatter, hist])
    #     pipe.fit(DTR, LTR)

    # # print scatters and histograms for L2 + PCA
    # pca = PCA()
    # for i in range(2, 8)[::-1]:
    #     labelDim = []
    #     for j in range(i):
    #         labelDim.append("PCA-" + str(j))
    #     scatter \
    #         .setTitle("PCA-" + str(i)) \
    #         .setDimensions(labelDim) \
    #         .setLabels(classLabel) \
    #         .setSaveDirectoryDPI("./plots/scatter/l2_pca" + str(i), "", "png", 600)
    #     hist \
    #         .setTitle("PCA-" + str(i)) \
    #         .setDimensions(labelDim) \
    #         .setLabels(classLabel) \
    #         .setElemPerBin(200) \
    #         .setSaveDirectoryDPI("./plots/histogram/l2_pca" + str(i), "", "png", 600)
    #     center = Center()
    #     std = StandardizeVariance()
    #     l2 = L2Norm()
    #     pca.setDimension(i)
    #     pipe.setStages([center, std, l2, pca, scatter, hist])
    #     pipe.fit(DTR, LTR)
