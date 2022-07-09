import numpy as np

from Pipeline import Pipeline, VoidStage, CrossValidator
from Tools import load_dataset, DCF_min, assign_label_bin, accuracy
from classifiers import MVG, NaiveBayesMVG, TiedMVG, TiedNaiveBayesMVG
from plots import Scatter, Histogram, print_pearson_correlation_matrices
from preProc import PCA, L2Norm, ZNorm, ZNorm_f

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
        .setSizeBin(200)\
        .setSaveDirectoryDPI("./plots/histogram/raw", "", "png", 600)

    # pipe.setStages([scatter, hist])
    # pipe.fit(DTR, LTR)
    # print_pearson_correlation_matrices(DTR, LTR, classLabel, "./plots/correlation")

    # scatter.setSaveDirectoryDPI("./plots/scatter/Znorm", "", "png", 600).setTitle("Znorm")
    # hist.setSaveDirectoryDPI("./plots/histogram/Znorm", "", "png", 600).setTitle("Znorm")
    # pipe.setStages([ZNorm(), scatter, hist])
    # pipe.fit(DTR, LTR)

    # scatter.setSaveDirectoryDPI("./plots/scatter/L2norm", "", "png", 600).setTitle("L2norm")
    # hist.setSaveDirectoryDPI("./plots/histogram/L2norm", "", "png", 600).setTitle("L2norm")
    # pipe.setStages([L2Norm(), scatter, hist])
    # pipe.fit(DTR, LTR)

    effPriors = [0.5, 0.1, 0.9]

    # MVG
    mvg = MVG()
    mvgNaive = NaiveBayesMVG()
    mvgTied = TiedMVG()
    mvgTiedNaive = TiedNaiveBayesMVG()

    cv = CrossValidator()
    cv.setNumFolds(8)

    def forEachGenerativeModel(dataPrec, featureExtr):
        for classificator in [mvg, mvgNaive, mvgTied, mvgTiedNaive]:
            print("%-18s" % classificator.__str__(), end="\t")
            pipe.setStages([dataPrec, featureExtr, classificator])
            cv.setEstimator(pipe)
            llr = cv.fit(DTR, LTR)
            for prio in effPriors:
                minDFC = DCF_min(llr, LTR, pi=prio)
                # pred = assign_label_bin(llr, p=prio)
                # acc = accuracy(pred, LTR)
                print("%.3f" % minDFC, end="\t\t")
            print()

    print("MVG Classifiers")
    print("%-18s\tpi = 0.5\tpi = 0.1\tpi = 0.9" % "")
    for dataPrec in [VoidStage(), ZNorm(), L2Norm()]:
        for featureExtr in [VoidStage(), PCA()]:
            if type(featureExtr) is PCA:
                for i in range(2, 8)[::-1]:
                    featureExtr.setDimension(i)
                    print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
                    forEachGenerativeModel(dataPrec, featureExtr)
            else:
                print("%s -- %s" % (dataPrec.__str__(), featureExtr.__str__()))
                forEachGenerativeModel(dataPrec, featureExtr)


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
    #         .setSizeBin(200) \
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
    #         .setSizeBin(200) \
    #         .setSaveDirectoryDPI("./plots/histogram/l2_pca" + str(i), "", "png", 600)
    #     zNorm = ZNorm()
    #     l2 = L2Norm()
    #     pca.setDimension(i)
    #     pipe.setStages([zNorm, l2, pca, scatter, hist])
    #     pipe.fit(DTR, LTR)
