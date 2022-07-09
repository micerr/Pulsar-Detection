import numpy as np

from Pipeline import Pipeline
from Tools import mcol, vec, load_dataset, assign_label_bin, accuracy, DCF_norm_bin, DCF_min, logpdf_GMM, EM, mrow, \
    LBG_x2_Cluster, assign_label_multi
from plots import Scatter, Histogram, print_pearson_correlation_matrices
from preProc import PCA, L2Norm, ZNorm, ZNorm_f
from classifiers import MVG

if __name__ == "__main__":
    (DTR, LTR), (DTE, LTE), labelDict = load_dataset()
    classLabel = {
        0: 'False',
        1: 'True'
    }

    """
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

    pipe.setStages([scatter, hist])
    pipe.fit(DTR, LTR)
    print_pearson_correlation_matrices(DTR, LTR, classLabel, "./plots/correlation")

    scatter.setSaveDirectoryDPI("./plots/scatter/Znorm", "", "png", 600).setTitle("Znorm")
    hist.setSaveDirectoryDPI("./plots/histogram/Znorm", "", "png", 600).setTitle("Znorm")
    pipe.setStages([ZNorm(), scatter, hist])
    pipe.fit(DTR, LTR)

    scatter.setSaveDirectoryDPI("./plots/scatter/L2norm", "", "png", 600).setTitle("L2norm")
    hist.setSaveDirectoryDPI("./plots/histogram/L2norm", "", "png", 600).setTitle("L2norm")
    pipe.setStages([L2Norm(), scatter, hist])
    pipe.fit(DTR, LTR)
    """
    
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
    
    pipe = Pipeline()
    mvg = MVG()
    pipe.setStages([mvg])
    model = pipe.fit(DTR, LTR)
    llr = model.transform(DTE, LTE)
    # Application (0.5, 1, 1)
    DCFmin = DCF_min(llr, LTE, pi=0.5, Cfn=1, Cfp=1)
    print("DCF_min = %f" % DCFmin)
    
    pipe = Pipeline()
    mvg = MVG()
    pipe.setStages([mvg])
    model = pipe.fit(DTR, LTR)
    llr = model.transform(DTE, LTE)
    # Application (0.1, 1, 1)
    DCFmin = DCF_min(llr, LTE, pi=0.1, Cfn=1, Cfp=1)
    print("DCF_min = %f" % DCFmin)
    
    pipe = Pipeline()
    mvg = MVG()
    pipe.setStages([mvg])
    model = pipe.fit(DTR, LTR)
    llr = model.transform(DTE, LTE)
    # Application (0.9, 1, 1)
    DCFmin = DCF_min(llr, LTE, pi=0.9, Cfn=1, Cfp=1)
    print("DCF_min = %f" % DCFmin)
    
    pipe = Pipeline()
    pca = PCA()
    pca.setDimension(7)
    mvg = MVG()
    pipe.setStages([pca, mvg])
    model = pipe.fit(DTR, LTR)
    llr = model.transform(DTE, LTE)
    # Application (0.5, 1, 1)
    DCFmin = DCF_min(llr, LTE, pi=0.5, Cfn=1, Cfp=1)
    print("DCF_min = %f" % DCFmin)
    
    pipe = Pipeline()
    pca = PCA()
    pca.setDimension(7)
    mvg = MVG()
    pipe.setStages([pca, mvg])
    model = pipe.fit(DTR, LTR)
    llr = model.transform(DTE, LTE)
    # Application (0.1, 1, 1)
    DCFmin = DCF_min(llr, LTE, pi=0.1, Cfn=1, Cfp=1)
    print("DCF_min = %f" % DCFmin)
    
    pipe = Pipeline()
    pca = PCA()
    pca.setDimension(7)
    mvg = MVG()
    pipe.setStages([pca, mvg])
    model = pipe.fit(DTR, LTR)
    llr = model.transform(DTE, LTE)
    # Application (0.9, 1, 1)
    DCFmin = DCF_min(llr, LTE, pi=0.9, Cfn=1, Cfp=1)
    print("DCF_min = %f" % DCFmin)
    