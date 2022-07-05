import numpy

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def vec(M):
    return M.reshape((-1, 1), order="F")

def logpdf_GAU_ND(X, mu, C):
    M = mu.shape[0]  # dimensions
    pi = numpy.pi
    Precision = numpy.linalg.inv(C)

    first = -M / 2 * numpy.log(2 * pi)
    second = -0.5 * numpy.linalg.slogdet(C)[1]
    XC = X - mcol(mu)  # center the values
    third_1 = numpy.dot(XC.T, Precision)
    third = -0.5 * (third_1.T * XC).sum(0)
    return first + second + third

def load_dataset():
    
    DTR = []
    LTR = []
    DTE = []
    LTE = []
    
    labelDict = {
        0: 'Mean of the integrated profile',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'Skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve',
        5: 'Standard deviation of the DM-SNR curve',
        6: 'Excess kurtosis of the DM-SNR curve',
        7: 'Skewness of the DM-SNR curve',
    }
    
    with open('./dataset/Train.txt', 'r') as trainData:
        for line in trainData:
            try:
                fields = line.split(',')[0:8]
                fields = mcol(numpy.array([float(i) for i in fields]))
                label = line.split(',')[-1].strip()
                DTR.append(fields)
                LTR.append(label)
            except:
                pass
        
        DTR = numpy.hstack(DTR)
        LTR = numpy.array(LTR, dtype=numpy.int32)
    
    with open('./dataset/Test.txt', 'r') as testData:
        for line in testData:
            try:
                fields = line.split(',')[0:8]
                fields = mcol(numpy.array([float(i) for i in fields]))
                label = line.split(',')[-1].strip()
                DTE.append(fields)
                LTE.append(label)
            except:
                pass
            
        DTE = numpy.hstack(DTE)
        LTE = numpy.array(LTE, dtype=numpy.int32)
    
    return (DTR, LTR), (DTE, LTE), labelDict
            