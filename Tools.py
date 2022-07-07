import numpy
import scipy.special


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

def logpdf_GMM(X, gmm):
    # gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]
    M = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((M, N))
    for g, (w, mu, C) in enumerate(gmm):
        S[g:g+1, :] = logpdf_GAU_ND(X, mu, C)  # log Gaussian
        S[g, :] += numpy.log(w)  # log join density
    return scipy.special.logsumexp(S, axis=0)  # marginal density == density of GMM

def confusion_matrix(P, L):
    # P => Predictions
    # L => Labels
    K = L.max()+1
    M = numpy.zeros((K, K))
    for i in numpy.arange(K):
        for j in numpy.arange(K):
            M[i, j] = ((P == i) * (L == j)).sum()
    return M

def accuracy(P, L):
    """
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label
    (in alphabetical order)
    """

    NCorrect = (P.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect) / float(NTotal)

def assign_label_bin(llr, p=0.5, Cfn=1, Cfp=1, t=None):
    if t is None:
        t = - numpy.log((p * Cfn) / ((1 - p) * Cfp))
    P = (llr > t) + 0  # + 0 converts booleans to integers
    return P

def DCF_u_bin(llr, L, pi=0.5, Cfn=1, Cfp=1, t=None):
    P = assign_label_bin(llr, pi, Cfn, Cfp, t)
    M = confusion_matrix(P, L)

    TN, FN, FP, TP = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    FNR = FN/(FN + TP)
    FPR = FP/(FP + TN)

    B = pi*Cfn*FNR + (1-pi)*Cfp*FPR
    return B

def DCF_norm_bin(llr, L, pi=0.5, Cfn=1, Cfp=1, t=None):
    B = DCF_u_bin(llr, L, pi, Cfn, Cfp, t)
    BDummy = min(pi*Cfn, (1 - pi)*Cfp)
    return B/BDummy

def DCF_min(llr, L, pi=0.5, Cfn=1, Cfp=1):
    # llr => log-likelihood ratio ; p => prior prob; Cfn => FNR; Cfp => FPR
    ts = numpy.array(llr).ravel()
    ts.sort()
    ts = numpy.concatenate((numpy.array([-numpy.inf]), ts, numpy.array([+numpy.inf])))

    B_min = numpy.inf
    for t in ts:
        B_norm = DCF_norm_bin(llr, L, pi, Cfn, Cfp, t)
        if B_norm < B_min:
            B_min = B_norm
    return B_min

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
            
