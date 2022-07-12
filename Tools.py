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
        S[g:g+1, :] = logpdf_GAU_ND(X, mu, C)  # log Gaussian (Cluster component) == density of MVG
        S[g, :] += numpy.log(w)  # log join density (add the log prior density)
    return scipy.special.logsumexp(S, axis=0)  # marginal density == density of GMM

def EM(X, gmm, psi=0, diag=False, tied=False):
    # gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]
    M = len(gmm)  # Num clusters
    N = X.shape[1]  # Num samples
    D = X.shape[0]  # Num features

    """
    The GMM log-likelihood is not bounded above for M >= 2. Indeed, we can have arbitrarily high log-
    likelihood by centering one component at one of the N samples, and letting the corresponding covariance
    matrix shrink towards zero. To avoid these kind of degenerate solutions, we can constrain the minimum
    values of the eigenvalues of the covariance matrices.
    """

    if psi > 0:
        regularizedGMM = []
        for g, (w, mu, C) in enumerate(gmm):
            if diag:
                C = C * numpy.eye(D)

            U, s, _ = numpy.linalg.svd(C)
            s[s < psi] = psi
            C = numpy.dot(U, mcol(s) * U.T)
            regularizedGMM.append((w, mu, C))
        gmm = regularizedGMM

    def stepE(gmm):
        S = numpy.zeros((M, N))
        for g, (w, mu, C) in enumerate(gmm):
            # TODO Implement logpdf_GAU_ND for multiple Clusters, it will improve the efficiency
            S[g:g+1, :] = logpdf_GAU_ND(X, mu, C)  # log Gaussian (Cluster component) == density of MVG
            S[g, :] += numpy.log(w)  # log join density (add the log prior density)
        marginalS = scipy.special.logsumexp(S, axis=0)  # marginal density == density of GMM
        return numpy.exp(S - marginalS)  # responsibilities (M, N)

    def stepM(r):
        Z = numpy.sum(r, axis=1)  # (M, 1) Zero Order Statistic
        F = numpy.dot(r, X.T)  # (M, D) First Order Statistic
        # FFor = numpy.zeros((M, D))
        # for g in range(M):
        #     FFor[g:g+1, :] = numpy.sum(r[g]*X, axis=1)
        # TODO why the compact form doesn't work??
        Sec = numpy.dot((numpy.tile(r, D).reshape(M * D, N) * numpy.repeat(X, M, axis=0)).reshape(M, D, N),
                        X.T)  # (M, D, D)
        SecFor = numpy.zeros((M, D, D))
        for g in range(M):
            SecFor[g:g+1, :, :] = numpy.dot(r[g, :] * X, X.T)

        MU = F / mcol(Z)  # (M, D) for each cluster there is a mean
        # MUCOV = numpy.zeros((M, D, D))
        # for g in range(M):
        #     MUCOV[g:g+1, :] = numpy.dot(mcol(MU[g, :]), mrow(MU[g, :]))
        # SIGFor = SecFor / Z.reshape(M, 1, 1) - MUCOV
        SIG = SecFor / Z.reshape(M, 1, 1) - (mrow(MU).T.reshape(M, D, 1) * MU.reshape(M, 1, D))  # (M, D, D)
        W = Z / N  # (M, 1) for each cluster there is a weight

        # Tied Computation, do just one time for all
        tiedC = numpy.zeros((D, D))
        if tied:
            tiedC = numpy.sum(Z.reshape(M, 1, 1) * SIG, axis=0) / N

        for g, C in enumerate(SIG):
            # Diagonal Covariance matrices, it doesn't mean Naive Bayes
            if diag:
                SIG[g] = SIG[g] * numpy.eye(D)

            # Tied Covariance
            if tied:
                SIG[g] = tiedC

            """
            The GMM log-likelihood is not bounded above for M >= 2. Indeed, we can have arbitrarily high log-
            likelihood by centering one component at one of the N samples, and letting the corresponding covariance
            matrix shrink towards zero. To avoid these kind of degenerate solutions, we can constrain the minimum
            values of the eigenvalues of the covariance matrices.
            """
            if psi > 0:
                U, s, _ = numpy.linalg.svd(C)
                s[s < psi] = psi
                SIG[g] = numpy.dot(U, mcol(s) * U.T)

        return [(W[g], MU[g], SIG[g]) for g in range(M)]

    def stopCriteria(gmm, newGMM, delta):
        old = numpy.mean(logpdf_GMM(X, gmm))
        new = numpy.mean(logpdf_GMM(X, newGMM))
        # print("Old: ", old, "New: ", new)
        if new < old:
            print("PANIC!!!!!!!")
        return new - old < delta

    while True:
        r = stepE(gmm)
        newGMM = stepM(r)
        if stopCriteria(gmm, newGMM, 10**-6):
            break
        gmm = newGMM

    return newGMM

def LBG(gmm, alpha):
    # gmm = [(w1, mu1, C1), (w2, mu2, C2), ...]
    newGMM = []
    for g, (w, mu, C) in enumerate(gmm):
        """
        we are displacing the new components along the direction of maximum variance, using a
        step that is proportional to the standard deviation of the component we are splitting.
        """
        U, s, Vh = numpy.linalg.svd(C)  # U := eigenvector ; s := eigenvalue
        d = U[:, 0:1] * s[0] ** 0.5 * alpha  # we take the eigenvector with the maximum variance
        newGMM.append((w/2, mcol(mu)+d, C))
        newGMM.append((w/2, mcol(mu)-d, C))
    return newGMM

def LBG_x2_Cluster(X, gmm, alpha, i, psi=0, diag=False, tied=False):
    """
    Usually the starting point is:
    w = 1
    mu = empirical mean of the dataset
    C = empirical covariance matrix of the dataset

    But they are Hyper parameters (The problem is not a convex one)
    Alpha is a H-parameter too
    """

    """
    If i == 0 there is just a single component, so the best solution si taking the mean of the Dataset, and the 
    Covariance matrix. EM and LBG algorithm are not needed
    """
    if i == 0:
        mu = numpy.mean(X, axis=1)
        XC = X - mcol(mu)
        C = numpy.dot(XC, XC.T) / X.shape[1]
        if diag:
            C = C * numpy.eye(X.shape[0])
        return [(1.0, mcol(mu), C)]

    for _ in range(i):
        gmm = LBG(gmm, alpha)  # G -> 2G
        gmm = EM(X, gmm, psi, diag, tied)  # Apply EM algorithm

    return gmm  # gmm * 2^i Clusters
    
def center_data(D):
    D = D - mcol(D.mean(1))
    return D

def cov_mat(D):
    DC = center_data(D)
    C = numpy.dot(DC, DC.T) / float(D.shape[1])
    return C

def within_cov_mat(D, L):
    K = L.max() + 1
    nSamples = D.shape[1]

    Sw = 0
    for i in range(K):
        DCl = D[:, L == i]  # take only samples of the class-i
        mu = mcol(DCl.mean(1))  # compute the mean of the class data
        DClC = DCl - mu  # center the class data
        Sw += numpy.dot(DClC, DClC.T)
    Sw = Sw / nSamples
    return Sw

def pearson_correlation_mat(D):
    C = cov_mat(D)
    sqrtV = mcol(numpy.diag(C)**(1/2))
    R = C * numpy.dot(sqrtV**-1, (sqrtV**-1).T)
    return numpy.abs(R)

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
        t = - numpy.log((p * Cfn) / ((1 - p) * Cfp))  # optimal threshold
    P = (llr > t) + 0  # + 0 converts booleans to integers
    return P

def assign_label_multi(post):
    return numpy.argmax(post, axis=0)

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

# Splitting the dataset in two parts: one part will be used for model training,
# the second one for evaluation.
# Single-Fold approach
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) # 100 samples for training, 50 for evaluation
    numpy.random.seed(seed)
    # Shuffling data, for picking up two splits
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)


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
  
