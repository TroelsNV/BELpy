import numpy as np


def GetNumHarmonics(ExplainedVariance, MinEigenValues, EigenTolerance):
    """
    :param ExplainedVariance: the explained variance from PCA
    :param MinEigenValues: minimum number of eigenvalues to keep
    :param EigenTolerance: the percentage of variation we would like to keep
    :return: nHarm: number of components to keep
    """
    CumVar = np.cumsum(ExplainedVariance)
    return np.max([MinEigenValues,np.sum(CumVar <= EigenTolerance)])


def ScaleVariables(SimVar, ObsVar=False):
    """
    # performs standard scaling according to:
    # z = (x - u) / s
    :param SimVar: Simulated variables
    :param ObsVar: observed variables if False, return only scaled simulations
    :return: scaled simulations and observation
    """
    NumRealizations, NumObs, NumResponses  = np.shape(SimVar)

    ScaleData = np.zeros_like(SimVar)
    if ObsVar is not False:
        ScaleObs = np.zeros_like(ObsVar)

    for ir in range(NumResponses):
        MeanVar = np.mean(SimVar[:, :, ir], axis=0)
        StdVar = np.std(SimVar[:, :, ir], axis=0)

        for ireal in range(NumRealizations):
            ScaleData[ireal, :, ir] = (SimVar[ireal, :, ir] - MeanVar) / StdVar

        if ObsVar is not False:
            ScaleObs[:, ir] = (ObsVar[:, ir] - MeanVar) / StdVar

    if ObsVar is not False:
        return ScaleData, ScaleObs
    else:
        return ScaleData



def mixedPCA(PriorList, ObsList, eigentolerance = 1.):
    """
    Performs mixed principle component analysis according to Abdi et al. (2013)
    :param PriorList:
    :param ObsList:
    :param eigentolerance:
    :return:
    """
    from sklearn.decomposition import PCA as PCA

    numTypes = len(PriorList)
    if numTypes != len(ObsList):
        raise Exception('Number of observation types must be equal to prior')

    norm_scores_data = np.empty((np.shape(PriorList[0])[0], 0), dtype = PriorList[0].dtype)
    norm_scores_obs = np.empty((np.shape(ObsList[0])[0], 0), dtype=ObsList[0].dtype)

    for it in np.arange(numTypes):
        for idat in np.arange(np.shape(PriorList[it])[2]):
            pca = PCA()
            data_trans = pca.fit_transform(PriorList[it][:, :, idat])
            expVar = pca.explained_variance_

            norm_score_data = PriorList[it][:, :, idat]/np.sqrt(expVar[0])
            norm_scores_data = np.append(norm_scores_data, norm_score_data, axis=1)

            norm_score_obs = ObsList[it][:,idat]/np.sqrt(expVar[0])
            norm_scores_obs = np.append(norm_scores_obs, norm_score_obs)

    pcac = PCA()
    mpca_scores = pcac.fit_transform(norm_scores_data)
    explained = np.cumsum(pcac.explained_variance_ratio_)
    components_c = pcac.components_

    mpca_obs = np.dot(norm_scores_obs, components_c)

    eigenToKeep = 3

    if eigentolerance < 1.:
        ix = np.max([np.where(explained > eigentolerance)[0][0], eigenToKeep])
    else:
        ix = len(explained)

    return(mpca_scores[:,:ix], mpca_obs[:ix], pcac.explained_variance_ratio_)

def CCana(X,Y):
    """

    :param X:
    :param Y:
    :return:
    """
    # import local modules:
    from scipy.linalg import qr as qr
    from numpy.linalg import svd as svd
    from numpy.linalg import lstsq as lstsq


    ndata,p1 = np.shape(X)

    if np.shape(Y)[0] != ndata:
        raise Exception('Number of samples (rows) in X and Y must be equal')

    p2 = np.shape(Y)[1]

    if ndata <= 1:
        raise Exception('Not enough samples (rows). number of rows is {:d}'.format(ndata))

    # center variables:
    X = X - np.tile(np.mean(X,axis=0), (ndata, 1))
    Y = Y - np.tile(np.mean(Y, axis=0), (ndata, 1))

    Q1, T11, perm1 = qr(X,mode='economic',pivoting=True)
    rankX = np.sum(np.absolute(np.diag(T11)) > np.spacing(np.absolute(T11[0, 0])*np.max([ndata,p1])))

    if rankX == 0:
        raise Exception('BadData X')
    elif rankX < p1:
        print('X not full rank')
        Q1 = Q1[:, 1:rankX]
        T11 = T11[1:rankX, 1: rankX]

    Q2, T22, perm2 = qr(Y, mode='economic', pivoting=True)
    rankY = np.sum(np.absolute(np.diag(T22)) > np.spacing(np.absolute(T22[0, 0]) * np.max([ndata, p2])))

    if rankY == 0:
        raise Exception('BadData Y')
    elif rankY < p2:
        print('Y not full rank')
        Q2 = Q2[:, 1:rankY]
        T22 = T22[1:rankY, 1: rankY]

    d = np.min([rankX, rankY])
    L, D, M = svd(np.dot(np.transpose(Q1), Q2),full_matrices=True, compute_uv=True)
    D = np.diag(D)

    xl, _, _, _ = lstsq(T11, L[:, :d])
    A = xl * np.sqrt(ndata - 1)
    xl, _, _, _ = lstsq(T22, M[:, :d])
    B = xl * np.sqrt(ndata - 1)

    r = np.min(np.max(np.transpose(np.diag(D[:,:d]))))

    A[perm1, :] = np.append(A, np.zeros((p1 - rankX, d)), axis=0)
    B[perm2, :] = np.append(B, np.zeros((p2 - rankY, d)), axis=0)

    U = np.dot(X, A)
    V = np.dot(Y, B)
    return(A, B, U, V)

def ComputeHarmonicScores(DataDict, PlotLevel = 0):
    """

    :param DataDict:
    :param PlotLevel:
    :return: dataFPCA
    """
    pass
    #import function specific modules
    #from scipy import interpolate
    #
    #StartTime = np.copy(np.min(DataDict['time']))
    #EndTime = np.copy(np.max(DataDict['time']))
    #
    #norder = DataDict['spline'][0]
    #nknots = DataDict['spline'][1]
    #nbasis = nknots + norder - 2
    #
    #NumResponses = np.shape(DataDict['data'])[2]
    #
    #dataFPCA = {}
    #
    #for ir in np.arange(NumResponses):
    #    CurrentResponse = DataDict['data'][:,:, ir]
    #
    #    if 'dataTrue' in DataDict.keys():
    #        CurrentResponse = np.vstack((CurrentResponse, DataDict['dataTrue']))





