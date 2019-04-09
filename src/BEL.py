import numpy as np
import pandas as pd


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
        return ScaleData, ScaleObs, MeanVar, StdVar
    else:
        return ScaleData, MeanVar, StdVar



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

    eigenToKeep = 2

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

    xl, _, _, _ = lstsq(T11, L[:, :d],rcond=None)
    A = xl * np.sqrt(ndata - 1)
    xl, _, _, _ = lstsq(T22, M[:, :d],rcond=None)
    B = xl * np.sqrt(ndata - 1)

    r = np.min(np.max(np.transpose(np.diag(D[:,:d]))))

    A[perm1, :] = np.append(A, np.zeros((p1 - rankX, d)), axis=0)
    B[perm2, :] = np.append(B, np.zeros((p2 - rankY, d)), axis=0)

    U = np.dot(X, A)
    V = np.dot(Y, B)
    return(A, B, U, V)

def FPCA(DataDict, SmoothFac=10, k=3):
    """ """
    #import function specific modules
    from scipy.interpolate import UnivariateSpline

    # if 'time' is a pandas DatetimeIndex, convert to days as type(float)
    if isinstance(DataDict['time'], pd.DatetimeIndex):
        x = np.array((DataDict['time']-DataDict['time'][0]) / np.timedelta64(1, 'D'))
    else:
        x = DataDict['time']

    NumRealizations, NumObs, NumResponses = np.shape(DataDict['data'])

    coef = np.zeros_like(DataDict['data'])
    for ir in range(NumResponses):
        for ireal in range(NumRealizations):
            y = PriorFlows['data'][0, :, 0]
            spl = UnivariateSpline(x, y, s=SmoothFac, k=k)

            if ireal == 0:
                ncoef = len(spl.get_coeffs())
                coef = np.zeros((NumRealizations,ncoef,NumResponses))

            coef[ireal, :, ir] = spl.get_coeffs()

    "Not completed - should be redefined completely"


def PCanalysis(DataDict, eigentolerance=1., response=0, Obs=False):
    """

    :param DataDict: dictionary containing the data
    :param eigentolerance: determines the amount of variance to keep
    :param response: integer to determine response for dimension reduction
    :param Obs: boolean, if true perform dimension reduction on observation
    :return: returns the scores and the explained variance
    """
    from sklearn.decomposition import PCA as PCA

    NumRealizations, NumObs, NumResponses = np.shape(DataDict['data'])

    pca = PCA()
    pca_score = pca.fit_transform(DataDict['data'][:, :, response])

    explained = np.cumsum(pca.explained_variance_ratio_)
    if Obs:
        pca_scoreObs = np.dot(DataDict['dataObs'], pca.components_)

    eigenToKeep = 2

    if eigentolerance < 1.:
        ix = np.max([np.where(explained > eigentolerance)[0][0], eigenToKeep])
    else:
        ix = len(explained)

    if Obs:
        return pca_score[:, :ix], pca_scoreObs[:ix], pca.explained_variance_ratio_, pca.components_
    else:
        return pca_score[:, :ix], pca.explained_variance_ratio_, pca.components_


def NormalScoreTransform(Untransformed, plot = False):
    """

    :param Untransformed:
    :param plot:
    :return:
    """

    from statsmodels.distributions.empirical_distribution import ECDF
    from scipy.stats import norm
    from scipy import interpolate

    Transformed = np.zeros_like(Untransformed)

    for ii in range(np.shape(Transformed)[1]):
        h = Untransformed[:, ii]
        edf = ECDF(h)
        y = norm.ppf(edf.y[1:-1], np.mean(h), scale=np.std(h))
        x = edf.x[1:-1]
        f_e = interpolate.interp1d(x, y, fill_value='extrapolate')
        Transformed[:, ii] = f_e(h)

    return Transformed

def SampleCanonicalPosterior(mu_posterior, C_posterior, NumPosteriorSamples,
                             h_c, undotransform = True):
    """

    :param mu_posterior:
    :param C_posterior:
    :param NumPosteriorSamples:
    :param h_c:
    :return: h_c_post
    """
    from numpy.random import multivariate_normal as mvn
    from statsmodels.distributions.empirical_distribution import ECDF
    from scipy.stats import norm

    PosteriorSamples = mvn(mu_posterior, C_posterior, NumPosteriorSamples)

    if undotransform:
        PosteriorSamplesTransformed = np.zeros_like(PosteriorSamples)
        # back transform Normal Score Transformation:
        for ii in range(np.shape(h_c)[1]):
            OriginalScores = h_c[:, ii]
            TransformedSamples = np.copy(PosteriorSamples[:,ii])
            BackTransformedValue = np.zeros_like(TransformedSamples)

            edf = ECDF(OriginalScores)
            F,x = edf.y[1:], edf.x[1:]

            FStar = norm.cdf(TransformedSamples, np.mean(OriginalScores),
                         scale=np.std(OriginalScores))

            # for each FStar, find closest F
            for jj in range(len(FStar)):
                index = np.argmin(np.abs(F - FStar[jj]))

                if index==1:
                    BackTransformedValue[jj] = x[index]
                elif index==len(x):
                    BackTransformedValue[jj] = x[-1]
                elif F[index] < FStar[jj]:
                    BackTransformedValue[jj] = 0.5 * (x[index] + x[index - 1])
                elif index+1 == len(F):
                    BackTransformedValue[jj] = x[index]
                else:
                    BackTransformedValue[jj] = 0.5 * (x[index] + x[index + 1])

                if BackTransformedValue[jj]==float('inf') or BackTransformedValue[jj]==-float('inf'):
                    raise Exception('Bad interpolation')

            PosteriorSamplesTransformed[:, ii] = np.copy(BackTransformedValue)

        h_c_post = np.copy(PosteriorSamplesTransformed)

    else:
        h_c_post = np.copy(PosteriorSamples)

    return h_c_post

def UndoCanonicalPCA(h_c_post,B, h, V, mean, std):
    """

    :param h_c_post:
    :param B:
    :param h:
    :param V:
    :param mean:
    :param std:
    :return: h_reconstructed: (NReals X NOriginalDim) Reconstructed posterior realizations
    """
    from scipy.linalg import pinv as pinv
    from numpy.matlib import repmat as repmat

    NumPosteriorSamples = np.shape(h_c_post)[0]

    # Undo CCA
    HpostCoef = np.dot(h_c_post, pinv(B)) + repmat(np.mean(h, axis=0), NumPosteriorSamples, 1)

    # Undo PCA
    numPredCoeffs = np.shape(h)[1]
    h_scaled = np.dot(HpostCoef, V[:numPredCoeffs, :])

    #Undo standard scaling:
    h_reconstructed = (h_scaled*std)+mean

    return h_reconstructed

def EstimateQuantiles(post, prior = False, quantiles = [.1,.5,.9]):
    """

    :param post:
    :param prior:
    :return:
    """
    post_q = np.quantile(post, quantiles,axis = 0)

    if prior is not False:
        prior_q = np.quantile(prior, quantiles, axis = 0)

        return post_q, prior_q

    else:
        return post_q




















