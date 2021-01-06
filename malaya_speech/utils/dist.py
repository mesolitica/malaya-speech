import numpy as np
import scipy.spatial.distance
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform


def l2_normalize(X):
    """
    L2 normalize vectors.

    Parameters
    ----------
    X : np.ndarray
        (n_samples, n_dimensions) vectors.
        
    Returns
    -------
    normalized : np.ndarray
        (n_samples, n_dimensions) L2-normalized vectors
    """

    norm = np.sqrt(np.sum(X ** 2, axis = 1))
    norm[norm == 0] = 1.0
    return (X.T / norm).T


def _pdist_func_1D(X, func):

    X = X.squeeze()
    n_items, = X.shape

    distances = []

    for i in range(n_items - 1):
        distance = func(X[i], X[i + 1 :])
        distances.append(distance)

    return np.hstack(distances)


def pdist(fX, metric = 'euclidean', **kwargs):
    """
    Same as scipy.spatial.distance with support for additional metrics
    * 'angular': pairwise angular distance
    * 'equal':   pairwise equality check (only for 1-dimensional fX)
    * 'minimum': pairwise minimum (only for 1-dimensional fX)
    * 'maximum': pairwise maximum (only for 1-dimensional fX)
    * 'average': pairwise average (only for 1-dimensional fX)
    """

    if metric == 'angular':
        cosine = scipy.spatial.distance.pdist(fX, metric = 'cosine', **kwargs)
        return np.arccos(np.clip(1.0 - cosine, -1.0, 1.0))

    elif metric == 'equal':
        return _pdist_func_1D(fX, lambda x, X: x == X)

    elif metric == 'minimum':
        return _pdist_func_1D(fX, np.minimum)

    elif metric == 'maximum':
        return _pdist_func_1D(fX, np.maximum)

    elif metric == 'average':
        return _pdist_func_1D(fX, lambda x, X: 0.5 * (x + X))

    else:
        return scipy.spatial.distance.pdist(fX, metric = metric, **kwargs)


def _cdist_func_1D(X_trn, X_tst, func):
    X_trn = X_trn.squeeze()
    X_tst = X_tst.squeeze()
    return np.vstack(func(x_trn, X_tst) for x_trn in iter(X_trn))


def cdist(fX_trn, fX_tst, metric = 'euclidean', **kwargs):
    """
    Same as scipy.spatial.distance.cdist with support for additional metrics
    * 'angular': pairwise angular distance
    * 'equal':   pairwise equality check (only for 1-dimensional fX)
    * 'minimum': pairwise minimum (only for 1-dimensional fX)
    * 'maximum': pairwise maximum (only for 1-dimensional fX)
    * 'average': pairwise average (only for 1-dimensional fX)
    """

    if metric == 'angular':
        cosine = scipy.spatial.distance.cdist(
            fX_trn, fX_tst, metric = 'cosine', **kwargs
        )
        return np.arccos(np.clip(1.0 - cosine, -1.0, 1.0))

    elif metric == 'equal':
        return _cdist_func_1D(
            fX_trn, fX_tst, lambda x_trn, X_tst: x_trn == X_tst
        )

    elif metric == 'minimum':
        return _cdist_func_1D(fX_trn, fX_tst, np.minimum)

    elif metric == 'maximum':
        return _cdist_func_1D(fX_trn, fX_tst, np.maximum)

    elif metric == 'average':
        return _cdist_func_1D(
            fX_trn, fX_tst, lambda x_trn, X_tst: 0.5 * (x_trn + X_tst)
        )

    else:
        return scipy.spatial.distance.cdist(
            fX_trn, fX_tst, metric = metric, **kwargs
        )


def compute_log_dist_matrix(X, metric = 'angular'):
    dist = pdist(X, metric = metric)
    dist_matrix = squareform((dist)) * (-1.0)
    return dist_matrix
