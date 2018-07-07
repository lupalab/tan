import numpy as np


def ocd_unstable(cov=None, X=None, gamma_init=None):
    """Implements orthogonal correlated directions (ocd).
    Original Code, seems numerically unstanble."""
    if cov is None:
        assert X is not None
        M = X - np.mean(X, axis=0)
        cov = np.matmul(M.T, M)
    d = cov.shape[0]
    if gamma_init is None:
        gamma_init = np.random.normal(size=(d, 1))
    gamma_k = gamma_init/np.linalg.norm(gamma_init)
    Gamma = gamma_k.T
    for k in range(d-1):
        covg = np.matmul(cov, gamma_k)
        gdir = covg-np.matmul(Gamma.T, np.matmul(Gamma, covg))
        gamma_k = gdir/np.linalg.norm(gdir)
        Gamma = np.concatenate([Gamma, gamma_k.T], axis=0)
    return Gamma


def null(a, rtol=1e-5):
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return v[rank:].T.copy()


def ocd(cov=None, X=None, gamma_init=None):
    """Implements orthogonal correlated directions (OCD).
    Given previous orthogonal directions we find a new orthognal direction
    that maximizes the (sample) correlation w.r.t. the last given direction.
    Args:
        cov: d x d precomputed covariance matrix.
        X: N x d matrix of instances (if cov not given).
        gamma_init: starting direction, is random normal by default.
    Return:
        Gamma: d x d matrix where rows are ocd directions.
    """
    if cov is None:
        assert X is not None
        M = X - np.mean(X, axis=0)
        cov = np.matmul(M.T, M)
    d = cov.shape[0]
    if gamma_init is None:
        gamma_init = np.random.normal(size=(d, 1))
    gamma_k = gamma_init/np.linalg.norm(gamma_init)
    Gamma = gamma_k.T
    for k in range(d-1):
        U = null(Gamma)
        gdir = np.matmul(U, np.matmul(U.T, np.matmul(cov, gamma_k)))
        gamma_k = gdir/np.linalg.norm(gdir)
        Gamma = np.concatenate([Gamma, gamma_k.T], axis=0)
    return Gamma
