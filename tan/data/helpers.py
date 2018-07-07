import tensorflow as tf  # noqa
import numpy as np
import os as os
import pickle as pickle
import scipy.linalg as linalg
from ..utils import ocd # noqa


# TODO: Make loader and saver functions for pickle/npz versions


def add_noise_pickle(data_path, noise_scale=0.01, standardize=False):
    dataset = pickle.load(open(data_path, 'rb'))
    if standardize:
        meanvec = np.mean(dataset['train'], axis=0)
        stdvec = np.std(dataset['train'], axis=0)
        dataset = {d: (dataset[d]-meanvec)/stdvec for d in dataset}
    dataset_noise = {
        'train': dataset['train']+np.random.normal(
            size=dataset['train'].shape, scale=noise_scale
        ),
        'valid': dataset['valid']+np.random.normal(
            size=dataset['valid'].shape, scale=noise_scale
        ),
        'test': dataset['test']
    }
    save_dir, base = os.path.split(data_path)
    save_path = os.path.join(save_dir, os.path.splitext(base)[0]+'_noise.p')
    pickle.dump(dataset_noise, open(save_path, 'wb'))


def make_uci_npz(data_path, save_dir):
    """Helper function to save npz of splits in pickled UCI files."""
    dataset = pickle.load(open(data_path, 'rb'))
    base = os.path.splitext(os.path.basename(data_path))[0]+'.npz'
    save_path = os.path.join(save_dir, base)
    np.savez(save_path, **dataset)


def make_uci_data_dict(data_path, save_dir, unique_thresh=128):
    """Helper function to load, standardize, and pickle UCI csv files."""
    all_data = np.loadtxt(data_path, delimiter=",", skiprows=0,
                          dtype=np.float32)
    if len(all_data.shape) == 1:
        # Skip if only has one feature.
        return
    uvals = np.array(
        [len(np.unique(all_data[:, i])) for i in range(all_data.shape[1])]
    )
    real_dims = np.greater(uvals, unique_thresh)
    if np.sum(real_dims) <= 1:
        # Skip if only has one feature.
        return
    all_data = all_data[:, real_dims]
    N = len(all_data)
    rprm = np.random.permutation(N)
    perm_data = all_data[rprm]
    N_train = int(0.8*N)
    N_hold = int(0.1*N)
    dataset_unnorm = {
        'train': perm_data[:N_train, :],
        'valid': perm_data[N_train:N_train+N_hold, :],
        'test': perm_data[N_train+N_hold:, :],
    }
    mean_vec = np.mean(dataset_unnorm['train'], 0)
    std_vec = np.std(dataset_unnorm['train'], 0)
    dataset = {
        'train': (perm_data[:N_train, :]-mean_vec)/std_vec,
        'valid': (perm_data[N_train:N_train+N_hold, :]-mean_vec)/std_vec,
        'test': (perm_data[N_train+N_hold:, :]-mean_vec)/std_vec,
    }
    base = os.path.splitext(os.path.basename(data_path))[0]+'.p'
    save_path = os.path.join(save_dir, base)
    pickle.dump(dataset, open(save_path, 'wb'))


def pca(cov):
    """ Helper function to return PCA transformation given covariance matrix."""
    eigvals, eigvecs = np.linalg.eig(cov)
    sorted_eigvals = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_eigvals].T
    return eigvecs


def get_initmap(X, A=None, standardize=False, cov_func=None):
    """ Give back parameters such that we have the L U decomposition of the
    product with A (if given, or the PCA scores if not).
    That is we will get back:
        X[:, perm]*L*U + b = ((X-meanvec)/stdvec)*A
        where A are PCA directions if not given, L, U are LU decomposition,
        and meanvec, stdvec are zeros, ones vectors if not standardizing.
    Args:
        X: N x d array of training data
        A: d x d linear map to decompose, XA+b, (uses Identity if None given
            with no cov_func).
        standardize: boolean that indicates to standardize the dimensions
            of X after applying linear map.
        cov_func: function that yeilds a linear map given covariance matrix of
            X.
    Returns:
        init_mat: d x d matrix where stricly lower triangle is corresponds to L
            and upper triangle corresponds to U.
        b: d length vector of offset
        perm: permuation of dimensions of X
    """
    # import pdb; pdb.set_trace()  # XXX BREAKPOINT
    N, d = X.shape
    if A is None:
        if cov_func is None:
            A = np.eye(d)
            b = np.zeros((1, d))
        else:
            b = -np.mean(X, 0, keepdims=True)
            M = (X+b)  # Has mean zero.
            cov = np.matmul(M.T, M)/N
            A = cov_func(cov)
            b = np.matmul(b, A)
    if standardize:
        z = np.matmul(X, A)+b
        mean_vec = np.mean(z, 0, keepdims=True)
        # std_vec = np.std(z, 0, keepdims=True)
        # Standardizing may lead to outliers, better to get things in [-1, 1].
        # std_vec = np.max(np.abs(z-mean_vec), 0, keepdims=True)
        std_vec = np.maximum(np.max(np.abs(z-mean_vec), 0, keepdims=True),
                             np.ones((1, d)), dtype=np.float32)
        # import pdb; pdb.set_trace()  # XXX BREAKPOINT
    else:
        mean_vec = np.zeros((1, d))
        std_vec = np.ones((1, d))
    AS = np.divide(A, std_vec)
    P, L, U = linalg.lu(AS)
    perm = np.concatenate([np.flatnonzero(P[:, i]) for i in range(P.shape[1])])
    init_mat = np.tril(L, -1) + U
    init_b = np.squeeze((b-mean_vec)/std_vec)
    return np.float32(init_mat), np.float32(init_b), perm
