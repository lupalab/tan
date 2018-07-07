"""likelihoods.py

Logic to get likelihoods given parameters to conditional mixtures.
- `mixture_likelihoods` function produces log likelihoods on transformed
covariates given parameters
- `make_nll_loss` gives negative log likelihood of data in batch
"""

import tensorflow as tf
import numpy as np


def mixture_likelihoods(params, targets, base_distribution='gaussian',
                        name='mixture_likelihood'):
    """Given log-unnormalized mixture weights, shift, and log scale parameters
    for mixture components, return the likelihoods for targets.
    Args:
        params: N x d x 3*ncomp tensor of parameters of mixture model
            where weight_logits, means, log_sigmas = tf.split(params, 3, 2).
        targets: N x d x 1 tensor of 1d targets to get likelihoods for.
        base_distribution: {'gaussian', 'laplace', 'logistic'} the base
            distribution of mixture components.
    Return:
        likelihoods: N x d  tensor of likelihoods.
    """
    base_distribution = base_distribution.lower()
    with tf.variable_scope(name):
        # Compute likelihoods per target and component
        # Write log likelihood as logsumexp.
        logits, means, lsigmas = tf.split(params, 3, 2)
        sigmas = tf.exp(lsigmas)
        if base_distribution == 'gaussian':
            log_norm_consts = -lsigmas - 0.5*np.log(2.0*np.pi)
            log_kernel = -0.5*tf.square((targets-means)/sigmas)
        elif base_distribution == 'laplace':
            log_norm_consts = -lsigmas - np.log(2.0)
            log_kernel = -tf.abs(targets-means)/sigmas
        elif base_distribution == 'logistic':
            log_norm_consts = -lsigmas
            diff = (targets-means)/sigmas
            log_kernel = -tf.nn.softplus(diff) - tf.nn.softplus(-diff)
        else:
            raise NotImplementedError
        log_exp_terms = log_kernel + log_norm_consts + logits
        log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1) - \
            tf.reduce_logsumexp(logits, -1)
    return log_likelihoods


def mixture_mse(params, targets, **kwargs):
    """Given log-unnormalized mixture weights for equi-spaced truncated
    Gaussians on the unit interval, return the likelihoods for targets.
    (Currently unused.)
    Args:
        params: N x d x 3*ncomp tensor of parameters of mixture model.
        targets: N x d x 1 tensor of 1d targets to get likelihoods for.
    Return:
        sq_diff: N x d  tensor of negative squared differences.
    """
    with tf.variable_scope('mixture_likelihood'):
        logits, means, lsigmas = tf.split(params, 3, 2)
        weights = tf.nn.softmax(logits, name='comp_weights')
        sigmas = tf.exp(lsigmas)
        # Compute the normalizers for the truncated normals.
        gmm_mus = tf.reduce_sum(means*weights, -1, keep_dims=True)
        gmm_sigmas = tf.reduce_sum(
            (tf.square(sigmas)+tf.square(means-gmm_mus))*weights, -1,
            keep_dims=True
        )
        sq_diff = tf.squeeze(
            tf.square(targets-gmm_mus) + gmm_sigmas, axis=2
        )
    return -0.5*sq_diff


def make_nll_loss(logits, targets, logdetmap, likefunc=mixture_likelihoods,
                  min_like=None):
    """Given log-unnormalized mixture weights for equi-spaced truncated
    Gaussians on the unit interval, return the likelihoods for targets.
    Args:
        logits: N x d x 3*ncomp tensor of log unnormalized logits to be
            softmaxed for respective weights on mixture components.
        targets: N x d x 1 tensor of 1d targets to get likelihoods for.
        logdetmap: N tensor (or scalar) of determinant normalizers
        likefunc: function to compute conditional log likelihoods on each
            dimension.
        min_like: scalar Minimum likelihood to truncate, (None used by default).
    Return:
        loss: scalar nll on batch.
        ll: N tensor of log likelihoods.
    """
    with tf.variable_scope('nll_loss'):
        lls = log_likelihoods(logits, targets, logdetmap, likefunc=likefunc,
                              min_like=min_like)
        loss = -tf.reduce_mean(lls)
    return loss, lls


def log_likelihoods(logits, targets, logdetmap, likefunc=mixture_likelihoods,
                    min_like=None):
    """Convinience function that returns the unavaraged tensor of log
    likelihoods.
    """
    with tf.variable_scope('ll'):
        cllikes = likefunc(logits, targets)
        mix_ll = tf.reduce_sum(cllikes, -1)
        lls = logdetmap + mix_ll
        if min_like is not None:
            trunc_lls = tf.maximum(lls, np.log(min_like), 'trunc_likes')
        else:
            trunc_lls = lls
    return trunc_lls
