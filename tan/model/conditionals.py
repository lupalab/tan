"""
Conditional component to TANs, and code to sample.
- Conditional models are functions that
  - take in:
    - inputs of transformed covariates
      (padded, excluding last dimension)`[N x d]`
    - (possibly) extraneous data to condition on `[N x p]`
    - (possibly) function to pass
      `[N x d x nparams]` params and conditioning values
  - return:
    - parameters for the conditional density of each
      dimension `[N x d x nparams]`
    - sampler function that takes in a batch size and type
      of base component distribution and outputs sample `[batch x d]`
"""

import tensorflow as tf
from ..utils import linear


def make_in_out(y, prepend_func=tf.ones, scope='make_in_out'):
    """Takes map y=transform(x) and prepends -1.
    Args:
        y: N x d tensor of variables whose conditionals you want to model.
        prepend_func: function to make prepending values.
        labels: N x c tensor of labeled (conditioning) values to append to
            input section of y output
    Returns:
        y_input: N x d tensor of transformed data to feed into conditional
            predictor as inputs. (Contains -1 and the first d-1 dimensions.)
        y_output: N x d x 1 tensor of transformed data to feed into conditional
            predictor as outputs. (Contains the dth dimensions.)
    """
    with tf.variable_scope(scope):
        d = int(y.get_shape()[1])
        # Prepend nulls.
        y_cat = tf.concat(
            (-prepend_func((tf.shape(y)[0], 1), dtype=tf.float32), y), 1
        )
        # Make input and output sections of y.
        y_input = tf.slice(y_cat, [0, 0], [-1, d])
    return y_input, tf.expand_dims(y, -1)


def sample_mm(params_dim, base_distribution='gaussian'):
    """ Helper function to sample from a 1d mixture model.
    Args:
        params_dim: N x (ncomps*3) real tensor of
            (logit weights, means, logsigmas).
        base_distribution: string in {'gaussian', 'laplace', 'logistic'}
            specifying what the base distribution for mixture components is.
    Returns:
        samp: N x 1 real tensor of sample points.
    """
    base_distribution = base_distribution.lower()
    batch_size = tf.to_int64(tf.shape(params_dim)[0])
    logits, means, lsigmas = tf.split(params_dim, 3, 1)
    sigmas = tf.exp(lsigmas, name='sigmas')
    # sample multinomial
    js = tf.multinomial(logits, 1, name='js')  # int64
    inds = tf.concat(
        (tf.expand_dims(tf.range(batch_size, dtype=tf.int64), -1), js),
        1, name='inds')
    # Sample from base distribution.
    if base_distribution == 'gaussian':
        zs = tf.random_normal((batch_size, 1))
    elif base_distribution == 'laplace':
        zs = tf.log(tf.random_uniform((batch_size, 1))) - \
            tf.log(tf.random_uniform((batch_size, 1)))
    elif base_distribution == 'logistic':
        x = tf.random_uniform((batch_size, 1))
        zs = tf.log(x) - tf.log(1.0-x)
    else:
        raise NotImplementedError
    # scale and shift
    mu_zs = tf.expand_dims(
        tf.gather_nd(means, inds, name='mu_zs'), -1)
    sigma_zs = tf.expand_dims(
        tf.gather_nd(sigmas, inds, name='sigma_zs'), -1)
    samp = sigma_zs*zs + mu_zs
    return samp


def independent_model(inputs, nparams, single_marginal=False,
                      standard=False, param_func=None, conditioning=None,
                      use_conditioning=False):
    """ Independent conditional model, where the hidden state for the conditional
    of the ith dimension is a nparams length vector (independent of previous
    dimensions).

    Args:
        inputs: N x d real tensor of the input covariates. (Ignored, only used
            for shape information.)
        nparams: int of number of parameters to output per dimension.
        single_marginal: boolean indicating if all conditionals are the same.
        standard: boolean indicating if params are untrainable zeros
            (corresponding to some standard component, e.g. 0 mean, unit std).
        param_func: optional function to apply on the N x d x nparams parameter
            tensor and conditioning values.
        conditioning: N x p real tensor of values to condition on
        use_condtioning: boolean indicating whether to use the conditioning
            values within the model (conditioning is passed along to param_func
            regardless).
    Returns:
        params: 1 x d x nparams tensor of parameters for the conditional density
            of each dimension. (Is used on each of N instances, since it's
            independent of covariates.)
        sampler: function that takes in a batch size and base component
            distribution and outputs a tensor of batch_size x d of samples.
    """
    d = int(inputs.get_shape()[1])
    if single_marginal:
        d_mod = 1
    else:
        d_mod = d
    with tf.variable_scope('independent_model') as scope:
        if standard:
            params = tf.get_variable('independent', dtype=tf.float32,
                                     trainable=False,
                                     initializer=tf.zeros((1, d_mod, nparams)))
        else:
            params = tf.get_variable('independent', dtype=tf.float32,
                                     trainable=True,
                                     initializer=tf.zeros((1, d_mod, nparams)))
        if param_func is not None:
            with tf.variable_scope('param_func'):
                params = param_func(params, conditioning)

    # sampling code
    def sampler(batch_size, base_distribution='gaussian', conditioning=None):
        with tf.variable_scope(scope, reuse=True):
            # assuming padding with -1 to start.
            y_dims = []
            params_sqz = tf.squeeze(params, 0)
            for j in range(d):
                if single_marginal:
                    params_dim = params_sqz
                else:
                    params_dim = tf.expand_dims(tf.gather(params_sqz, j), 0)
                if param_func is not None:
                    with tf.variable_scope('param_func', reuse=True):
                        params_dim = param_func(params_dim, conditioning)
                params_dim = tf.tile(params_dim, [batch_size, 1])
                input_ = sample_mm(params_dim,
                                   base_distribution=base_distribution)
                y_dims.append(input_)
            y = tf.concat(y_dims, 1, 'y_samp')
        return y
    return params, sampler


# TODO: simplify?
# TODO: change to lam_model.
# pylama:ignore=C901, Complexity is in the eye of the beholder.
def cond_model(inputs, nparams, tied_model=False, tied_bias=True,
               param_func=None, conditioning=None, use_conditioning=True):
    """ LAM linear conditional model, where the hidden state for the conditional
    of the ith dimension is:
    \[
        h_i = W^{(i)} x_{<i} + b^{(i)}
    \]

    Args:
        inputs: N x d real tensor of the input covariates.
        nparams: int of number of parameters to output per dimension.
        tied_model: boolean indicating if the linear weights W  are shared
            across conditional tasks (like NADE).
        tied_bias: boolean indicating whether to keep biases tied, b^{(i)} = b.
        param_func: optional function to apply on the N x d x nparams parameter
            tensor and conditioning values.
        conditioning: N x p real tensor of values to condition on
        use_condtioning: boolean indicating whether to use the conditioning
            values within the model (conditioning is passed along to param_func
            regardless).
    Returns:
        params: N x d x nparams tensor of parameters for the coditional density
            of each dimension.
        sampler: function that takes in a batch size and base component
            distribution and outputs a tensor of batch_size x d of samples.
    """
    # Set up conditioning values if not used.
    param_conditioning = conditioning
    if not use_conditioning:
        conditioning = None
    with tf.variable_scope('linear_cond_model') as scope:
        d = int(inputs.get_shape()[1])
        # Get weights for covariates contributions.
        if tied_model:
            W = tf.get_variable('W', (d, nparams), dtype=tf.float32)
            Ws = [
                tf.slice(W, [0, 0], [j+1, -1], name='W{}'.format(j))
                for j in range(d)
            ]
        else:
            Ws = [
                tf.get_variable('W{}'.format(j), (j+1, nparams),
                                dtype=tf.float32)
                for j in range(d)
            ]
        # Get the biases.
        if tied_bias:
            bs = [tf.get_variable('b', (1, nparams), dtype=tf.float32)]*d
        else:
            bs = [
                tf.get_variable('b{}'.format(j), (1, nparams), dtype=tf.float32)
                for j in range(d)
            ]
        # Get weights for conditioning covariate contributions.
        if conditioning is not None:
            d_cond = int(conditioning.get_shape()[1])
            if tied_model:
                W_cond = tf.get_variable(
                    'W_cond', (d_cond, nparams), dtype=tf.float32)
            else:
                W_conds = [
                    tf.get_variable('W_cond{}'.format(j), (d_cond, nparams),
                                    dtype=tf.float32)
                    for j in range(d)
                ]
        # Go through and compute hidden state for each dimension.
        outs = []
        params_start = 0.0
        # Avoid unneeded computation if conditioning with tied model.
        if conditioning is not None and tied_model:
            params_start = tf.matmul(conditioning, W_cond)
        for j in range(d):
            params_j = params_start
            if conditioning is not None and not tied_model:
                params_j += tf.matmul(conditioning, W_conds[j])
            params_j += \
                tf.matmul(tf.slice(inputs, [0, 0], [-1, j+1]), Ws[j]) + bs[j]
            outs.append(tf.expand_dims(params_j, 1))
        params = tf.concat(outs, 1, 'params')
        if param_func is not None:
            with tf.variable_scope('param_func'):
                params = param_func(params, param_conditioning)

    # sampling code
    def sampler(batch_size, base_distribution='gaussian', conditioning=None):
        param_conditioning = conditioning
        if not use_conditioning:
            conditioning = None
        with tf.variable_scope(scope, reuse=True):
            # assuming padding with -1 to start.
            y = -tf.ones((batch_size, 1))
            for j in range(d):
                params_dim = params_start
                if conditioning is not None and not tied_model:
                    params_dim += tf.matmul(conditioning, W_conds[j])
                params_dim += tf.matmul(y, Ws[j]) + bs[j]
                if param_func is not None:
                    with tf.variable_scope('param_func', reuse=True):
                        params_dim = param_func(params_dim, param_conditioning)
                input_ = sample_mm(params_dim,
                                   base_distribution=base_distribution)
                y = tf.concat((y, input_), 1, 'y_samp{}'.format(j))
            y = tf.slice(y, [0, 1], [-1, -1], 'y_samp')
        return y
    return params, sampler


# TODO: change to ram_model.
def rnn_model(inputs, nparams, rnn_class, param_func=None, conditioning=None,
              conditioning_dim=None, use_conditioning=True):
    """ RAM RNN conditional model, where the hidden state for the conditional
    of the ith dimension is:
    \[
        h_i = g(x_{i-1}, h_{i-1})
    \]

    Args:
        inputs: N x d real tensor of the input covariates.
        nparams: int of number of parameters to output per dimension.
        rnn_class: function that returns an rnn_cell that outputs nparams when
            called rnn_class(nparams).
        param_func: optional function to apply on the N x d x nparams parameter
            tensor and conditioning values.
        conditioning: N x p real tensor of values to condition on
        conditioning_dim: scalar, if not None, linearly projects conitioning to
            vectors of length condtioning_dim from p.
        use_condtioning: boolean indicating whether to use the conditioning
            values within the model (conditioning is passed along to param_func
            regardless).
    Returns:
        params: N x d x nparams tensor of parameters for the coditional density
            of each dimension.
        sampler: function that takes in a batch size and base component
            distribution and outputs a tensor of batch_size x d of samples.
    """
    param_conditioning = conditioning
    if not use_conditioning:
        conditioning = None
    with tf.variable_scope('rnn_model') as scope:
        rnn_cell = rnn_class(nparams)
        d = int(inputs.get_shape()[1])
        inputs = tf.expand_dims(inputs, -1)
        # If conditioning on values, make values visible to RNN at each
        # dimension by concating them to input dims.
        if conditioning is not None:
            if conditioning_dim is not None:
                conditioning = linear.linear(conditioning, conditioning_dim)
            tiled_conditioning = tf.tile(
                tf.expand_dims(conditioning, 1), [1, d, 1])
            inputs = tf.concat(
                (inputs, tiled_conditioning), 2)
        params = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)[0]
        if param_func is not None:
            with tf.variable_scope('param_func'):
                params = param_func(params, param_conditioning)

    # sampling code
    def sampler(batch_size, base_distribution='gaussian', conditioning=None):
        param_conditioning = conditioning
        if not use_conditioning:
            conditioning = None
        with tf.variable_scope(scope, reuse=True):
            state = rnn_cell.zero_state(batch_size, tf.float32)
            # assuming padding with -1 to start.
            input_ = -tf.ones((batch_size, 1))
            y_dims = []
            for j in range(d):
                if conditioning is not None:
                    if conditioning_dim is not None:
                        conditioning = linear.linear(
                            conditioning, conditioning_dim)
                    input_ = tf.concat((input_, conditioning), 1)
                with tf.variable_scope('rnn'):
                    params_dim, state = rnn_cell(input_, state)
                if param_func is not None:
                    with tf.variable_scope('param_func'):
                        params_dim = param_func(params_dim, param_conditioning)
                input_ = sample_mm(params_dim,
                                   base_distribution=base_distribution)
                y_dims.append(input_)
            y = tf.concat(y_dims, 1, 'y_samp')
        return y
    return params, sampler
