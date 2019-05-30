from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from ..utils import nn
from . import transforms as trans
from . import likelihoods as likes
from . import conditionals as conds


class Model:
    __metaclass__ = ABCMeta

    @abstractmethod
    def build_graph(self, inputs, conditioning=None):
        """ Build graph given inputs of real valued tensors, conditioning
        extraneous data.

        Args:
            inputs: N x ... real tensor of inputs to model.
            conditioning: N x p real tensor of conditioning values to use.

        Returns:
            loss: a loss associated with model to optimize.
            llikes: tensor of log-likelihoods from model.
            sampler: a tensor that returns samples given condtioning values.
        """
        pass


class TANModel(Model):

    # TODO: docstring.
    def __init__(self, transformations, conditional_model,
                 likefunc=likes.mixture_likelihoods, param_nlayers=None,
                 hidden_activation=tf.nn.relu, cond_param_irange=None,
                 dropout_keep_prob=None, nparams=None,
                 preproc_func=None, name='tan', base_distribution='gaussian',
                 sample_size=128,
                 trans_conditioning=True, conditional_conditioning=True,
                 fc_conditioning=True):
        """
        Args:
            transformations: list of transformation functions that take input
                (and possibly conditioning) variables to transform and return
                output, logdet of Jacobian, and inverse for transformation.
            conditional_model: autoregressive conditional model function that
                takes in padded inputs (-1, and the first d-1 covariates),
                function to feed hidden states into (see param_func below),
                conditioning values.
            likefunc: function to give the d likelihoods conditional likelihoods
                for covariates (see likelihoods.mixture_likelihoods).
            param_nlayers: int, number of layers to feed conditional_model's
                hidden state through.
            hidden_activation: activation function to apply to conditional
                hidden states before applying param_func.
            cond_param_irange: scalar range of uniform random initializer for
                hidden state param_func.
            dropout_keep_prob:
            nparams:
            preproc_func:
            name:
            base_distribution:
            sample_size:
            trans_conditioning:
            conditional_conditioning:
            fc_conditioning:
        """
        # Parameters
        self.transformations = transformations
        self.conditional_model = conditional_model
        self.base_likefunc = likefunc
        self.param_nlayers = param_nlayers
        self.hidden_activation = hidden_activation
        self.cond_param_irange = cond_param_irange
        self.dropout_keep_prob = dropout_keep_prob
        self.nparams = nparams
        self.base_distribution = base_distribution
        self.sample_size = sample_size
        self.preproc_func = preproc_func
        self.name = name
        self.trans_conditioning = trans_conditioning
        self.conditional_conditioning = conditional_conditioning
        self.fc_conditioning = fc_conditioning

    def build_graph(self, inputs, conditioning=None, sampler_conditioning=None):
        print('Building {} Graph,\n\tconditioning {}'.format(
            self.name, conditioning))
        # Place holder for model input.
        if self.preproc_func is not None:
            inputs, inv_preproc = self.preproc_func(inputs)
        else:
            inv_preproc = None
        inputs_shape = inputs.get_shape().as_list()[1:]
        # Flatten to trest as real vector.
        inputs = tf.reshape(inputs, (-1, np.prod(inputs_shape)), 'inputs_flat')
        self.d = int(inputs.get_shape()[1])

        # Sampling extreneous coditioning values.
        if sampler_conditioning is None:
            sampler_conditioning = conditioning
        else:
            # Allows for sampling procedure to be independent from any
            # placeholder/input.
            assert conditioning is not None  # Need to also train conditioning.

        # Do transformation on input variables.
        # TODO: Regularizer option
        with tf.variable_scope(
            'transformations',
            # regularizer=tf.contrib.layers.l2_regularizer(1.0, 'l2_trans_reg'),
        ) as trans_scope:
            self.z, self.logdet, self.invmap = trans.transformer(
                inputs, self.transformations,
                conditioning if self.trans_conditioning else None)

        # Get conditional parameters, feed through more layers
        # TODO: Regularizer option
        with tf.variable_scope(
            'conditionals',
            # regularizer=tf.contrib.layers.l2_regularizer(1.0, 'l2_cond_reg'),
        ):
            if self.param_nlayers is not None:

                def param_func(params, conditioning=None,
                               use_conditioning=self.fc_conditioning):
                    """ Fully connected layers on top of hidden states to get
                    parameters to conditional densities.
                    Args:
                        params: N x d x s, or N x s
                        conditioning: N x p
                    """
                    nhidparams = int(params.get_shape()[-1])
                    self.nparams = nhidparams if self.nparams is None else \
                        self.nparams
                    params_in = self.hidden_activation(params) \
                        if self.hidden_activation is not None else params
                    if use_conditioning and conditioning is not None:
                        if len(params.get_shape()) == 3:
                            tiled_conditioning = tf.tile(
                                tf.expand_dims(conditioning, 1), [1, self.d, 1])
                        else:
                            tiled_conditioning = conditioning
                        # TODO: handle case when params is 1 x d x s and should
                        #       be tiled.
                        params_in = tf.concat(
                            (params_in, tiled_conditioning), -1)
                    return nn.fc_network(
                        params_in, self.nparams,
                        [nhidparams]*self.param_nlayers,
                        output_init_range=self.cond_param_irange,
                        name='cond_fc_net',
                        dropout_input=self.dropout_keep_prob is not None,
                        dropout_keep_prob=self.dropout_keep_prob
                    )
            else:
                param_func = None

            self.cond_inputs, self.cond_targets = conds.make_in_out(self.z)
            self.cond_params, self.cond_sampler = self.conditional_model(
                self.cond_inputs, param_func, conditioning)
            # Make transformed space samples.
            self.z_samples = self.cond_sampler(
                self.sample_size, self.base_distribution, sampler_conditioning)

        # Invert to get samples back in original space.
        with tf.variable_scope(trans_scope, reuse=True):
            self.sampler = self.invmap(self.z_samples, sampler_conditioning)
            self.sampler = tf.reshape(self.sampler, [-1]+inputs_shape)
        if inv_preproc is not None:
            self.sampler = inv_preproc(self.sampler)

        # Get likelihoods of targets.
        with tf.variable_scope('likelihoods'):
            self.nll, self.llikes = likes.make_nll_loss(
                self.cond_params, self.cond_targets, self.logdet, self.likefunc
            )

        return self.nll, self.llikes, self.sampler

    def likefunc(self, params, targets):
        """ Helper function for make_nnl_loss. """
        return self.base_likefunc(
            params, targets, base_distribution=self.base_distribution)
