import tensorflow as tf
from ..utils import misc
import sru
import utils


class GRUCell:

    def __init__(self, **kwargs):
        self._units = misc.get_default(kwargs, 'units', 256)
        self._num_layers = misc.get_default(kwargs, 'num_layers', 1)

    def __call__(self, nout):
        if self._num_layers > 1:
            gru_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(self._units)
                 for _ in range(self._num_layers)]
            )
        else:
            gru_cell = tf.contrib.rnn.GRUCell(self._units)
        return tf.contrib.rnn.OutputProjectionWrapper(gru_cell, nout)


class GRUResidual:

    def __init__(self, **kwargs):
        self._units = misc.get_default(kwargs, 'units', 256)
        self._num_layers = misc.get_default(kwargs, 'num_layers', 1)

    def __call__(self, nout):
        if self._num_layers > 1:
            gru_cell = tf.contrib.rnn.MultiRNNCell(
                [utils.ProjectedResidualWrapper(
                    tf.contrib.rnn.GRUCell(self._units))
                 for _ in range(self._num_layers)]
            )
        else:
            gru_cell = tf.contrib.rnn.GRUCell(self._units)
            gru_cell = utils.ProjectedResidualWrapper(gru_cell)
        return tf.contrib.rnn.OutputProjectionWrapper(gru_cell, nout)


class LSTMCell:

    def __init__(self, **kwargs):
        self._units = misc.get_default(kwargs, 'units', 256)
        self._num_layers = misc.get_default(kwargs, 'num_layers', 1)

    def __call__(self, nout):
        if self._num_layers > 1:
            lstm_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.BasicLSTMCell(
                    self._units, state_is_tuple=False)
                 for _ in range(self._num_layers)]
            )
        else:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self._units,
                                                     state_is_tuple=False)
        return tf.contrib.rnn.OutputProjectionWrapper(lstm_cell, nout)


class SRUCell:

    def __init__(self, **kwargs):
        self._units = misc.get_default(kwargs, 'units', 256)
        self._num_stats = misc.get_default(kwargs, 'num_stats', self._units)
        self._mavg_alphas = misc.get_default(
            kwargs, 'mavg_alphas', [0.0, 0.5, 0.9, 0.99, 0.999])
        self._recur_dims = misc.get_default(kwargs, 'recur_dims', self._units/4)
        self._num_layers = misc.get_default(kwargs, 'num_layers', 1)

    def __call__(self, nout):
        if self._num_layers > 1:
            sru_cell = tf.contrib.rnn.MultiRNNCell(
                [sru.SimpleSRUCell(
                    num_stats=self._num_stats,
                    mavg_alphas=tf.constant(self._mavg_alphas),
                    output_dims=self._units,
                    recur_dims=self._recur_dims,
                    linear_out=False,
                    include_input=False)
                 for _ in range(self._num_layers)]
            )
        else:
            sru_cell = sru.SimpleSRUCell(
                num_stats=self._num_stats,
                mavg_alphas=tf.constant(self._mavg_alphas),
                output_dims=self._units,
                recur_dims=self._recur_dims,
                linear_out=False,
                include_input=False)
        return tf.contrib.rnn.OutputProjectionWrapper(sru_cell, nout)


class GRUSRUCell:

    def __init__(self, **kwargs):
        self._units = misc.get_default(kwargs, 'units', 256)
        self._num_stats = misc.get_default(kwargs, 'num_stats', self._units)
        self._mavg_alphas = misc.get_default(
            kwargs, 'mavg_alphas', [0.0, 0.5, 0.9, 0.99, 0.999])
        self._recur_dims = misc.get_default(kwargs, 'recur_dims', self._units/4)
        self._num_layers = misc.get_default(kwargs, 'num_layers', 1)

    def __call__(self, nout):
        if self._num_layers > 1:
            joint_cell = tf.contrib.rnn.MultiRNNCell(
                [utils.JointCell(
                    tf.contrib.rnn.GRUCell(self._units),
                    sru.SimpleSRUCell(
                        num_stats=self._num_stats,
                        mavg_alphas=tf.constant(self._mavg_alphas),
                        output_dims=self._units,
                        recur_dims=self._recur_dims,
                        linear_out=False,
                        include_input=False))
                 for _ in range(self._num_layers)]
            )
        else:
            joint_cell = utils.JointCell(
                tf.contrib.rnn.GRUCell(self._units),
                sru.SimpleSRUCell(
                    num_stats=self._num_stats,
                    mavg_alphas=tf.constant(self._mavg_alphas),
                    output_dims=self._units,
                    recur_dims=self._recur_dims,
                    linear_out=False,
                    include_input=False)
            )
        return tf.contrib.rnn.OutputProjectionWrapper(joint_cell, nout)
