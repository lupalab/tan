import tensorflow as tf
from ..utils import nn


class JointCell(tf.contrib.rnn.RNNCell):
    """ Run dual RNN cells (side by side). """

    def __init__(self, cell1, cell2):
        self.cell1 = cell1
        self.cell2 = cell2

    @property
    def state_size(self):
        return int(self.cell1.state_size) + int(self.cell2.state_size)

    @property
    def output_size(self):
        return int(self.cell1.output_size) + int(self.cell2.output_size)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            state1, state2 = tf.split(
                state,
                [int(self.cell1.state_size), int(self.cell2.state_size)],
                1)
            out1, ostate1 = self.cell1(inputs, state1)
            out2, ostate2 = self.cell2(inputs, state2)
            output = tf.concat((out1, out2), 1)
            ostate = tf.concat((ostate1, ostate2), 1)
        return output, ostate


class ProjectedResidualWrapper(tf.contrib.rnn.RNNCell):
    """RNNCell wrapper that ensures cell inputs are added to the outputs."""

    def __init__(self, cell):
        """Constructs a `ResidualWrapper` for `cell`.
        Args:
            cell: An instance of `RNNCell`.
        """
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope='ProjResid'):
        """Run the cell and then apply the projected residual on its inputs to
        its outputs.
        Args:
            inputs: cell inputs.
            state: cell state.
            scope: optional cell scope.
        Returns:
            Tuple of cell outputs and new state.
        Raises:
            TypeError: If cell inputs and outputs have different structure
                (type).
            ValueError: If cell inputs and outputs have different structure
                (value).
        """
        with tf.variable_scope(scope):
            outputs, new_state = self._cell(inputs, state)
            res_outputs = inputs + nn.fc_network(
                outputs, inputs.get_shape()[-1], [])
        return (res_outputs, new_state)
