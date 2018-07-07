import tensorflow as tf


def leaky_relu(x, alpha):
    return tf.maximum(x, alpha*x)  # Assumes alpha <= 1.0


def general_leaky_relu(x, alpha):
    return tf.nn.relu(x) - alpha*tf.nn.relu(-x)


class Simple1dCell(tf.contrib.rnn.RNNCell):
    """Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs.
    Assumes that alpha in (0, 1]
    """

    def __init__(self, state_size, alpha=None, max_alpha=1.0,
                 state_activation=tf.nn.relu):
        self._state_size = state_size
        self._output_dims = 1
        if alpha is None:
            assert max_alpha <= 1.0
            self._alpha = max_alpha*tf.nn.sigmoid(
                tf.get_variable('log_alpha', initializer=5.0, dtype=tf.float32))
            # self._alpha = tf.minimum(
            #     tf.exp(tf.get_variable('log_alpha', initializer=0.0,
            #                            dtype=tf.float32)),
            #     max_alpha)
            # TODO: get rid of debug code
            # self._alpha = tf.Print(self._alpha, [self._alpha], 'alpha')
        else:
            if isinstance(alpha, float):
                assert alpha <= 1.0
            self._alpha = alpha
        if state_activation is not None:
            self._state_activation = state_activation
        else:
            def lr(x):
                return leaky_relu(x, self._alpha)

            self._state_activation = lr

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_dims

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Output parameters.
            self._w_z_y = tf.get_variable(
                'w_z_y',  # shape=(1,),
                dtype=tf.float32,
                initializer=tf.ones((1,), dtype=tf.float32))
            self._w_z_h = tf.get_variable(
                'w_z_h',  # shape=(self.state_size, 1),
                dtype=tf.float32,
                initializer=tf.zeros((self.state_size, 1), tf.float32))
            self._b_z = tf.get_variable(
                'b_z',  # shape=(1,),
                dtype=tf.float32,
                initializer=tf.zeros((1,), tf.float32))
            # State parameters.
            self._w_h_y = tf.get_variable('w_h_y', shape=(1,), dtype=tf.float32)
            self._w_h_h = tf.get_variable(
                'w_h_h',  # shape=(self.state_size, self.state_size),
                dtype=tf.float32,
                initializer=tf.eye(self.state_size, dtype=tf.float32))
            self._b_h = tf.get_variable(
                'b_h',  # shape=(self._state_size,),
                dtype=tf.float32,
                initializer=tf.zeros((self._state_size,), dtype=tf.float32))
            output = leaky_relu(
                self._w_z_y*inputs + tf.matmul(state, self._w_z_h) +
                self._b_z,
                self._alpha)
            out_state = self._state_activation(
                self._w_h_y*inputs + tf.matmul(state, self._w_h_h) + self._b_h)
        return output, out_state

    def transform(self, inputs, scope='rnn_transform'):
        batch_size = tf.shape(inputs)[0]
        d = int(inputs.get_shape()[1])
        state = self.zero_state(batch_size, dtype=tf.float32)
        z_list = []
        with tf.variable_scope(scope or type(self).__name__):
            for t in range(d):
                z_t, state = self(tf.expand_dims(inputs[:, t], -1), state)
                z_list.append(z_t)
            z = tf.concat(z_list, 1)
        return z

    def inverse(self, output, scope=None):
        """ Computes the inverse mapping for this rnn. May not be numerically
        stable for small |w_z_y|.
        Args:
            output: ? x d tensors
        Returns:
            inverse: ? x d tensor of original values
        """
        batch_size = tf.shape(output)[0]
        d = int(output.get_shape()[1])
        state = self.zero_state(batch_size, dtype=tf.float32)
        y_list = []
        with tf.variable_scope(scope or type(self).__name__):
            for t in range(d):
                z_t = tf.expand_dims(output[:, t], -1)
                z_t_scaled = tf.minimum(z_t, z_t/self._alpha)
                y_t = (z_t_scaled - tf.matmul(state, self._w_z_h) - self._b_z)
                y_t /= self._w_z_y
                y_list.append(y_t)
                state = self._state_activation(
                    self._w_h_y*y_t + tf.matmul(state, self._w_h_h) +
                    self._b_h
                )
            y = tf.concat(y_list, 1)
        return y
