import tensorflow as tf


def get_dense(dims, zero_initialized=False):
    """Return a fully connected network with given dimensions. We use tanh as activation functions in hidden layers.

    Args:
        dims (list of ints): dimensions of the input layer (dims[0]), hidden layers (dims[1:-1]) and output layer (dims[-1])
        zero_initialized (bool): whether to initialize the output values as zeros

    Return:
        model (Sequential): the tf module for the network
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(dims[0],)))
    for i in range(1, len(dims)):
        activation = "tanh" if i < len(dims) - 1 else None # no activation for output
        bias_init = "glorot_uniform" if i < len(dims) - 1 or not zero_initialized else "zeros"
        model.add(tf.keras.layers.Dense(dims[i], activation=activation, kernel_initializer="zeros", bias_initializer=bias_init))
    return model


class ConditionalDenseNetwork():
    """Class for fully connected neural networks with weights and biases dependent on other parameters. 
    The dependencies are parametrized by dense networks as well. We use tanh as activation functions.
    """

    def __init__(self, dims, dim_cond):
        """
        Args:
            dims (list of ints): dimensions of the input layer (dims[0]), hidden layers (dims[1:-1]) and output layer (dims[-1])
            dim_cond (int): the number of parameters that the weights and biases depend on
        """
        self.dims = dims
        self.dim_cond = dim_cond
        # the weights and biases depend on dim_cond numbers via dense networks with one hidden layer
        # the weights are initialized to be zero to avoid explosion of gradients
        self.denses = [get_dense([dim_cond, dim_cond, dims[i] * dims[i+1]], zero_initialized=True) for i in range(len(dims) - 1)]
        self.biases = [get_dense([dim_cond, dim_cond, dims[i+1]]) for i in range(len(dims) - 1)]

    @property
    def trainable_variables(self):
        var = []
        for _dense in self.denses:
            var = var + _dense.trainable_variables
        for _bias in self.biases:
            var = var + _bias.trainable_variables
        return var

    def __call__(self, x, cond):
        """Return forward pass of input x, with parameters given by cond.

        Args:
            x (tensor of shape (..., dims[0])): the input
            cond (tensor of shape (..., dim_cond)): the conditional parameters

        Return:
            y (tensor of shape (..., dims[-1])): the output
        """
        assert x.shape[:-1] == cond.shape[:-1], "incompatible batch shapes"
        assert int(x.shape[-1]) == self.dims[0] and int(cond.shape[-1]) == self.dim_cond
        y = x
        for i in range(len(self.denses)):
            d_in, d_out = self.dims[i], self.dims[i + 1]
            assert int(y.shape[-1]) == d_in
            dense, bias = self.denses[i](cond), self.biases[i](cond)
            y = tf.linalg.matvec(tf.reshape(dense, dense.shape[:-1] + [d_out, d_in]), y) + bias
            if i < len(self.denses) - 1:
                y = tf.math.tanh(y)
        return y

