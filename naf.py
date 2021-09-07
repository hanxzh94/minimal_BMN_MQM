import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
import numpy as np
import math


def get_weight_mask(dim_in, dim_out, num_blocks):
    """Return the block autoregressive diagonal and off-diagonal masks of the weight.

    Args:
        dim_in (int): input dimension of the weight
        dim_out (int): output dimension of the weight
        num_blocks (int): the number of autoregressive blocks

    Return:
        mask_d (tensor of shape (dim_out, dim_in)): a binary mask where the weights within each block are ones
        mask_o (tensor of shape (dim_out, dim_in)): a binary mask where the weights from block i to block j and i < j are ones
    """
    assert dim_in % num_blocks == 0 and dim_out % num_blocks == 0, f"invalid {dim_in}, {dim_out}, {num_blocks}"
    step_in, step_out =  dim_in // num_blocks, dim_out // num_blocks
    # mask for the block diagonals of the weights
    mask_d = np.zeros((dim_out, dim_in))
    for i in range(num_blocks):
        mask_d[i * step_out : (i + 1) * step_out, i * step_in : (i + 1) * step_in] += 1
    # mask for the block autoregressive off-diagonals
    mask_o = np.ones((dim_out, dim_in))
    for i in range(num_blocks):
        mask_o[i * step_out : (i + 1) * step_out, i * step_in : ] -= 1
    return tf.constant(mask_d, dtype=tf.float32), tf.constant(mask_o, dtype=tf.float32)

def block_diag(mat):
    """Construct block diagonal matrices for a batch of diagonal blocks.

    Args:
        mat (tensor of shape (batch_size, num_blocks, dim_out, dim_in)): a batch of diagonal blocks of matrices

    Return:
        res (tensor of shape (batch_size, num_blocks * dim_out, num_blocks * dim_in)): the block diagonal matrices from the input
    """
    shape = mat.shape.as_list()
    num_blocks, dim_out, dim_in = shape[-3], shape[-2], shape[-1]
    # batch_size, num_blocks, dim_out, dim_in = mat.shape.as_list()
    diag = np.zeros((num_blocks, num_blocks, num_blocks))
    for i in range(num_blocks):
        diag[i, i, i] = 1
    return tf.reshape(tf.einsum("bijk,imn->bmjnk", tf.reshape(mat, (-1, num_blocks, dim_out, dim_in)), tf.constant(diag, dtype=tf.float32)), shape[:-3] + [num_blocks * dim_out, num_blocks * dim_in])


class BNAF(tfb.Bijector):
    """Class for block neural autoregressive flow. See https://github.com/nicola-decao/BNAF for details."""

    @classmethod
    def num_variables(cls, dims):
        """Returns the total number of trainable variables required."""
        tot = 0
        assert dims[0] == dims[-1], "input and output dimensions should be equal"
        num_blocks = dims[0] # assuming that each input unit is one block
        for i in range(len(dims) - 1):
            dim_in, dim_out = dims[i], dims[i + 1]
            step_in, step_out = dim_in // num_blocks, dim_out // num_blocks
            # number of weights + number of biases + number of diagonal weights
            tot += dim_in * dim_out + dim_out + num_blocks * step_in * step_out
            if i < len(dims) - 2:
                tot += 2 * dim_out # params for the activation; output layer excluded
        return tot

    def __init__(self, dims, w=None, num=1):
        super().__init__(forward_min_event_ndims=1)
        assert dims[0] == dims[-1], "first and last dimensions must be the same"
        assert all(d % dims[0] == 0 for d in dims), "input dimension must divide all hidden dimensions"
        self.dims = dims
        if w is None:
            # initialize the parameters
            # num is a batch size in case we need multiple flows in parallel
            cur = 0
            w = np.zeros((num, self.num_variables(dims)))
            for i in range(len(dims) - 1):
                dim_in, dim_out = dims[i], dims[i + 1]
                # xavier initialization
                scale = 1 / max(1., (dim_in + dim_out)/2.)
                limit = math.sqrt(3.0 * scale)
                w[..., cur : cur + dim_in * dim_out] = np.random.uniform(-limit, limit, size=(num, dim_in * dim_out))
                cur += dim_in * dim_out + dim_in * dim_out // dims[0]
                scale = 1 / max(1., dim_out)
                limit = math.sqrt(1.0 * scale)
                w[..., cur : cur + dim_out] = np.random.uniform(-limit, limit, size=(num, dim_out))
                cur += dim_out
                # params for the activation
                if i < len(dims) - 2:
                    cur += 2 * dim_out
            assert cur == self.num_variables(dims) # all variables should be used up
            self.weight = tf.Variable(w, dtype=tf.float32)
        else:
            # the parameters are given by w
            self.weight = w

    @property
    def trainable_variables(self):
        return [self.weight]
    
    def forward(self, x):
        """Forward pass of the flow; output has the same shape as the input."""
        dims, weight = self.dims, self.weight
        assert int(x.shape[-1]) == dims[0], "dimension mismatch"
        num_blocks = dims[0]
        cur = 0 # counting the number of parameters we have used
        fwd_x = x
        # log_jac[..., i, j] is log d y_ij / d x_i
        # y_ij are hidden variables in block i and x_i are inputs
        log_jac = tf.expand_dims(tf.zeros_like(x), axis=-1) # initially each block has one variable
        for i in range(len(dims) - 1):
            dim_in, dim_out = dims[i], dims[i + 1]
            step_in, step_out = dim_in // num_blocks, dim_out // num_blocks
            assert int(fwd_x.shape[-1]) == dim_in
            assert log_jac.shape.as_list()[-2:] == [num_blocks, step_in]
            # get parameters from vectors
            def get_params(cur, num):
                return weight[..., cur : cur + num], cur + num
            # autoregressive weight with positive block diagonals
            _w, cur = get_params(cur, dim_in * dim_out)
            _w = tf.reshape(_w, _w.shape[:-1] + [dim_out, dim_in])
            mask_d, mask_o = get_weight_mask(dim_in, dim_out, num_blocks)
            w = tf.math.exp(_w) * mask_d + _w * mask_o
            # normalization (not necessary?)
            w_squared_norm = tf.math.reduce_sum(tf.math.square(w), axis=-1, keepdims=True) # shape (..., dim_out, 1)
            _w_diag, cur = get_params(cur, num_blocks * step_in * step_out)
            _w_diag = tf.reshape(_w_diag, _w_diag.shape[:-1] + [num_blocks, step_out, step_in])
            w = tf.math.exp(block_diag(_w_diag)) * w / tf.math.sqrt(w_squared_norm)
            w_diag = tf.stack([w[..., j * step_out : (j + 1) * step_out, j * step_in : (j + 1) * step_in] for j in range(num_blocks)], axis=-3)
            assert w_diag.shape.as_list()[-3:] == [num_blocks, step_out, step_in]
            # bias
            b, cur = get_params(cur, dim_out)
            # forward and compute Jacobian
            fwd_x = tf.squeeze(tf.linalg.matmul(w, tf.expand_dims(fwd_x, axis=-1)), axis=-1) + b
            log_jac = tf.math.reduce_logsumexp(tf.math.log(w_diag) + tf.expand_dims(log_jac, axis=-2), axis=-1)
            assert log_jac.shape.as_list()[-2:] == [num_blocks, step_out]
            if i < len(dims) - 2:
                # SinhArcsinh activations
                params, cur = get_params(cur, dim_out * 2)
                p0, p1 = params[..., :dim_out], tf.math.exp(params[..., dim_out:])
                activation = tfb.SinhArcsinh(skewness=p0, tailweight=p1)
                # activation bijector
                log_d = tf.reshape(activation.forward_log_det_jacobian(fwd_x, event_ndims=0), fwd_x.shape[:-1] + [num_blocks, step_out])
                log_jac = log_jac + log_d
                fwd_x = activation(fwd_x)
            assert int(fwd_x.shape[-1]) == dim_out
        logdet = tf.reduce_sum(log_jac, axis=[-1, -2])
        assert logdet.shape == x.shape[:-1]
        # check that we have used all parameters
        assert cur == int(self.weight.shape[-1]), "unused parameters"
        return fwd_x, logdet


class BNAFDensityEstimator():
    """Density estimator with BNAF."""

    def __init__(self, dims, w=None):
        self.dim = dims[0]
        self.flow = BNAF(dims, w=w)

    @property
    def trainable_variables(self):
        return self.flow.trainable_variables

    def log_prob(self, x):
        assert int(x.shape[-1]) == self.dim, "incompatible input shape"
        base = tfd.Independent(tfd.Normal(loc=[0]*self.dim, scale=[1]*self.dim), reinterpreted_batch_ndims=1)
        y, logdet = self.flow.forward(x)
        return base.log_prob(y) + logdet


class BNAFSampler():
    """Sampler with BNAF."""

    def __init__(self, dims):
        self.dim = dims[0]
        self.flow = BNAF(dims)

    @property
    def trainable_variables(self):
        return self.flow.trainable_variables

    def sample(self, batch_size):
        base = tfd.Independent(tfd.Normal(loc=[0]*self.dim, scale=[1]*self.dim), reinterpreted_batch_ndims=1)
        x = base.sample(batch_size)
        y, logdet = self.flow.forward(x)
        # return both the samples and their log probs
        return y, base.log_prob(x) - logdet

