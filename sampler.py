import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from naf import BNAFSampler


class Sampler():
    """Class for a neural Monte Carlo sampler. The sampler has variables that 
    are trained to minimize the Monte Carlo variance. Must implement the `sample'
    and the `log_prob' methods.
    """

    def sample(self, batch_size):
        raise NotImplementedError

    def log_prob(self, x):
        raise NotImplementedError


class BlockAutoregressiveSampler(Sampler):
    """Generative neural sampler with a block autoregressive flow ansatz.

    Args:
        bosonic_dim (int): the number of bosonic coordinates.
        fermionic_dim (int): the number of fermionic coordinates.
    """

    def __init__(self, bosonic_dim=0, fermionic_dim=0):
        self.dim_b = bosonic_dim
        self.dim_f = fermionic_dim
        super().__init__()

    @property
    def trainable_variables(self):
        return self.bosonic_network.trainable_variables

    def _get_bosonic_autoregressive(self):
        dim = self.dim_b
        if not hasattr(self, "bosonic_network"):
            self.bosonic_network = BNAFSampler([dim, 1 * dim, dim])
        return self.bosonic_network

    def sample(self, batch_size, with_log_prob=False):
        # sample bosonic variables first
        dist_b = self._get_bosonic_autoregressive()
        sample_b, log_prob = dist_b.sample(batch_size)
        assert sample_b.shape == [batch_size, self.dim_b], f"bosonic sample shape error: {sample_b.shape}"
        # sample fermionic variables randomly
        if self.dim_f > 0:
            dist_f = tfd.Independent(tfd.Bernoulli(logits=[0]*self.dim_f), reinterpreted_batch_ndims=1)
            sample_f = tf.cast(dist_f.sample(batch_size), tf.float32)
            f_log_prob = dist_f.log_prob(sample_f)
            log_prob = log_prob + f_log_prob
        else:
            sample_f = tf.zeros((batch_size, 0))
            f_log_prob = tf.zeros((batch_size,))
        assert sample_f.shape == [batch_size, self.dim_f], f"fermionic sample shape error: {sample_f.shape}"
        # final result 
        sample = tf.concat([sample_b, sample_f], axis=-1)
        if with_log_prob:
            # also return the log probs of the generated samples
            return sample, log_prob
        return sample

