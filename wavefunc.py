import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from dense import ConditionalDenseNetwork
from naf import BNAFDensityEstimator


class Wavefunction():
    """Class for a neural quantum wavefunction. Wavefunctions map coordinate
    vectors to complex amplitudes. Must implement the `log_psi' method.
    """

    def __init__(self, log_psi_func=None):
        self.log_psi_func = log_psi_func

    @property
    def trainable_variables(self):
        return []

    def log_psi(self, x):
        """Compute the log amplitude of the wavefunction at the given coordinates.

        Args:
            x (tensor): a batch of coordinates.

        Return:
            log_psi (tensor of dtype tf.complex64): a batch of log amplitudes at x.
        """
        if self.log_psi_func is None:
            raise NotImplementedError
        return self.log_psi_func(x)

    def log_prob(self, x):
        return 2 * tf.math.real(self.log_psi(x))


class BlockAutoregressiveWavefunction(Wavefunction):
    """Neural wavefunction with a block neural autoregressive flow ansatz.

    Args:
        bosonic_dim (int): the number of bosonic coordinates.
        fermionic_dim (int): the number of fermionic coordinates.
    """

    def __init__(self, bosonic_dim=0, fermionic_dim=0, alpha=20):
        self.dim_b = bosonic_dim
        self.dim_f = fermionic_dim
        self.alpha = alpha
        super().__init__()

    @property
    def trainable_variables(self):
        var = []
        for attr in ["bosonic_network", "fermion_real", "fermion_imag"]:
            if hasattr(self, attr):
                var = var + getattr(self, attr).trainable_variables
        return var

    def _get_bosonic_autoregressive(self):
        dim = self.dim_b
        if not hasattr(self, "bosonic_network"):
            self.bosonic_network = BNAFDensityEstimator([dim, self.alpha * dim, dim])
        return self.bosonic_network

    def _get_fermionic_amplitudes(self, sample_b, sample_f):
        dim = self.dim_b
        if not hasattr(self, "fermion_real"):
            self.fermion_real = ConditionalDenseNetwork([dim, dim, 1], self.dim_f)
        if not hasattr(self, "fermion_imag"):
            self.fermion_imag = ConditionalDenseNetwork([dim, dim, 1], self.dim_f)
        return tf.complex(self.fermion_real(sample_b, sample_f), self.fermion_imag(sample_b, sample_f))

    def log_psi(self, x):
        sample_b, sample_f = x[..., :self.dim_b], x[..., self.dim_b:]
        # bosonic part
        dist_b = self._get_bosonic_autoregressive()
        log_prob_b = tf.squeeze(dist_b.log_prob(tf.expand_dims(sample_b, axis=-2)), axis=-1)
        assert log_prob_b.shape == x.shape[:-1], f"{log_prob_b.shape} != {x.shape}[:-1]"
        # fermionic part
        if self.dim_f > 0:
            psi_f = tf.squeeze(self._get_fermionic_amplitudes(sample_b, sample_f), axis=-1)
            assert psi_f.shape == x.shape[:-1], f"{psi_f.shape} != {x.shape}[:-1]"
        else:
            # no fermions
            psi_f = tf.cast(1.0, tf.complex64)
        return tf.cast(log_prob_b / 2, tf.complex64) + tf.math.log(psi_f)

