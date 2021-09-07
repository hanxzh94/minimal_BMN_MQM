import tensorflow as tf
import numpy as np
import random


class LieAlgebra():
    """Base class for a lie algebra module, including methods necessary to compute casimirs 
    (see obs.py for more info)."""

    def __init__(self, basis):
        """Initializes the algebra given a matrix representation.

        Arguments:
            basis (list of numpy arrays): matrix representation of a basis of the algebra,
                with each element in the list an N-by-N hermitian numpy matrix for a basis element
        """
        self.N = basis[0].shape[0]
        self.dim = len(basis)
        self.basis = tf.constant(np.array(basis), dtype=tf.complex64)
        metric = np.zeros((self.dim, self.dim), dtype=complex)
        for i, s in enumerate(basis):
            assert np.allclose(s, np.transpose(s.conj())), "matrices should be hermitian"
            for j, t in enumerate(basis):
                metric[i, j] = np.trace(np.matmul(s, t))
        self.metric = tf.constant(metric, dtype=tf.complex64)

    def infinitesimal_action(self, dg, x):
        """Acts the lie algebra element on the input. Note dg x = d exp (i s dg) x / ds (s = 0).

        Arguments:
            dg (tensor of shape (batch_size, dim)): the lie algebra elements (written in self.basis)
            x (tensor of shape (batch_size, ...)): the inputs

        Returns:
            dg x (tensor of shape (batch_size, ...)): the inputs after action
        """
        raise NotImplementedError

    def random_algebra_element(self, batch_size):
        """Generates a random lie algebra element with norm == sqrt(dim).

        Arguments:
            batch_size (int): the number of samples to generate

        Returns:
            dg (tensor of shape (batch_size, dim)): random elements generated
        """
        # note this implementation assumes that self.metric is proportional to identity
        dg = tf.random.normal([batch_size, self.dim])
        return tf.stop_gradient(tf.sqrt(self.dim * 1.0) * dg / tf.norm(dg, axis=-1, keepdims=True))

    def vector_to_matrix(self, vec):
        """Converts from vector representations of lie algebra elements to matrices.

        Arguments:
            vec (tensor of shape (..., dim)): coefficients on the basis

        Returns:
            mat (tensor of shape (..., N, N)): matrix representations
        """
        assert vec.shape[-1] == self.dim, "dimension mismatch"
        vec = tf.cast(vec, tf.complex64)
        mat = tf.tensordot(vec, self.basis, axes=[[-1], [0]])
        return mat

    def matrix_to_vector(self, mat):
        """Converts from matrix representations of lie algebra elements to vectors.

        Arguments:
            mat (tensor of shape (..., N, N)): matrix representations
            
        Returns:
            vec (tensor of shape (..., dim)): coefficients on the basis
        """
        assert mat.shape[-1] == mat.shape[-2] == self.N, "dimension mismatch"
        mat = tf.cast(mat, tf.complex64)
        vec = tf.tensordot(mat, self.basis, axes=[[-1, -2], [1, 2]])
        vec = tf.tensordot(vec, tf.linalg.inv(self.metric), axes=[[-1], [0]])
        return vec


class Group():
    """Base class for the gauge groups, including methods indispensable for the algorithm 
    (used in the Wavefunction class)."""

    def action(self, g, x):
        """Returns g x for a batch of gauge group elements g and input coordinates x.

        Arguments:
            g (tensor of shape (batch_size, ...)): a batch of group elements
            x (tensor of shape (batch_size, ...)): a batch of inputs

        Returns:
            g x (tensor of same shape as x): for each sample in the batch, 
                the result of input coordinates acted by g from left
        """
        raise NotImplementedError

    def random_element(self, batch_size):
        """Generates a batch of random group elements.

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            g (tensor of shape (batch_size, ...)): the random elements generated
        """
        raise NotImplementedError


class Trivial(Group):
    """Class for the trivial group."""

    def action(self, g, x):
        assert g is None, "unrecognized group element"
        return x

    def random_element(self, batch_size):
        return None


class SU(Group, LieAlgebra):
    """Class for the SU(N) gauge group."""

    def __init__(self, N, pool_size=10000):
        basis = []
        # ordering of the basis here is important for gauge fixing
        # see the implementation of Vectorizer and gauge_fixing method for more info
        # N*(N-1)/2 nondiagonal elements
        for i in range(1, N):
            for j in range(0, N - i):
                m = np.zeros((N, N), dtype=complex)
                m[j, j + i] = 1 / np.sqrt(2)
                m[j + i, j] = 1 / np.sqrt(2)
                basis.append(m)
            for j in range(0, N - i):
                m = np.zeros((N, N), dtype=complex)
                m[j, j + i] = 1j / np.sqrt(2)
                m[j + i, j] = -1j / np.sqrt(2)
                basis.append(m)
        # Cartan subalgebra of dimension (N-1)
        for i in range(1, N):
            m = np.zeros((N, N), dtype=complex)
            for j in range(i):
                m[j, j] = 1 / np.sqrt(i * i + i)
            m[i, i] = -i / np.sqrt(i * i + i)
            basis.append(m)
        super().__init__(basis)

        self.pool_size = pool_size
        self.pool = self._random_element(pool_size) # creates a pool of random elements

    def action(self, g, x):
        """Returns g x inv(g) for a batch of unitary matrices g and input matrices x.

        Arguments:
            g (tensor of shape (batch_size, N, N)): a batch of unitary matrices
            x (tensor of shape (batch_size, ..., N, N)): a batch of matrices

        Returns:
            g x inv(g) (tensor of same shape as x): for each sample in the batch, 
                the result of input matrices conjugated by g
        """
        assert g.shape[0] == x.shape[0], "dimension mismatch"
        batch_size = g.shape[0]
        assert g.shape[-1] == g.shape[-2] == x.shape[-1] == x.shape[-2] == self.N, \
                "dimension mismatch"
        x_flat = tf.reshape(x, [batch_size, -1, self.N, self.N])
        y_flat = tf.einsum("bij,btjk,bkl->btil", g, x_flat, tf.linalg.adjoint(g))
        return tf.reshape(y_flat, x.shape)

    def infinitesimal_action(self, dg, x):
        """Returns i [dg, x] for a batch of hermitian matrices dg and input matrices x.

        Arguments:
            dg (tensor of shape (batch_size, dim)): a batch of lie algebra elements
            x (tensor of shape (batch_size, ..., N, N)): a batch of matrices

        Returns:
            i [dg, x] (tensor of same shape as x): for each sample in the batch, 
                the result of input matrices taking commutators with dg
        """
        assert dg.shape[0] == x.shape[0], "dimension mismatch"
        batch_size = dg.shape[0]
        assert dg.shape[-1] == self.dim and x.shape[-1] == x.shape[-2] == self.N, \
                "dimension mismatch"
        x_flat = tf.reshape(x, [batch_size, -1, self.N, self.N])
        dg_mat = self.vector_to_matrix(dg)
        y_flat = tf.einsum("bij,btjk->btik", dg_mat, x_flat) - tf.einsum("bjk,btij->btik", dg_mat, x_flat)
        return tf.reshape(1j * y_flat, x.shape)

    def _random_element(self, batch_size):
        """Generates a batch of random Haar unitaries in SU(N). 

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            g (numpy array of shape (batch_size, N, N)): the random unitaries generated
        """
        N = self.N
        random_mat = np.random.normal(size=[batch_size, N, N]) + 1j * np.random.normal(size=[batch_size, N, N])
        q = np.array([np.linalg.qr(random_mat[i])[0] for i in range(batch_size)])
        return q / np.float_power(np.linalg.det(q)[:, np.newaxis, np.newaxis], 1 / N)

    def random_element(self, batch_size):
        """Gets a batch of random Haar unitaries in SU(N) from the pool.

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            g (tensor of shape (batch_size, N, N)): the random unitaries generated
        """
        idx = random.randint(0, self.pool_size - batch_size)
        return tf.constant(self.pool[idx : idx + batch_size], dtype=tf.complex64)


class U(Group, LieAlgebra):
    """Class for the U(N) algebra."""

    def __init__(self, N):
        basis = []
        # Cartan subalgebra of dimension N
        for i in range(N):
            m = np.zeros((N, N), dtype=complex)
            m[i, i] = 1
            basis.append(m)
        # other N*(N-1) elements
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                m = np.zeros((N, N), dtype=complex)
                m[i, j] = 1 / np.sqrt(2)
                m[j, i] = 1 / np.sqrt(2)
                basis.append(m)
                m = np.zeros((N, N), dtype=complex)
                m[i, j] = -1j / np.sqrt(2)
                m[j, i] = 1j / np.sqrt(2)
                basis.append(m)
        super().__init__(basis)

    def action(self, g, x):
        """Returns g x inv(g) for a batch of unitary matrices g and input matrices x.

        Arguments:
            g (tensor of shape (batch_size, N, N)): a batch of unitary matrices
            x (tensor of shape (batch_size, ..., N, N)): a batch of matrices

        Returns:
            g x inv(g) (tensor of same shape as x): for each sample in the batch, 
                the result of input matrices conjugated by g
        """
        assert g.shape[0] == x.shape[0], "dimension mismatch"
        batch_size = g.shape[0]
        assert g.shape[-1] == g.shape[-2] == x.shape[-1] == x.shape[-2] == self.N, \
                "dimension mismatch"
        x_flat = tf.reshape(x, [batch_size, -1, self.N, self.N])
        y_flat = tf.einsum("bij,btjk,bkl->btil", g, x_flat, tf.linalg.adjoint(g))
        return tf.reshape(y_flat, x.shape)

    def infinitesimal_action(self, dg, x):
        """Returns i [dg, x] for a batch of hermitian matrices dg and input matrices x.

        Arguments:
            dg (tensor of shape (batch_size, dim)): a batch of lie algebra elements
            x (tensor of shape (batch_size, ..., N, N)): a batch of matrices

        Returns:
            i [dg, x] (tensor of same shape as x): for each sample in the batch, 
                the result of input matrices taking commutators with dg
        """
        assert dg.shape[0] == x.shape[0], "dimension mismatch"
        batch_size = dg.shape[0]
        assert dg.shape[-1] == self.dim and x.shape[-1] == x.shape[-2] == self.N, \
                "dimension mismatch"
        x_flat = tf.reshape(x, [batch_size, -1, self.N, self.N])
        dg_mat = self.vector_to_matrix(dg)
        y_flat = tf.einsum("bij,btjk->btik", dg_mat, x_flat) - tf.einsum("bjk,btij->btik", dg_mat, x_flat)
        return tf.reshape(1j * y_flat, x.shape)

    def _random_element(self, batch_size):
        """Generates a batch of random Haar unitaries in U(N). 

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            g (numpy array of shape (batch_size, N, N)): the random unitaries generated
        """
        N = self.N
        random_mat = np.random.normal(size=[batch_size, N, N]) + 1j * np.random.normal(size=[batch_size, N, N])
        q = np.array([np.linalg.qr(random_mat[i])[0] for i in range(batch_size)])
        return q

    def random_element(self, batch_size):
        """Gets a batch of random Haar unitaries in U(N) from the pool.

        Arguments:
            batch_size (int): number of samples to generate

        Returns:
            g (tensor of shape (batch_size, N, N)): the random unitaries generated
        """
        idx = random.randint(0, self.pool_size - batch_size)
        return tf.constant(self.pool[idx : idx + batch_size], dtype=tf.complex64)


class SO2(LieAlgebra):
    """Class for the so(2) representation."""

    def __init__(self):
        S = np.array([[0,  -1j], [1j,  0]])
        super().__init__([S])

    def infinitesimal_action(self, dg, x):
        if x.shape[1] == 2:
            # bosonic
            return tf.stack([x[:, 1, :, :], -x[:, 0, :, :]], axis=1)
        if x.shape[1] == 3:
            # with fermions
            return tf.stack([x[:, 1, :, :], -x[:, 0, :, :], 0.5j * x[:, 2, :, :]], axis=1)
        raise NotImplementedError


class MatrixModel(LieAlgebra):
    """Class for vectorization of multiple matrices."""

    def __init__(self, algebra, num_bosonic_matrices, num_fermionic_matrices=0):
        self.alg = algebra
        self.num_b = num_bosonic_matrices
        self.num_f = num_fermionic_matrices
        self.num = self.num_b + self.num_f
        self.dim_b = self.num_b * self.alg.dim
        self.dim_f = self.num_f * self.alg.dim
        self.dim = self.num * self.alg.dim

    def infinitesimal_action(self, dg, x):
        """Translation by dg of bosonic matrices.

        Arguments:
            dg (tensor of shape (batch_size, dim)): the lie algebra elements
            x (tensor of shape (batch_size, num, N, N): the input matrices

        Returns:
            dg (tensor of shape (batch_size, ...)): the translation
        """
        if int(dg.shape[-1]) != self.dim or x.shape[1:].as_list() != [self.num, self.alg.N, self.alg.N]:
            raise ValueError(f"invalid shape dg {dg.shape} x {x.shape}")
        return self.alg.vector_to_matrix(tf.reshape(dg, dg.shape[:-1] + [self.num, self.alg.dim]))

    def random_algebra_element(self, batch_size):
        """Generates a random lie algebra element with norm == sqrt(dim) for each bosonic matrix.

        Arguments:
            batch_size (int): the number of samples to generate

        Returns:
            dg (tensor of shape (batch_size, num * dim)): random elements generated
        """
        return tf.concat(
            [self.alg.random_algebra_element(batch_size) for _ in range(self.num_b)] 
            + [tf.zeros((batch_size, self.num_f * self.alg.dim))], axis=-1)

    def vector_to_matrix(self, vec):
        """Converts from vector representations to matrices.

        Arguments:
            vec (tensor of shape (..., dim)): coefficients on the basis

        Returns:
            mat (tensor of shape (..., num, N, N)): matrix representations
        """
        num, dim = self.num, self.alg.dim
        if vec.shape[-1] != num * dim:
            raise ValueError(f"dimension mismatch {vec.shape}[-1] != {num * dim}")
        mat = self.alg.vector_to_matrix(tf.reshape(vec, vec.shape[:-1] + [num, dim]))
        return mat

    def matrix_to_vector(self, mat):
        """Converts from matrix representations to vectors.

        Arguments:
            mat (tensor of shape (..., num, N, N)): matrix representations
            
        Returns:
            vec (tensor of shape (..., dim)): coefficients on the basis
        """
        num, dim, N = self.num, self.alg.dim, self.alg.N
        if mat.shape[-1] != N or mat.shape[-2] != N or mat.shape[-3] != num:
            raise ValueError(f"dimension mismatch {mat.shape}")
        vec = tf.reshape(self.alg.matrix_to_vector(mat), mat.shape[:-3] + [num * dim])
        return tf.cast(vec, tf.float32)

