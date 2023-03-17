import tensorflow as tf
from train import minimize, evaluate, gauge_evaluate
from wavefunc import BlockAutoregressiveWavefunction
from sampler import BlockAutoregressiveSampler
from obs import casimir, minimal_BMN_energy, group_action, fermion_number
from algebra import SU, MatrixModel, SO2
import math


# problem parameters
N = 11
l = 0.2
alpha = 20
g, m = math.sqrt(l / N), 1.0
algebra = MatrixModel(SU(N), 2, 0)
wavefunc = BlockAutoregressiveWavefunction(algebra.dim_b, algebra.dim_f, alpha=alpha)
sampler = BlockAutoregressiveSampler(algebra.dim_b, algebra.dim_f)
# observables
batch_size = 512
hamil = minimal_BMN_energy(algebra, g, m, bosonic_only=True)
gauge = casimir(algebra, SU(N))
rotation = group_action(algebra, SO2(), tf.zeros((batch_size, 1)))
obs = {"H": hamil, "G": gauge, "R": rotation}
# training parameters
lr = tf.Variable(1e-3)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
c = 0.0
filename = f"results/bosonic/N={N},l={l:.4f},c={c:.2f},alpha={alpha}"
print(filename)
minimize(wavefunc, hamil + c * gauge, obs, sampler, lr, optimizer, batch_size=batch_size, filename=filename)
# evaluation
for o in obs:
    print("Evaluating " + o)
    evaluate(wavefunc, sampler, obs[o], 1_000_000, True, filename=filename)
    # evaluate(wavefunc, sampler, obs[o], 1_000_000, False, filename=filename)
    print("Gauge evaluating " + o)
    gauge_evaluate(wavefunc, sampler, obs[o], 1_000_000, algebra, SU(N), True, filename=filename)
    # gauge_evaluate(wavefunc, sampler, obs[o], 1_000_000, algebra, SU(N), False, filename=filename)
