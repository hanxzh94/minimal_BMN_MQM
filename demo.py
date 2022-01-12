import tensorflow as tf
from train import minimize, evaluate, gauge_evaluate
from wavefunc import BlockAutoregressiveWavefunction
from sampler import BlockAutoregressiveSampler
from obs import casimir, minimal_BMN_energy, group_action, fermion_number
from algebra import SU, MatrixModel, SO2
import math


# problem parameters
N = 4
l = 0.2
alpha = 20
g, m = math.sqrt(l / N), 1.0
algebra = MatrixModel(SU(N), 2, 1) # SU(N) matrices two bosonic + one fermionic
# dim_b: number of bosonic variables: (N^2 - 1) * # bosonic matrices
# dim_f: number of fermionic variables: (N^2 - 1) * # fermionic matrices
# alpha is the hidden-to-visible ratio used in BAFs (BlockAutoregressiveFlow)
wavefunc = BlockAutoregressiveWavefunction(algebra.dim_b, algebra.dim_f, alpha=alpha)
sampler = BlockAutoregressiveSampler(algebra.dim_b, algebra.dim_f)
# observables
batch_size = 256
hamil = minimal_BMN_energy(algebra, g, m, bosonic_only=False) # set bosonic_only to True if you want the bosonic model
num = fermion_number(algebra)
gauge = casimir(algebra, SU(N))
rotation = group_action(algebra, SO2(), tf.zeros((batch_size, 1)))
obs = {"H": hamil, "G": gauge, "R": rotation, "N": num}
# training parameters
lr = tf.Variable(1e-3)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=1.0)
c = 0.0
filename = f"results/susy/N={N},l={l:.4f},c={c:.2f},alpha={alpha}"
print(filename)
minimize(wavefunc, hamil, obs, sampler, lr, optimizer, batch_size=batch_size, filename=filename)
# evaluation
for o in obs:
    print("Evaluating " + o)
    evaluate(wavefunc, sampler, obs[o], 1_000_000, True)
    evaluate(wavefunc, sampler, obs[o], 1_000_000, False)
