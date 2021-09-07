import tensorflow as tf
import tensorflow_probability as tfp
from obs import identity_operator
from datetime import datetime
import pickle
import numpy as np


def save_model(wavefunc, filename, metadata=[]):
    # dump all variables in wavefunc into the file along with some metadata
    with open(filename, 'wb+') as file:
        pickle.dump([v.numpy() for v in wavefunc.trainable_variables], file)
        pickle.dump(metadata, file)
    # print(f"Model saved to {filename}")

def load_model(wavefunc, filename):
    # recover the wavefunc from the file with some metadata
    with open(filename, 'rb') as file:
        variables = pickle.load(file)
        meta = pickle.load(file)
    for a, b in zip(wavefunc.trainable_variables, variables):
        a.assign(b)
    # print(f"Model loaded from {filename} with metadata {meta}")
    return meta


def replace_nan(grad, val):
    if grad is None:
        return grad
    return tf.where(tf.math.is_nan(grad), val * tf.ones_like(grad), grad)

@tf.function
def expectation_value(obs, wavefunc, sampler, batch_size, normalized=True):
    with tf.GradientTape() as tape:
        loss = obs.evaluate(wavefunc, sampler, batch_size, normalized)
    return loss

def evaluate(wavefunc, sampler, obs, num_samples, normalized=True, filename="", batch_size=1000):
    """Evaluate the expectation value of obs in wavefunc, with num_samples samples generated
    from the sampler. If filename is not empty, load the wavefunc and the sampler from file
    first.
    """ 
    if filename:
        load_model(wavefunc, filename + "_wf")
        load_model(sampler, filename + "_sm")
    vals = []
    ones = []
    eye = identity_operator()
    for _ in range(num_samples // batch_size):
        val = expectation_value(obs, wavefunc, sampler, batch_size, normalized)
        if not np.isnan(val):
            vals.append(val)
        if not normalized:
            ones.append(expectation_value(eye, wavefunc, sampler, batch_size, normalized))
    if normalized:
        # return the mean and its uncertainty
        m, s = np.mean(vals), np.std(vals) / np.sqrt(len(vals))
        print(f"Value: {m:.8f} +- {s:.8f}")
        return m, s
    else:
        m1, s1 = np.mean(vals), np.std(vals) / np.sqrt(len(vals))
        m2, s2 = np.mean(ones), np.std(ones) / np.sqrt(len(ones))
        print(f"Value: {m1:.8f} +- {s1:.8f} / {m2:.8f} +- {s2:.8f} = {m1/m2:.8f}")
        return m1, s1

@tf.function
def gauge_expectation_value(obs, wavefunc, sampler, algebra, group, batch_size, normalized=True):
    with tf.GradientTape() as tape:
        loss = obs.gauge_evaluate(wavefunc, sampler, algebra, group, batch_size, normalized)
    return loss

def gauge_evaluate(wavefunc, sampler, obs, num_samples, algebra, group, normalized=True, filename=""):
    """Evaluate the expectation value of obs in wavefunc averaged over gauge orbits, with 
    num_samples samples generated from the sampler. If filename is not empty, load the 
    wavefunc and the sampler from file first.
    """ 
    if filename:
        load_model(wavefunc, filename + "_wf")
        load_model(sampler, filename + "_sm")
    # divide into batches of size 1000
    batch_size = 1000
    vals = []
    ones = []
    eye = identity_operator()
    for _ in range(num_samples // batch_size):
        val = gauge_expectation_value(obs, wavefunc, sampler, algebra, group, 1000, normalized)
        if not np.isnan(val):
            vals.append(val)
        if not normalized:
            ones.append(expectation_value(eye, wavefunc, sampler, 1000, normalized))
    if normalized:
        # return the mean and its uncertainty
        m, s = np.mean(vals), np.std(vals) / np.sqrt(len(vals))
        print(f"Value: {m:.8f} +- {s:.8f}")
        return m, s
    else:
        m1, s1 = np.mean(vals), np.std(vals) / np.sqrt(len(vals))
        m2, s2 = np.mean(ones), np.std(ones) / np.sqrt(len(ones))
        print(f"Value: {m1:.8f} +- {s1:.8f} / {m2:.8f} +- {s2:.8f} = {m1/m2:.8f}")
        return m1, s1


@tf.function
def _minimize_step(wavefunc, hamil, sampler, optimizer, batch_size):
    """Gradient descent on the wavefunc to minimize hamil."""
    with tf.GradientTape() as tape:
        loss = tf.math.real(hamil.evaluate(wavefunc, sampler, batch_size))
    variables = wavefunc.trainable_variables
    gradients = tape.gradient(loss, variables)
    gradients = [replace_nan(grad, 0) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

@tf.function
def _sampler_step(wavefunc, hamil, sampler, optimizer, batch_size):
    """Gradient descent on the sampler to minimize the KL divergence between wavefunc and sampler probability distributions."""
    with tf.GradientTape() as tape:
        sample, sample_log_prob = sampler.sample(batch_size, with_log_prob=True)
        # compute the KL divergence for unnormalized wavefunc
        target_log_prob = wavefunc.log_prob(sample)
        target_log_norm = tfp.math.reduce_logmeanexp(target_log_prob - sample_log_prob)
        target_log_prob -= tf.stop_gradient(target_log_norm)
        loss = tf.math.reduce_mean(sample_log_prob - target_log_prob)
    variables = sampler.trainable_variables
    gradients = tape.gradient(loss, variables)
    gradients = [replace_nan(grad, 0) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

@tf.function
def _sampler_step_with_wavefunc(wavefunc, hamil, sampler, optimizer, batch_size):
    """Gradient descent on both the sampler and the wavefunc to minimize the KL divergence between wavefunc 
    and sampler probability distributions."""
    with tf.GradientTape() as tape:
        sample, sample_log_prob = sampler.sample(batch_size, with_log_prob=True)
        # compute the KL divergence for unnormalized wavefunc
        target_log_prob = wavefunc.log_prob(sample)
        target_log_norm = tfp.math.reduce_logmeanexp(target_log_prob - sample_log_prob)
        target_log_prob -= tf.stop_gradient(target_log_norm)
        loss = tf.math.reduce_mean(sample_log_prob - target_log_prob)
    variables = sampler.trainable_variables + wavefunc.trainable_variables
    gradients = tape.gradient(loss, variables)
    gradients = [replace_nan(grad, 0) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, variables))
    return loss

def minimize(wavefunc, hamil, obs_dict, sampler, lr, optimizer, lr_max=1e-3, lr_min=1e-5, batch_size=100, max_epochs=1000, num_iters=1000, thres=4, filename=""):
    """Minimizes hamil of the wavefunc and the KL divergence between the wavefunc and the sampler.

    The algorithm starts with lr = lr_max, and lr -> lr / 2 after 10 times no loss improvement until lr < lr_min. In each epoch, the optimizer takes num_iters iterations, 
    where in each iteration hamil is evaluated on batch_size samples. Optimizer minimizes hamil for wavefunc and KL divergence between wavefunc and sampler for the sampler.
    In case that the KL divergence is larger than thres, and hence the evaluation might be unstable, both wavefunc and sampler try to minimize the KL divergence. When
    the epoch is done, observables in obs_dict are printed and variables are saved if filename is not empty.

    Args:
        wavefunc (Wavefunction): the variational wavefunction
        hamil (Observable): the observable to minimize
        obs_dict (dict from strings to Observables): the observables to monitor
        sampler (Sampler): the Monte Carlo sampler
        lr (scalar): the current learning rate
        optimizer (Optimizer): the optimizer to use
        lr_max, lr_min (float): the starting and ending learning rates
        batch_size (int): the number of samples in each iteration
        max_epochs (int): the maximal number of epochs
        num_iters (int): the number of iterations in each epoch
        thres (float): the KL divergence threshold
        filename (string): name of the file to save the variables after each epoch, if not empty

    Return:
        last_loss (float): the loss of the last epoch
    """
    lr.assign(lr_max)
    last_loss = 1e9
    tot_fail = 0
    loss_sampler = 1e9
    for i in range(max_epochs):
        # reset all metrics
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        sampler_loss = tf.keras.metrics.Mean(name='sampler_loss')
        obs_val = {obs: tf.keras.metrics.Mean(name=obs) for obs in obs_dict}
        # optimizer iterations
        for j in range(num_iters):
            if loss_sampler > thres and i < 10:
                loss_sampler = _sampler_step_with_wavefunc(wavefunc, hamil, sampler, optimizer, batch_size)
            else:
                loss_sampler = _sampler_step(wavefunc, hamil, sampler, optimizer, batch_size)
            sampler_loss(loss_sampler)
            loss = _minimize_step(wavefunc, hamil, sampler, optimizer, batch_size)
            train_loss(loss)
            if j % 10 == 0:
                # evaluate obs_dict every 10 iters
                for obs in obs_val:
                    obs_val[obs](expectation_value(obs_dict[obs], wavefunc, sampler, batch_size))
        # end of epoch
        loss_mean = train_loss.result()
        sampler_loss_mean = sampler_loss.result()
        # print observables
        time = datetime.now().strftime("%H:%M:%S.%f")
        obs_str = " ".join([f"{obs} = {obs_val[obs].result():+.8f}" for obs in obs_dict])
        print(time + f": Epoch {i:03}: loss = {loss_mean:+.8f} lr = {lr.numpy():.8f} DKL = {sampler_loss_mean:+.8f} " + obs_str)
        # save variables
        if filename:
            save_model(wavefunc, filename + "_wf")
            save_model(sampler, filename + "_sm")
        # adjust lr
        if loss_mean > last_loss:
            tot_fail += 1
            if tot_fail == 10:
                tot_fail = 0
                lr.assign(lr / 2)
        last_loss = loss_mean
        if lr < lr_min:
            # optimization finished
            break
    return last_loss

