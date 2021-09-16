# Minimal BMN Matrix Quantum Mechanics
Variational Quantum Monte Carlo with Block Autoregressive Flow Ansatz for Minimal BMN Matrix Quantum Mechanics. 

Results in the _Deep Learning_ section of the publication [Rinaldi et al. (2021)](https://www.arxiv.org/abs/2108.02942) can be reproduced with this code.
Consider the citation in [Cite](#cite).

# Code

The code is tested on 
- Python 3.7.8, TensorFlow 2.3.1, TensorFlow Probability 0.11.1
- Python 3.6.9, TensorFlow 2.6.0, TensorFlow Probability 0.14.0 (on GPU from `Dockerfile`)

## Run scripts 

Run `python demo.py` to start training a variational wavefunction for the minimal BMN model. 
Change parameters `N`, `l` in `demo.py` for different matrix sizes and couplings, `alpha` for different hidden-to-visible ratios in block autoregressive flows.

Run `python bosonic.py` to start training a variational wavefunction for the bosonic part of the minimal BMN model. Same parameters apply.

## Docker

A `Dockerfile` is provided for running the code inside a container when a NVidia GPU is available.
Build the container with `docker build -t <your-image-name> .` and then run it with `docker run --rm --gpus all -it -v $PWD:/workdir -w /workdir <your-image-name>`.
This will open a terminal inside the container with a prompt inside this repository's folder (called `/workdir` inside the container).
From this terminal you can run the code as described in [Run scripts](#run-scripts).

# Cite

If you use this code (or parts of it), please consider citing our paper:
```bibtex
@misc{rinaldi2021matrixmodels,
    title   = {Matrix Model simulations using Quantum Computing, Deep Learning, and Lattice Monte Carlo}, 
    author  = {Rinaldi, Enrico and Han, Xizhi and Hassan, Mohammad and Feng, Yuan and Nori, Franco and McGuigan, Michael and Hanada, Masanori},
    year    = {2021},
    eprint  = {2108.02942},
    archivePrefix = {arXiv},
    primaryClass = {quant-ph}
}
```