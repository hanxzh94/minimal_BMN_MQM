# Minimal BMN Matrix Quantum Mechanics
Variational Quantum Monte Carlo with Block Autoregressive Flow Ansatz for Minimal BMN Matrix Quantum Mechanics. 

The code is tested on Python 3.7.8, TensorFlow 2.3.1 with TensorFlow Probability 0.11.1. 

Run `python demo.py` to start training a variational wavefunction for the minimal BMN model. 
Change parameters `N`, `l` in `demo.py` for different matrix sizes and couplings, `alpha` for different hidden-to-visible ratios in block autoregressive flows.

Run `python bosonic.py` to start training a variational wavefunction for the bosonic part of the minimal BMN model. Same parameters apply.

Please see `arXiv:2108.02942` for more details. Data in the deep learning section can be reproduced with this code.
