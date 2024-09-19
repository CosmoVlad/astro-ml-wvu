## Training a neural network

### Plan

- auto-diff package JAX,
- gradient descent in a simple problem,
- gradient descent for training a NN,
- training a network: training, validation, and test datasets.
  
#### JAX

[JAX](https://jax.readthedocs.io/en/latest/) is a package for auto-differentiation which internally uses the usual differentiation rules (most importantly, the chain rule) to compute derivatives of almost arbitrarily complicated functions. The auto-diff can be thought of as a middle ground between symbolic and numeric differentiation.

To install:

`pip install jax`

JAX can also make use of GPUs but the CPU version is sufficient for now.