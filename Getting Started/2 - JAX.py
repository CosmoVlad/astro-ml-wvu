# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# In this notebook we test the `grad` function from JAX on 
# $$
# f(\mathbf{x})=\mathbf{x}^{\rm T}\mathbf{x}\,, \quad \mathbf{x}=\left(\begin{array}{c} x_1\\x_2\\ \vdots\\x_N\end{array}\right)
# $$
#
# - implement $f(\mathbf{x})$ for an input of arbitrary length $N$,
# - check that the `grad` applied to the function gives the right answer $\nabla_\mathbf{x} f = 2\mathbf{x}$,
# - run the gradient descent (GD) algorithm for randomly initialized $\mathbf{x}_0\sim \mathcal{N}(0;1)$,
# - check that it converges to the minimum $\mathbf{x}=\mathbf{0}$.
#
# <ins>Follow-up</ins>: Experiment with the step size and number of steps of the GD.

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rcdefaults()

#plt.rc('font',**{'family':'serif','serif':['Times']})
#plt.rc('text', usetex=True)

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# %matplotlib inline

# %%
# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')

#fig.tight_layout()
#fig.savefig('test.pdf')

# %%
import jax.numpy as jnp
from jax import grad

from jax import config; config.update("jax_enable_x64", True)


# %%
def f(x):

    return jnp.sum(x**2)

grad_f = grad(f)

# %%
x = jnp.linspace(0,10,101)

x[:10], grad_f(x)[:10]

# %%
rng = np.random.default_rng()

x = rng.normal(size=101)

print(x.shape, grad_f(x).shape)

step = 0.9
n_steps = 500

x_values = [x]
func_values = [f(x)]


for i in range(n_steps):

    x = x - step * grad_f(x)

    x_values.append(x)
    func_values.append(f(x))

x_values = np.array(x_values)
func_values = np.array(func_values)

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(np.arange(n_steps+1),func_values)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('step')
ax.set_ylabel('$f(x)$')

# %%
