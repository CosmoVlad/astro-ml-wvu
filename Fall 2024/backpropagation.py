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
# Here we implement backpropagation for a simple network which predicts the range of motion for a projectile launched at an angle $\alpha$ with an initial velocity $v_0$.
#
# - refactor the forward propagation as two functions:
#   - function `forward_prop` takes the input parameter vector $\mathbf{p}$ and network parameters $\left(W_1,W_2,\mathbf{b}_1,\mathbf{b}_2\right)$ (a.k.a. weights) and returns a prediction for $d$;
#   - function `error_func` takes the prediction for $d$ and the weights $\left(W_1,W_2,\mathbf{b}_1,\mathbf{b}_2\right)$ and returns the mean squared error (MSE) between the prediction and the ground truth.
# - use the JAX function `grad` to calculate the gradient of `error_func` w.r.t. the weights,
# - use the GD method to minimize the MSE between predictions and the ground truth (a.k.a. training the NN).
#
# <ins>Follow-up</ins>: The GD below does not seem to result in optimal weights. Debug the pipeline.

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

# %% [markdown]
# #### Generating mock data
#
# $$
# d = \frac{v_0^2\sin{2\alpha}}{g}\,.
# $$
#
# - $v_0$ -- m/s;
# - $\alpha$ -- radian;

# %%
rng = np.random.default_rng()

size = 1000
g = 9.8

velocities = 10*rng.uniform(size=size)        # v in [0,10]
angles = np.pi/2 * rng.uniform(size=size)     # alpha in [0,pi/2]

distances = velocities**2/g * np.sin(2*angles)

distances += rng.normal(loc=0., scale=distances/10, size=size)

# %% [markdown]
# #### Forward propagation

# %%
import jax.numpy as jnp
from jax import grad, random

from jax import config; config.update("jax_enable_x64", True)

def layer_mapping(p_old, W, b):

    # p_old [m x N]
    # p_new [n x N]
    # b [n x 1] broadcast to [n x N]
    # W [n x m]
    # W * p_old + b

    return jnp.dot(W, p_old) + b   # = p_new

def ReLU(p):

    return p * (p>0).astype(float)

def sigmoid(p):

    return 1 / (1 + jnp.exp(-p))

# input [2x1] -> hidden1 [5x1] -> output [1x1]


p_init = jnp.array([velocities,angles])

def forward_prop(p, W1, W2, b1, b2):

    p = layer_mapping(p, W1, b1)
    p = ReLU(p)
    p = layer_mapping(p, W2, b2)
    p = ReLU(p)

    return p

def error_func(W1, W2, b1, b2, p_input, p_data):

    p = forward_prop(p_input, W1, W2, b1, b2)
    
    return jnp.sum((p - p_data)**2) / size


# %% [markdown]
# #### Backpropagation

# %%
# err = err(W1,W2,b1,b2)

# grad_err
# W1 = W1 - step*grad_err(W1)
# W2 = W2 - step*grad_err(W2)
# b1 = b1 - step*grad_err(b1)
# b2 = b2 - step*grad_err(b2)

# %%
grad_err = grad(error_func, argnums=[0,1,2,3])

step = 0.1
n_steps = 2000

seed = 1701

key = random.key(seed)
W1 = random.normal(key, shape=(5,2))

key,_ = random.split(key)
W2 = random.normal(key, shape=(1,5))

key,_ = random.split(key)
b1 = random.normal(key, shape=(5,1))

key,_ = random.split(key)
b2 = random.normal(key, shape=(1,1))

error_values = [error_func(W1,W2,b1,b2,p_init,distances)]

for i in range(n_steps):

    grad_W1,grad_W2,grad_b1,grad_b2 = grad_err(W1,W2,b1,b2,p_init,distances)

    W1 -= step * grad_W1
    W2 -= step * grad_W2
    b1 -= step * grad_b1
    b2 -= step * grad_b2

    error_values.append(error_func(W1,W2,b1,b2,p_init,distances))

error_values = np.array(error_values)

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(np.arange(n_steps+1),error_values)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('step')
ax.set_ylabel('mean_sq_err')

# %%
key,_ = random.split(key)
test_velocities = 10*random.uniform(key, shape=(size,))        # v in [0,10]

key,_ = random.split(key)
test_angles = jnp.pi/2 * random.uniform(key, shape=(size,))     # alpha in [0,pi/2]

test_distances = test_velocities**2/g * jnp.sin(2*test_angles)
test_distances += test_distances/10 * random.normal(key, shape=(size,))

p_test = jnp.array([test_velocities,test_angles])

pred_distances = forward_prop(p_test, W1,W2,b1,b2)

# %%
plt.scatter(test_distances, pred_distances.reshape(size,))

# %%
