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

size = 100
g = 9.8

velocities = 10*rng.uniform(size=size)        # v in [0,10]
angles = np.pi/2 * rng.uniform(size=size)     # alpha in [0,pi/2]

distances = velocities**2/g * np.sin(2*angles)

distances += rng.normal(loc=0., scale=distances/10, size=size)


# %% [markdown]
# #### Forward propagation

# %%
def layer_mapping(p_old, W, b):

    # p_old [m x N]
    # p_new [n x N]
    # b [n x 1] broadcast to [n x N]
    # W [n x m]
    # W * p_old + b

    return np.dot(W, p_old) + b   # = p_new

def ReLU(p):

    return p * (p>0).astype(float)

# input [2x1] -> hidden1 [5x1] -> output [1x1]

W1 = rng.normal(size=(5,2))
W2 = rng.normal(size=(1,5))
b1 = rng.normal(size=(5,1))
b2 = rng.normal(size=(1,1))


#vel,alpha = velocities[0],angles[0]
#p_init = np.array([[vel,alpha]]).T

p_init = np.array([velocities,angles])

print(p_init.shape)

p = layer_mapping(p_init, W1, b1)
p = ReLU(p)
print(p.shape)

p = layer_mapping(p, W2, b2)
p = ReLU(p)
print(p.shape)

err = np.sum((p - distances)**2) / size


# %% [markdown]
# #### Backpropagation

# %%
# err = err(W1,W2,b1,b2)

# grad_err
# W1 = W1 - step*grad_err(W1)
# W2 = W2 - step*grad_err(W2)
# b1 = b1 - step*grad_err(b1)
# b2 = b2 - step*grad_err(b2)
