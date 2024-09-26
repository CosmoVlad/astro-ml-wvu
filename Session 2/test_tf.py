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
# Here we use the framework `tensorflow` in an attempt to debug the pipeline we wrote from scratch in the previous two sessions. To install tensorflow (for CPU): `pip install tensorflow`.

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
# #### Mock dataset
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
vrange = 10.
alpha = np.pi/4

velocities = vrange*rng.uniform(size=size)        # v in [0,10]
angles = np.full_like(velocities, alpha)     # alpha = np.pi/4 for all

distances = velocities**2/g * np.sin(2*angles)
distances += rng.normal(loc=0., scale=distances/10, size=size)

vv = np.linspace(0,vrange,100)
dd = vv**2/g * np.sin(2*alpha)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.scatter(velocities, distances, s=5, label='data')
ax.plot(vv, dd, c='r', lw=2, label='theory')

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$v_0$')
ax.set_ylabel('$d$')

# %%
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# %%
