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


import tensorflow as tf 
from tensorflow import keras
from keras import layers

from tqdm.notebook import tqdm
import time as tm


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.run_functions_eagerly(False)


MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcdefaults()

# plt.rc('font',**{'family':'serif','serif':['Times']})
# plt.rc('text', usetex=True)

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# %matplotlib inline

# %%
rng = np.random.default_rng()

# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)


# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')

# %% [markdown]
# ### Sobol sequences

# %%
m = 5 # 2^m points in each dimension

sampler_bulk = scipy.stats.qmc.Sobol(d=2, scramble=True)
sampler_boundary = scipy.stats.qmc.Sobol(d=1, scramble=True)

samples_bulk = sampler_bulk.random_base2(m=2*m)
samples_boundary = sampler_boundary.random_base2(m=m+2)

arr1,arr2,arr3,arr4 = np.split(samples_boundary, indices_or_sections=4)

left_boundary = np.concatenate([np.full_like(arr1, 0.), arr1], axis=1)
right_boundary = np.concatenate([np.full_like(arr2, 1.), arr2], axis=1)
up_boundary = np.concatenate([arr3, np.full_like(arr3, 0.)], axis=1)
down_boundary = np.concatenate([arr4, np.full_like(arr4, 1.)], axis=1)


fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

ax.scatter(*samples_bulk.T, s=5)

for arr in [left_boundary,right_boundary,up_boundary,down_boundary]:
    ax.scatter(*arr.T, s=10, color='green')

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)


ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

# Sample data
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 0.5, 1.5, 1])

# Create Akima spline
akima = Akima1DInterpolator(x, y)

# Interpolate at new points
x_new = np.linspace(0, 4, 100)
y_new = akima(x_new)

# Plot
plt.plot(x, y, 'o', label='Data Points')
plt.plot(x_new, y_new, '-', label='Akima Spline')
plt.legend()
plt.show()


# %%
Msun = 5e-6  # s

Mbh = 1e+9

AU = 500.
pc = 206000.

Mbh*Msun / AU / pc

# %%
4.4*(1e+3/Mbh) / 10e-9 * (Mbh*Msun / AU / pc)

# %%
Mbh*100*Msun / AU / pc

# %%
2/7/86400

# %%
25400/42373

# %%
0.599438 * 3200

# %%
0.599438 * 2700

# %%
16920+53

# %%
42373-16973

# %%
25400-1618-1918

# %%
21864*0.0320

# %%
700 + 987

# %%
1736-1687

# %%
16920 / 42373

# %%
0.3993 * 1255

# %%
