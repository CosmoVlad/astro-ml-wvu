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

size = 10000
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
# import os
# os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# %%
train_size = 10000

velocities_train = 10*rng.uniform(size=train_size)        # v in [0,10]
angles_train = np.pi/2 * rng.uniform(size=train_size)     # alpha in [0,pi/2]

distances_train = velocities_train**2/g * np.sin(2*angles_train)

distances_train += rng.normal(loc=0., scale=distances_train/10, size=train_size)

x_train = np.c_[
    velocities_train,angles_train
]

y_train = distances_train.reshape(train_size,-1)

#x_train.shape, type(x_train)
#y_train.shape

# %%

# architecture
model = keras.Sequential(
    [
        layers.Dense(50, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(1, activation='relu'),
    ]
)

# optimization
model.compile(optimizer='Adam', loss='MSE')

# training
history = model.fit(
    x_train, y_train, 
    batch_size=64, epochs=100, validation_split=0.2, verbose=2
)




# %%
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss)
plt.plot(val_loss)

# %%
velocities_predict = np.linspace(0,10,100)        # v in [0,10]
angles_predict = np.full_like(velocities_predict, np.pi/6)    # alpha in [0,pi/2]

distances_truth = velocities_predict**2/g * np.sin(2*angles_predict)

#distances_train += rng.normal(loc=0., scale=distances_train/10, size=train_size)

x_predict = np.c_[
    velocities_predict,angles_predict
]

distances_predict = model.predict(x_predict)

# %%
plt.plot(velocities_predict, distances_predict)
plt.plot(velocities_predict, distances_truth)

# %%
test_size = 10000

velocities_predict = vrange*rng.uniform(size=test_size)        # v in [0,10]
angles_predict = np.pi/2 * rng.uniform(size=test_size)    # alpha in [0,pi/2]

distances_truth = velocities_predict**2/g * np.sin(2*angles_predict)

#distances_train += rng.normal(loc=0., scale=distances_train/10, size=train_size)

x_predict = np.c_[
    velocities_predict,angles_predict
]

distances_predict = model.predict(x_predict)

# %%
plt.scatter(distances_predict[:,0], distances_truth, s=5)

# %%
plt.hist(distances_predict[:,0] - distances_truth, bins=20)

# %%
distances_truth.shape

# %%
