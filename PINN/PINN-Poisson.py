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

print(samples_bulk.shape, samples_boundary.shape, left_boundary.shape)

# %% [markdown]
# ### Relaxation method
#
# The Laplace equation:
# $$
# \Delta u(x,y)=0\,, \qquad (x,y)\in[0,1]\times[0,1]
# $$
#
# $$
# u(1,y)=1\,,
# $$
# $$
# u(x,1)=1\,,
# $$
# $$
# u(x,0)=x\,,
# $$
# $$
# u(0,y)=y\,.
# $$

# %%
N = 256  #

u = rng.random(size=(N+2,N+2))

# upper

u[0,:] = 1.
u[:,-1] = 1.

u[-1,:] = np.linspace(0, 1, N+2)
u[:,0] = np.linspace(0, 1, N+2)[::-1]

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

ax.imshow(u)


ax.set_xlabel('$x$')
ax.set_ylabel('$y$')


# %%
num_steps = 50000

u_approx = u.copy()
errors = []

for i in range(num_steps):

    u_approx[1:-1,1:-1] = 1./4 * (u[1:-1,2:] + u[1:-1,:-2] + u[2:,1:-1] + u[:-2,1:-1])

    err = np.sqrt(np.mean((u_approx-u)**2))
    errors.append(err)

    u = u_approx.copy()


# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

ax.imshow(u)
CS = ax.contour(u, levels=np.linspace(0.1,0.9,9), colors='white')
ax.clabel(CS, fontsize=10)


ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.semilogy(1 + np.arange(num_steps), errors)


ax.set_xlabel('step')
ax.set_ylabel('MSE')


# %% [markdown]
# ### PINN

# %% [markdown]
# #### Data Generator

# %%
class Params:

    SEED = 13

class RandomGenerator(keras.utils.Sequence):

    def __init__(self, bulk_batch_size, boundary_batch_size, steps_per_epoch, seed=False, **kwargs):
        super(RandomGenerator, self).__init__(**kwargs)
        self.bulk_batch_size = bulk_batch_size
        self.boundary_batch_size = boundary_batch_size
        self.steps_per_epoch = steps_per_epoch

        if seed:
            self.rng = np.random.default_rng(seed=Params.SEED)
        else:
            self.rng = np.random.default_rng()


    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):

        samples_bulk = self.rng.random(size=(self.bulk_batch_size**2,2))
        samples_boundary = self.rng.random(size=(4*self.boundary_batch_size,1))

        arr1,arr2,arr3,arr4 = np.split(samples_boundary, indices_or_sections=4)

        left_boundary = np.concatenate([np.full_like(arr1, 0.), arr1], axis=1)
        right_boundary = np.concatenate([np.full_like(arr2, 1.), arr2], axis=1)
        top_boundary = np.concatenate([arr3, np.full_like(arr3, 0.)], axis=1)
        bottom_boundary = np.concatenate([arr4, np.full_like(arr4, 1.)], axis=1)

        u_left = 1-arr1
        u_right = tf.ones(shape=tf.shape(arr2))
        u_top = tf.ones(shape=tf.shape(arr3))
        u_bottom = arr4

        return tf.convert_to_tensor(samples_bulk),\
                    tf.concat([bottom_boundary,right_boundary,top_boundary,left_boundary], axis=0),\
                        tf.concat([u_bottom,u_right,u_top,u_left], axis=0)
        

    def on_epoch_end(self):
        pass


# %%
class PoissonPINN(keras.Model):
    def __init__(self, hidden_units, **kwargs):
        super(PoissonPINN, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.first_layer = layers.Dense(self.hidden_units[0], activation='tanh', **kwargs)
        self._layers = [layers.Dense(units, activation='tanh', **kwargs) for units in self.hidden_units[1:]]
        self._layers.append(layers.Dense(1, **kwargs))

    def call(self, input_data, training=False):

        if training:
            bulk, boundary = input_data
            
            u_boundary = self.first_layer(boundary, training=training)
            for layer in self._layers:
                u_boundary = layer(u_boundary, training=training)

            x,y = tf.unstack(bulk, axis=-1)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)
                tape2.watch(y)
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(x)
                    tape1.watch(y)
                    u_bulk = self.first_layer(tf.stack([x,y], axis=-1), training=training)
                    for layer in self._layers:
                        u_bulk = layer(u_bulk, training=training)
                u_x,u_y = tape1.gradient(u_bulk, [x,y])
            
            u_xx = tape2.gradient(u_x, x)
            u_yy = tape2.gradient(u_y, y)

            laplacian = u_xx + u_yy

            del tape1,tape2

            return laplacian[...,tf.newaxis], u_boundary

        u = self.first_layer(input_data, training=training)
        for layer in self._layers:
            u = layer(u, training=training)

        return u

    @tf.function
    def train_step(self, batch):
        bulk, boundary, u_boundary_true = batch

        with tf.GradientTape() as tape:
            laplacian, u_boundary_pred = self((bulk, boundary), training=True)
            train_loss = tf.reduce_mean((laplacian)**2) + tf.reduce_mean((u_boundary_pred-u_boundary_true)**2)

        gradients = tape.gradient(train_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return train_loss

    @tf.function
    def test_step(self, batch):
        bulk, boundary, u_boundary_true = batch

        laplacian, u_boundary_pred = self((bulk, boundary), training=True)
        test_loss = tf.reduce_mean((laplacian)**2) + tf.reduce_mean((u_boundary_pred-u_boundary_true)**2)

        return test_loss

                

# %%
from keras.optimizers.schedules import ExponentialDecay

bulk_batch_size = 128
boundary_batch_size = 1024
steps_per_epoch = 200
num_epochs = 10
hidden_units = [50,50,50,50,50]

train_generator = RandomGenerator(bulk_batch_size, boundary_batch_size, steps_per_epoch)
val_generator = RandomGenerator(bulk_batch_size, boundary_batch_size, int(steps_per_epoch/10))
model = PoissonPINN(hidden_units)

# Define the learning rate schedule
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=1/1.5,
    staircase=True
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))


for epoch in range(num_epochs):
    
    train_loss = 0.
    val_loss = 0.

    tepoch = tqdm(
            train_generator,
            desc=f"Epoch {epoch+1}/{num_epochs}"
    )
        
    for batch_idx,data_batch in enumerate(tepoch):

        if batch_idx == len(train_generator):
            break

        loss = model.train_step(data_batch)
        train_loss += loss.numpy()
        tepoch.set_postfix_str("batch={:d}, train loss={:.4f}".format(batch_idx+1, train_loss/(batch_idx+1)))


    for batch_idx,val_batch in enumerate(val_generator):

        if batch_idx == len(val_generator):
            break

        loss = model.test_step(val_batch)
        val_loss += loss.numpy()
    val_loss /= batch_idx + 1
    print("val loss={:.4f}".format(val_loss))

    train_generator.on_epoch_end()
    val_generator.on_epoch_end()



# %%
M = 256

xx_1D = np.linspace(0, 1, M+2)
yy_1D = np.linspace(0, 1, M+2)

xx,yy = np.meshgrid(xx_1D,yy_1D)

input_data = tf.stack([xx.flatten(), yy.flatten()], axis=-1)

u_pred = tf.squeeze(model(input_data, training=False)).numpy().reshape(M+2,M+2)

u_pred[0,:] = 1.
u_pred[:,-1] = 1.

u_pred[-1,:] = xx_1D.copy()
u_pred[:,0] = yy_1D.copy()[::-1]

# %%
num_steps = 10000

u_new = u_pred.copy()
errors_new = []

for i in range(num_steps):

    u_new[1:-1,1:-1] = 1./4 * (u_pred[1:-1,2:] + u_pred[1:-1,:-2] + u_pred[2:,1:-1] + u_pred[:-2,1:-1])

    err = np.sqrt(np.mean((u_new-u_pred)**2))
    errors_new.append(err)

    u_pred = u_new.copy()

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

ax.imshow(u_pred)
CS = ax.contour(u_pred, levels=np.linspace(0.1,0.9,9), colors='white')
ax.clabel(CS, fontsize=10)


ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.semilogy(1 + np.arange(num_steps), errors_new)


ax.set_xlabel('step')
ax.set_ylabel('MSE')

# %%
