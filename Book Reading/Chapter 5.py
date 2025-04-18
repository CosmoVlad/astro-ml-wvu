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
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from keras.layers import Dense

MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rcdefaults()

#plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex=True)

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

# %% [markdown]
# ### Chapter 5
#
# In this chapter, F. Chollet focuses on the following concepts:
# - the manifold hypothesis which serves as an intuitive justification of why neural networks are able to generalize after training;
# - trade-off between optimization to the training data and generalization to new (test) data;
# - best practices for both optimization and generalization:
#   - evaluation methods (train, validation, and test sets);
#   - baseline models that gauge our expectations about a neural network we are trying to train;
#   - combatting overfitting.

# %% [markdown]
# ### Manifold hypothesis
#
# Neural networks are able to generalize, because data they are supposed to work with constitutes a smooth manifold embedded in a higher-dimensional space. Then, individual samples of the data are "points" on this manifold, and generalization means "interpolating" between those "points". "Interpolation" is in quotation marks, because it is generally not the conventional interpolation.
#
# *Example*. The MNIST images of handwritten digits have $28\times 28=784$ pixels. With the value of each pixel ranging between integers 0 and 255, the space of all possible images contains $256^{784}\approx 10^{1888}$ "points". Only a small subset of those are the images of handwritten digits. 
# *An NN model can be viewed as a smooth curve in that space. During optimization, the curve is being continuously deformed to pass close to the MNIST "datapoints" and interpolate between them*. If data lacks a manifold structure or the NN is not a good first approximation to the data manifold, the NN will eventually perform well on training data but will fail to generalize to validation data (overfitting).

# %%
(x_train, y_train), _ = mnist.load_data()

x_train = x_train.reshape(-1, 28*28)
x_train = x_train.astype("float32") / 255

rng = np.random.default_rng()

def get_model(*units, last=10):

    model = keras.Sequential()

    for num_units in units:
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(last))

    return model

# %% [markdown]
# #### Original data

# %%


model = get_model(8,16,8)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train, 
    batch_size=128,
    epochs=10,
    validation_split=0.2
)

# %%
val_acc = history.history['val_accuracy']
train_acc = history.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.5, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %% [markdown]
# #### Imperfect data (random pixels or zero-padded)

# %%
x_train_random = np.concatenate((x_train, rng.random(size=(len(x_train), 784))), axis=1)
x_train_zeros = np.concatenate((x_train, np.zeros((len(x_train), 784))), axis=1)

del model

model = get_model(8,16,8)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history_random = model.fit(
    x_train_random, y_train, 
    batch_size=128,
    epochs=10,
    validation_split=0.2
)

del model

model = get_model(8,16,8)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history_zeros = model.fit(
    x_train_zeros, y_train, 
    batch_size=128,
    epochs=10,
    validation_split=0.2
)

# %%
accuracy_random = history_random.history['val_accuracy']
accuracy_zeros = history_zeros.history['val_accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(accuracy_random)), accuracy_random, label='random')
ax.plot(1 + np.arange(len(accuracy_random)), accuracy_zeros, label='zeros')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.5, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %% [markdown]
# #### Nonsensical data

# %%
del model

model = get_model(128,256,128,64,32)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

rng.shuffle(y_train)

history_shuffle = model.fit(
    x_train[:1000], y_train[:1000], 
    batch_size=128,
    epochs=50,
    validation_split=0.2
)

# %%
val_acc = history_shuffle.history['val_accuracy']
train_acc = history_shuffle.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0., 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %% [markdown]
# ### Optimization and generalization

# %%
size=200

mean1 = [1.,0.]
mean2 = [0.,1.]

cov = np.array([
    [0.7**2, 0.2*0.7*0.5],
    [0.2*0.7*0.5, 0.5**2]
])

samples1 = rng.multivariate_normal(mean1, cov, size=size)
samples2 = rng.multivariate_normal(mean2, cov, size=size)

y1 = np.full(size, True).astype('int')
y2 = np.full(size, False).astype('int')

indices = np.arange(2*size)
rng.shuffle(indices)

X = np.concatenate((samples1, samples2), axis=0)[indices]
Y = np.concatenate((y1,y2), axis=0)[indices]




# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))


# ax.scatter(*samples1.T, s=10, color='blue')
# ax.scatter(*samples2.T, s=10, color='yellow')

ax.scatter(*X.T, c=Y)


ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_aspect('equal')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# %%
del model

model = get_model(256,512,1024,2048,1024,512,256,128,64,32, last=1)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history_two = model.fit(
    X, Y, 
    batch_size=128,
    epochs=50,
    validation_split=0.2
)

# %%
val_acc = history_two.history['val_accuracy']
train_acc = history_two.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.8, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %%
xx = np.linspace(-1,2,100)
yy = np.linspace(-1,2,100)

xx,yy = np.meshgrid(xx,yy)

zz = model.predict(np.array([xx.flatten(),yy.flatten()]).T).reshape(100,100)

subset = rng.choice(indices, size=400, replace=False)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

CS = ax.contour(xx, yy, zz, colors='black', levels=[0.], linestyles='dashed')

ax.scatter(*X[subset].T, c=Y[subset])



ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_aspect('equal')

ax.set_xlim(-1,2)
ax.set_ylim(-1,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

fig.tight_layout()

# %% [markdown]
# ### Best practices

# %% [markdown]
# #### Splitting datasets
#
# Three subsets:
# - training set: the NN is directly trained on this data;
# - validation set: used to assess the NN's performance on the fly; not directly trained on, but the NN is tuned based on the validation set (possible "information leakage");
# - test set: never seen by the NN and used for the final assessment.
#
# Typical fraction: 60% training + 20% validation + 20% test or similar percentages (e.g., 70+20+10).
#
# #### Validation splits
#
# - hold-out validation: partition into training, validation, and test sets;
# - $K$-fold validation (for small datasets):
#     - partition into $K$ subsets, 
#     - use $(K-1)$ of them for training and 1 for validation,
#     - repeat $K$ times,
#     - take the average of the $K$ validation scores;
# - iterative $K$-fold validation (for very small datasets):
#     - use $K$-fold validation,
#     - shuffle the entire dataset,
#     - repeat some number of times,
#     - take the final average.

# %% [markdown]
# #### Baseline model
#
# This is our intuition of how well we can do with a task at hand without any NN. For example, if in the classification example above, we just randomly guess a class, we will be right 50% of the time. Therefore, we want to build a NN with an accuracy higher than 50% at the very least.

# %% [markdown]
# *Example*. The same classification problem, but the dataset is imbalanced: there is 90% of one class and 10% of the other. Our baseline model is to always guess the first class which will make us right 90% of the time.

# %%
size1=900
size2=100

samples1 = rng.multivariate_normal(mean1, cov, size=size1)
samples2 = rng.multivariate_normal(mean2, cov, size=size2)

y1 = np.full(size1, True).astype('int')
y2 = np.full(size2, False).astype('int')

indices = np.arange(size1+size2)
rng.shuffle(indices)

X = np.concatenate((samples1, samples2), axis=0)[indices]
Y = np.concatenate((y1,y2), axis=0)[indices]

################### plotting ###########################3

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

ax.scatter(*X.T, c=Y)


ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_aspect('equal')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# %%
del model

model = get_model(256,512,1024,2048,1024,512,256,128,64,32, last=1)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history_imbalanced = model.fit(
    X, Y, 
    batch_size=128,
    epochs=30,
    validation_split=0.2
)

# %%
val_acc = history_imbalanced.history['val_accuracy']
train_acc = history_imbalanced.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.8, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

################################################################

zz = model.predict(np.array([xx.flatten(),yy.flatten()]).T).reshape(100,100)

subset = rng.choice(indices, size=400, replace=False)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5))

CS = ax.contour(xx, yy, zz, colors='black', levels=[0.], linestyles='dashed')

ax.scatter(*X[subset].T, c=Y[subset])



ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_aspect('equal')

ax.set_xlim(-1,2)
ax.set_ylim(-1,2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

fig.tight_layout()

# %% [markdown]
# #### Getting started with a NN
#
# - make sure that the data is good enough;
# - make the training loss go down (optimization); if it doesn't:
#     - most likely, problem with the optimizer (learning rate, kind of the optimizer, etc.) 
# - make the NN beat a baseline model (optimization); if it doesn't:
#     - most likely, problems with the model's architecture (e.g., using Dense layers instead of Conv layers for image classification) 
# - make the NN overfit (figure out the generalization boundary);
#     - combat overfitting: use more data, a less complex NN, regularization 
# - tune (achieve both optimization and generalization).

# %% [markdown]
# #### Example: MNIST classification with a complex NN and a subset of the data
#
# ##### Overfitting

# %%
(x_train, y_train), _ = mnist.load_data()

x_train = x_train.reshape(-1, 28*28)
x_train = x_train.astype("float32") / 255

del model

model = get_model(128,256,128,64,32)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


history_over = model.fit(
    x_train[:1000], y_train[:1000], 
    batch_size=128,
    epochs=20,
    validation_split=0.2
)

################### plotting ########################

val_acc = history_over.history['val_accuracy']
train_acc = history_over.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.7, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %% [markdown]
# ##### More data

# %%
(x_train, y_train), _ = mnist.load_data()

x_train = x_train.reshape(-1, 28*28)
x_train = x_train.astype("float32") / 255

del model

model = get_model(128,256,128,64,32)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


history_over = model.fit(
    x_train, y_train, 
    batch_size=128,
    epochs=20,
    validation_split=0.2
)

################### plotting ########################

val_acc = history_over.history['val_accuracy']
train_acc = history_over.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.7, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %% [markdown]
# ##### Less complex model

# %%
(x_train, y_train), _ = mnist.load_data()

x_train = x_train.reshape(-1, 28*28)
x_train = x_train.astype("float32") / 255

del model

model = get_model(8,16,32,16,8)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


history_over = model.fit(
    x_train, y_train, 
    batch_size=128,
    epochs=20,
    validation_split=0.2
)

################### plotting ########################

val_acc = history_over.history['val_accuracy']
train_acc = history_over.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.7, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %% [markdown]
# ##### Regularization: $L_2$ and $L_1$

# %%
(x_train, y_train), _ = mnist.load_data()

x_train = x_train.reshape(-1, 28*28)
x_train = x_train.astype("float32") / 255

del model

model = keras.Sequential()

for num_units in [128,256,128,64,32]:
    model.add(Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))

model.add(Dense(10))


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


history_over = model.fit(
    x_train, y_train, 
    batch_size=128,
    epochs=20,
    validation_split=0.2
)

################### plotting ########################

val_acc = history_over.history['val_accuracy']
train_acc = history_over.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.7, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %% [markdown]
# ##### Regularization: Dropout

# %%
(x_train, y_train), _ = mnist.load_data()

x_train = x_train.reshape(-1, 28*28)
x_train = x_train.astype("float32") / 255

del model

model = keras.Sequential()

for num_units in [128,256,128,64,32]:
    model.add(Dense(num_units, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

model.add(Dense(10))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)


history_over = model.fit(
    x_train, y_train, 
    batch_size=128,
    epochs=20,
    validation_split=0.2
)

################### plotting ########################

val_acc = history_over.history['val_accuracy']
train_acc = history_over.history['accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.plot(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_ylim(0.7, 1.)

ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %%
