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
# Here we will work with the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset of handwritten digits. First, we will try fix the overfitting we ran into with dropout layers, and then we will experiment with a new architecture, the so-called convolutional neural networks (CNN) which are well suited for image recognition.

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt

MEDIUM_SIZE = 14
BIGGER_SIZE = 18

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
# #### Datasets
#
# Although it is possible to use `tensorflow` to load the MNIST datasets, let us first just download it from the source. The original MNIST website throws an access denied error. We will use a [GitHub repo](https://github.com/lorenmh/mnist_handwritten_json) instead to download a training set `mnist_handwritten_train.json` and a test set `mnist_handwritten_test.json`.

# %%
import json

with open('../Session 3/data/mnist_handwritten_train.json', 'r') as file:
    training_json = json.load(file)

with open('../Session 3/data/mnist_handwritten_test.json', 'r') as file:
    testing_json = json.load(file)

# %%
# arr = np.array(training_json[0]['image'])
# print(training_json[0]['label'])

# plt.imshow(arr.reshape(28,28))

rng = np.random.default_rng()

train_len = len(training_json)

images = []
labels = []
for i in range(25):
    index = rng.integers(train_len-1)
    images.append(np.array(training_json[index]['image']).reshape(28,28))
    labels.append(training_json[index]['label'])


fig,axes = plt.subplots(ncols=5, nrows=5, figsize=(10,10))


for image,label,ax in zip(images,labels,axes.flatten()):

    ax.imshow(image)
    ax.annotate(label, xy=(0.8,0.8), xycoords='axes fraction', color='white')
    ax.set_axis_off()
    #ax.title.set_text(label)


# %%
x_train = []
y_train = []
x_test = []
y_test = []

for data in training_json:
    x_train.append(data['image'])
    y_train.append(data['label'])

for data in testing_json:
    x_test.append(data['image'])
    y_test.append(data['label'])

x_train = np.array(x_train) / 255.
y_train = np.array(y_train)
x_test = np.array(x_test) / 255.
y_test = np.array(y_test)

# %%
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# #### Fully connected NN with dropout layers

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.optimizers import Adam
# digit 5
#[0,0,0,0,1,0,0,0,0,0]
#[0.1,0.1,0.01,0.01,0.7,,,,,]

# %% [markdown]
# A `Dropout(fraction)` layer breaks a fraction `fraction` of connections between two adjacent layers. The intuition is to prevent units in the two layers from "conspiring" to overfit the training data.
#

# %%
dropout_frac = 0.3

model = keras.Sequential(
    [
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout_frac),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_frac),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_frac),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout_frac),
        layers.Dense(32, activation='relu'),
        layers.Dense(10),
    ]
)

# optimization
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# training
history = model.fit(
    x_train, y_train, 
    batch_size=64, epochs=50, validation_split=0.2, verbose=2,
)

# %% [markdown]
# **Results**: From the plots below, it is clear that the model is overfitting: it performs much better on training data than on validation data. There are a few ways to combat overfitting:
# - make the model less complex (fewer trainable parameters),
# - introduce regularization,
# - introduce dropout layers (randomly drop a fraction of connections between two neighboring layers).

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))


ax.plot(acc, label='training')
ax.plot(val_acc, label='validation')
ax.set_xlabel('epoch #')
ax.set_ylabel('accuracy')

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)
ax.legend()


# %%
predictions = model.predict(x_test)

predictions = np.argmax(predictions, axis=1)

confusion_matrix = tf.math.confusion_matrix(y_test, predictions)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))

ax.imshow(confusion_matrix)

print(confusion_matrix)

# %%
cond = np.logical_and(y_test==4, predictions==9)

confused_images = x_test[cond]
confused_labels = y_test[cond]
conf_len = len(confused_images)

images = []
labels = []
for i in range(25):
    index = rng.integers(conf_len-1)
    images.append(np.array(confused_images[index]).reshape(28,28))
    labels.append(confused_labels[index])


fig,axes = plt.subplots(ncols=5, nrows=5, figsize=(10,10))


for image,label,ax in zip(images,labels,axes.flatten()):

    ax.imshow(image)
    ax.annotate(label, xy=(0.8,0.8), xycoords='axes fraction', color='white')
    ax.set_axis_off()
    #ax.title.set_text(label)

# %% [markdown]
# ### Convolutional NN

# %% [markdown]
# A convolutional NN works directly with images and expects its input in the form of 3D arrays, where the first two dimensions are for height and width and the third dimension is for channels (e.g., 3 channels for RGB and 1 channel for grat-scale images).

# %%
x_cnn_train = x_train.reshape(-1,28,28,1)
x_cnn_test = x_test.reshape(-1,28,28,1)

model = keras.Sequential(
    [
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(5, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(5, 3, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

# optimization
model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary()

# training
history = model.fit(
    x_cnn_train, y_train, 
    batch_size=64, epochs=50, validation_split=0.2, verbose=2,
)

# %% [markdown]
# As we can see from the confusion matrix below, the CNN performs better than a fully connected NN with Dense layers.

# %%
predictions = model.predict(x_cnn_test)

predictions = np.argmax(predictions, axis=1)

confusion_matrix = tf.math.confusion_matrix(y_test, predictions)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))

ax.imshow(confusion_matrix)

print(confusion_matrix)

# %%
