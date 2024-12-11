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
from tensorflow.keras import layers

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
(x_train,y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

x_train = x_train[...,tf.newaxis].astype('float32') / 256.
x_test = x_test[...,tf.newaxis].astype('float32') / 256.

x_train.shape, x_test.shape

# %%
model = keras.Sequential(
    [
        layers.Conv2D(5, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(5, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
)

# optimization
model.compile(
    optimizer=keras.optimizers.Adam(), 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)


# training
history = model.fit(
    x_train, y_train, 
    batch_size=64, epochs=10, validation_split=0.2, verbose=1,
)

filename = 'mnist_standard.keras'
model.save(filename)

# %%
model_trained = tf.keras.models.load_model(filename)

predictions = model_trained.predict(x_test)

predictions = np.argmax(predictions, axis=1)

confusion_matrix = tf.math.confusion_matrix(y_test, predictions)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))

ax.imshow(confusion_matrix)

print(confusion_matrix)

# %%
from tensorflow.keras import saving

saving.get_custom_objects().clear()

@saving.register_keras_serializable(name="CustomBlock")
class CustomBlock(keras.Layer):
    def __init__(self, units, **kwargs):
        super(CustomBlock, self).__init__(**kwargs)
        self.units = units
        self.conv2d = layers.Conv2D(self.units, 3, padding='same', activation='relu')
        self.maxpool = layers.MaxPooling2D()
        
    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
        }
        return {**base_config, **config}

    def call(self, input_tensor, training=False):

        x = self.conv2d(input_tensor)
        x = self.maxpool(x)
        return x

@saving.register_keras_serializable(name="CustomModel")
class CustomModel(keras.Model):
    def __init__(self, units, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.units = units
        self.block1 = CustomBlock(units)
        self.block2 = CustomBlock(units)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(10)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": self.units,
        }
        return {**base_config, **config}

    def call(self, input_tensor, training=False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x)
        x = self.flatten(x)

        return self.dense(x)
        


# %%
custom_model = CustomModel(5)

# optimization
custom_model.compile(
    optimizer=keras.optimizers.Adam(), 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy'],
    
)


# training
history = custom_model.fit(
    x_train, y_train, 
    batch_size=64, epochs=10, validation_split=0.2, verbose=1,
)

filename = "mnist_custom.keras"
custom_model.save(filename)

# %%
# Now, we can simply load without worrying about our custom objects.
reconstructed_model = keras.models.load_model(filename)

predictions = reconstructed_model.predict(x_test)

predictions = np.argmax(predictions, axis=1)

confusion_matrix = tf.math.confusion_matrix(y_test, predictions)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))

ax.imshow(confusion_matrix)

print(confusion_matrix)

# %%
