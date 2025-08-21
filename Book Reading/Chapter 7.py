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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

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
# ## Sequential API
#
# The Sequential API is the simplest way to build a model in Keras. It allows you to create models layer-by-layer in a step-by-step fashion. It is limited to models that have a single input and a single output, with layers stacked linearly.
# Sequential class
#
# You create a Sequential model by passing a list of layer instances to its constructor, or by using the .add() method.

# %%

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ],
    name="my_simple_model"
)

# %% [markdown]
# #### Models that aren't built yet have no weights
#
# When you create a Sequential model without an input shape, it doesn't know the size of the input it will receive. Therefore, it cannot create its weights (e.g., the kernel and bias matrices for a Dense layer). The model is "unbuilt."

# %%

print(f"Model weights: {model.weights}")


# %% [markdown]
# #### Calling the model for the first time to build it
#
# The model is "built" the first time you call it on some input data. Keras infers the input shape from the data and creates the weights accordingly.

# %%
# Create some dummy data with a batch size of 3 and 32 features
dummy_input = tf.ones(shape=(3, 32))

output = model(dummy_input)

print(f"Number of weight tensors: {len(model.weights)}")

# %% [markdown]
# #### summary() method
#
# The `model.summary()` method prints a string summary of the network, including the layers, their output shapes, and the number of parameters. The model must be built before you can call `.summary()`.

# %%
# Display the model's architecture
model.summary()

# %% [markdown]
# #### Naming models and layers
#
# You can give names to models and layers. This is useful for debugging, visualization, and accessing specific layers.

# %%
model = keras.Sequential(name="named_model")
model.add(layers.Dense(64, activation="relu", name="hidden_layer_1"))
model.add(layers.Dense(10, activation="softmax", name="output_layer"))


# %% [markdown]
# #### Specifying the input shape in advance
#
# You can build a model immediately by specifying the input shape. This can be done by passing an input_shape argument to the first layer or by adding a `keras.Input` object.

# %%

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu", input_shape=(784,)),
        layers.Dense(10, activation="softmax")
    ]
)

model.summary()

# %% [markdown]
# ## Functional API
#
# The Functional API is a more flexible way to build models. It can handle models with non-linear topology, shared layers, and multiple inputs or outputs. It works by creating a graph of layers.
#
# #### A simple model with two Dense layers
#
# A Functional model is built by defining an input node, then connecting the call chain of layers, and finally defining the `keras.Model` by specifying its inputs and outputs.

# %%

inputs = keras.Input(shape=(784,), name="img_input")

# Stack layers, treating them as functions
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
outputs = layers.Dense(10, activation="softmax", name="softmax_output")(x)


model = keras.Model(inputs=inputs, outputs=outputs, name="functional_mnist_model")

model.summary()

# %% [markdown]
# #### A multi-input, multi-output model
#
# The real power of the Functional API is evident when building complex models, such as one that takes multiple inputs and returns multiple outputs.

# %%
image_input = keras.Input(shape=(64, 64, 3), name="image")
metadata_input = keras.Input(shape=(10,), name="metadata")

# Image processing branch
x = layers.Conv2D(32, (3, 3), activation="relu")(image_input)
x = layers.Flatten()(x)
image_features = layers.Dense(16, activation="relu")(x)

# Concatenate image features with metadata
concatenated = layers.concatenate([image_features, metadata_input])

# Define two separate outputs (heads)
class_output = layers.Dense(10, activation="softmax", name="class_output")(concatenated)
aux_output = layers.Dense(1, activation="sigmoid", name="aux_output")(concatenated)

# Create the model with two inputs and two outputs
multi_io_model = keras.Model(
    inputs=[image_input, metadata_input],
    outputs=[class_output, aux_output]
)

keras.utils.plot_model(multi_io_model, "multi_io_model.png", show_shapes=True)
multi_io_model.summary()

# %% [markdown]
# #### Training a model by providing lists of input and target arrays
#
# When training a model with multiple inputs or outputs, you can provide the data as lists. The order of the lists must match the order specified in the `keras.Model` constructor.

# %%

image_input = keras.Input(shape=(64, 64, 3), name="image")
metadata_input = keras.Input(shape=(10,), name="metadata")

# Image processing branch
x = layers.Conv2D(32, (3, 3), activation="relu")(image_input)
x = layers.Flatten()(x)
image_features = layers.Dense(16, activation="relu")(x)

# Concatenate image features with metadata
concatenated = layers.concatenate([image_features, metadata_input])

# Define two separate outputs (heads)
class_output = layers.Dense(10, activation="softmax", name="class_output")(concatenated)
aux_output = layers.Dense(1, activation="sigmoid", name="aux_output")(concatenated)

multi_io_model = keras.Model(
    inputs=[image_input, metadata_input],
    outputs=[class_output, aux_output], 
    name="multi_io_model_list_version"
)

print("--- Model Summary ---")
multi_io_model.summary()


num_samples = 1000

img_data = np.random.rand(num_samples, 64, 64, 3)
meta_data = np.random.rand(num_samples, 10)


class_targets = np.random.randint(0, 10, num_samples)      # For class_output
aux_targets = np.random.randint(0, 2, (num_samples, 1)) # For aux_output


# The list of losses MUST match the model's output order.
# 1st loss is for the 1st output (class_output)
# 2nd loss is for the 2nd output (aux_output)
multi_io_model.compile(
    optimizer="adam",
    loss=["sparse_categorical_crossentropy", "binary_crossentropy"],
    loss_weights=[1.0, 0.2] # Optional
)


print("\n--- Training Model with Lists ---")
# The list of target arrays MUST also match the model's output order.
multi_io_model.fit(
    x=[img_data, meta_data],
    y=[class_targets, aux_targets],
    epochs=1,
    batch_size=32
)

print("\n--- List-based training complete! ---")

# %% [markdown]
# #### Training a model by providing dicts of input and target arrays
#
# Using dictionaries is often safer because it doesn't rely on the order of inputs/outputs. You map the names of the input and output layers to their corresponding data arrays.

# %%

image_input = keras.Input(shape=(64, 64, 3), name="image")
metadata_input = keras.Input(shape=(10,), name="metadata")

# Image processing branch
x = layers.Conv2D(32, (3, 3), activation="relu")(image_input)
x = layers.Flatten()(x)
image_features = layers.Dense(16, activation="relu")(x)

# Concatenate image features with metadata
concatenated = layers.concatenate([image_features, metadata_input])

# Define two separate outputs (heads)
class_output = layers.Dense(10, activation="softmax", name="class_output")(concatenated)
aux_output = layers.Dense(1, activation="sigmoid", name="aux_output")(concatenated)

multi_io_model = keras.Model(
    inputs=[image_input, metadata_input],
    outputs={"class_output": class_output, "aux_output": aux_output}, # Using a dict here for clarity
    name="multi_io_model"
)

multi_io_model.summary()


num_samples = 1000
img_data = np.random.rand(num_samples, 64, 64, 3)
meta_data = np.random.rand(num_samples, 10)

class_targets = np.random.randint(0, 10, num_samples) # Shape: (1000,)
aux_targets = np.random.randint(0, 2, (num_samples, 1)) # Shape: (1000, 1)


# We instantiate the loss objects directly and pass them in a dictionary.
multi_io_model.compile(
    optimizer="adam",
    loss={
        "class_output": keras.losses.SparseCategoricalCrossentropy(),
        "aux_output": keras.losses.BinaryCrossentropy()
    },
    metrics={
        "class_output": ["accuracy"],
        "aux_output": ["accuracy"]
    }
)


print("\n--- Training Model ---")
multi_io_model.fit(
    x={"image": img_data, "metadata": meta_data},
    y={"class_output": class_targets, "aux_output": aux_targets},
    epochs=1,
    batch_size=32,
    verbose=1
)

print("\n--- Training complete! ---")

# %% [markdown]
# #### Functional API: Access to layer connectivity
#
# The Functional API creates a graph of layers that you can inspect and reuse. You can get a layer from the model by its name and inspect its input and output tensors.

# %%
dense_layer = model.get_layer("dense_1")

print("Input to dense_1:", dense_layer.input)
print("Output of dense_1:", dense_layer.output)

# %% [markdown]
# #### Creating a new model by reusing intermediate layer outputs
#
# You can create a new model by using the output of an intermediate layer from an existing model. This is very useful for feature extraction.

# %%

feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("dense_1").output
)

dummy_input = tf.ones((2, 784))
features = feature_extractor(dummy_input)
print(f"Shape of extracted features: {features.shape}")


# %% [markdown]
# ## Subclassing the Model class
#
# For maximum flexibility, you can subclass keras.Model. This is a fully object-oriented approach where you define your model's architecture in Python code.
#
#  - `__init__(self)`: Define all the layers your model will use.
#
#  - `call(self, inputs)`: Define the forward pass of the model. This is where you specify how the layers connect to each other.
#
# #### A simple subclassed model
#
# Here we replicate our simple two-layer model by subclassing `keras.Model`.

# %%
class MySimpleModel(keras.Model):

    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

subclassed_model = MySimpleModel()

# building the model
_ = subclassed_model(tf.ones((1, 784)))

subclassed_model.summary()


# %% [markdown]
# #### Mixing and matching different components
#
# You can freely mix and match these different model-building styles. For instance, you can use a subclassed model as a layer inside a Functional model, and vice-versa.
#
# #### Creating a Functional model that includes a subclassed model
#
# A custom subclassed model can be used just like any other Keras layer.

# %%
class CustomBlock(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(32, activation="relu")
    
    def call(self, inputs):
        return self.dense2(self.dense1(inputs))

inputs = keras.Input(shape=(128,))
custom_features = CustomBlock()(inputs)
outputs = layers.Dense(10, activation="softmax")(custom_features)

combined_model = keras.Model(inputs=inputs, outputs=outputs)
combined_model.summary()

# %% [markdown]
# #### Creating a subclassed model that includes a Functional model
#
# You can also do the reverse: use a Functional model as a component inside a subclassed model.

# %%
encoder_input = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Flatten()(x)
encoder_output = layers.Dense(64, activation="relu")(x)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")

class Classifier(keras.Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = layers.Dense(10, activation="softmax")
        
    def call(self, inputs):
        features = self.encoder(inputs)
        return self.classifier(features)

model_with_functional_block = Classifier(encoder)
_ = model_with_functional_block(tf.ones((1, 28, 28, 1)))
model_with_functional_block.summary()

# %% [markdown]
# ## Built-in training and evaluation loops
#
# Keras provides a complete, high-level workflow for training, evaluating, and making predictions with your models.
# The standard workflow
#
# The standard workflow is:
#
#  - `compile()`: Configure the model for training. You specify the optimizer, loss function, and metrics.
#  - `fit()`: Train the model on your data.
#  - `evaluate()`: Evaluate the model's performance on a test set.
#  - `predict()`: Generate predictions for new data.

# %%
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(784,)),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

x_train = np.random.rand(1000, 784)
y_train = np.random.randint(0, 10, 1000)
x_val = np.random.rand(200, 784)
y_val = np.random.randint(0, 10, 200)

history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_val, y_val)
)

print("\n--- Evaluation ---")
loss, acc = model.evaluate(x_val, y_val)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}")

print("\n--- Prediction ---")
predictions = model.predict(x_val[:5])
print(f"Shape of predictions: {predictions.shape}")

# %% [markdown]
# #### Callbacks: EarlyStopping and ModelCheckpoint
#
# Callbacks are objects that can perform actions at various stages of training (e.g., at the start or end of an epoch). Two of the most common are `EarlyStopping` and `ModelCheckpoint`.
#
#  - `EarlyStopping`: Stops training when a monitored metric has stopped improving.
#  - `ModelCheckpoint`: Saves the model's weights during training, typically saving only the best-performing version.

# %%
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    patience=3,          # Stop after 3 epochs of no improvement
    verbose=1
)

model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath="best_model.keras",  # Path to save the model file
    save_best_only=True,       # Only save if val_loss improves
    monitor="val_loss",
    verbose=1
)

model.fit(
    x_train,
    y_train,
    epochs=20, # Set a high number of epochs; EarlyStopping will handle it
    validation_data=(x_val, y_val),
    callbacks=[early_stopping_cb, model_checkpoint_cb]
)


# %% [markdown]
# ## tf.function decorator
#
# The `@tf.function` decorator compiles a Python function that uses TensorFlow operations into a high-performance, static TensorFlow graph. This can significantly speed up your code. Keras automatically applies this optimization to the call method of any model when you use fit, evaluate, or predict.
#
# You can also use it manually for your own custom functions, which is essential when writing custom training loops.

# %%
def my_tf_function(x):
    return tf.square(x) + tf.reduce_mean(x)

@tf.function
def my_compiled_function(x):
    print("Tracing the function...")
    return tf.square(x) + tf.reduce_mean(x)

a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

print("First call:")
result1 = my_compiled_function(a) 
print(result1)

print("\nSecond call:")
result2 = my_compiled_function(b) 
print(result2)

# %% [markdown]
# The "Tracing the function..." message is only printed once for a given input signature, demonstrating that subsequent calls are using the pre-compiled, optimized graph instead of re-executing the Python code line-by-line. This is the key to TensorFlow's high performance.

# %%
