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

# %% [markdown] id="wbzUpaWqt1Il"
# This is a companion notebook for the book [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.6.

# %% [markdown] id="R7TfmwvRt1Ir"
# # Introduction to Keras and TensorFlow

# %% [markdown] id="sC0u-JhBt1Is" jp-MarkdownHeadingCollapsed=true
# ## What's TensorFlow?
# TensorFlow is Python-based machine learning software library. It is like Numpy in that it is used to manipulate mathematical expressions over numerical tensors. It exceeds the scope of Numpy because
#
# - It can automatically compute the gradient of any differentiable expression
# - It can easily parallelize on GPUs and TPUs
# - Computation in TensorFlow can be distributed across several machines
# - TensorFlow programs can be exported to other runtimes like C++, Javascript or TensorFlow Light

# %% [markdown] id="mo_wtouEt1It" jp-MarkdownHeadingCollapsed=true
# ## What's Keras?
# Keras is a deep learning API built on top of TensorFLow. It can run on CPUs, GPUs, or TPUs. See Figure 3.1, pg. 70

# %% [markdown] id="TCgyXGwZt1It" jp-MarkdownHeadingCollapsed=true
# ## Keras and TensorFlow: A brief history
#
#
# - Keras was created 8 months before TensorFlow both in 2015 (it was originally built on another tensor manipulation library called Theano)
# - In late 2015, Keras was refactored to be able to use either Theano or TensorFlow
# - By 2016, TensorFlow became the default backend for Keras

# %% [markdown] id="TxpWD5i1t1It"
# ## Setting up a deep-learning workspace
#
# Nvidia GPU recommended. Even on applications that can strictly run on CPU, GPU offers 5 to 10x speed up on average. It is also recommended to use a Unix workstation, not Windows. 

# %% [markdown] id="qspevM51t1Iu" jp-MarkdownHeadingCollapsed=true
# ### Jupyter notebooks: The preferred way to run deep-learning experiments
#
# Use a Jupyter Notebook

# %% [markdown] id="Oz4bG3fJt1Iv"
# ### Using Colaboratory
#
# Use Google Colab (a cloud service Jupyter Notebook) to run keras directly off the cloud and get access to free, but limited, GPU and TPU runtime. 

# %% [markdown] id="4BNYQk0Ct1Iv"
# #### First steps with Colaboratory
#
# Go to https://colab.research.google.com/ and you can start creating notebooks. You can create code cells (Figure 3.3) or text cells (Figure 3.4)

# %% [markdown] id="mmJzAEE0t1Iv"
# #### Installing packages with pip
#
# The deafault colab environment comes with Keras and Tensorflow already pre-installed. If you wish to install any additional packages, use pip install command via
#
# `!pip install package_name`
#
# The ! is to indicate this is a shell command, not python code. 

# %% [markdown] id="PDzB9qMTt1Iw"
# #### Using the GPU runtime
#
# To select GPU runtime in Colab, just click Runtime -> Change Runtime Type and then select GPU for Hardware Accelerator (Figure 3.5)
#
# Note: There is also a TPU option, but this requires some additional setup (see Ch. 13)

# %% [markdown] id="y90Q2EBKt1Iw"
# ## First steps with TensorFlow
#
# Training a neural network involves the following concepts:
#
# - First, low-level Tensor manipulation. This is what TensorFlow APIs is designed to do.
#     - _Tensors_, including special tensors that store the network's state (_variables_)
#     - _Tensor operations_ such as addition, `relu`, `matmul`
#     - _Backpropagation_, a way to compute the gradient of mathematical expressions (handled in TensorFlow via the `GradientTape` object)
# - Second, high-level deep learning concepts. This is what Keras APIs are designed to do.
#     - _Layers_, which are combined into a model
#     - A _loss function_, which defines the feedback signal used for learning
#     - An _optimizer_, which determines how learning proceeds
#     - _Metrics_ for evaluating model performance and accuracy
#     - A _training loop_ that performs mini-batch stochastic gradient descent

# %% [markdown] id="zPlMutOQt1Iw"
# #### Constant tensors and variables
#
# If you want to use TensorFlow, you first need to know how to make tensors. Tensors must be initialized with some value, like all ones, all zeroes, or something random. 

# %% [markdown] id="OAmJv-sct1Ix"
# **All-ones or all-zeros tensors**

# %% id="VhVWrihBt1Ix"
import tensorflow as tf
x = tf.ones(shape=(2, 1)) #This is equivalent to np.ones(shape = (2,1))
print(x)

# %% id="3clHgjdht1Iz"
x = tf.zeros(shape=(2, 1)) #This is equivalent to np.zeros(shape = (2,1))
print(x)

# %% [markdown] id="to3HNgAFt1Iz"
# **Random tensors**

# %% id="pwSB3A83t1Iz"
x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.) #Tensor of random values drawn from a normal distribution with mean 0 and standard deviation 1.
print(x) #This is equvalent to np.random.normal(size = (3,1), loc = 0., scale =1.)

# %% id="-MOuXGlct1Iz"
x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)  #Tensor of random values drawn from a uniform distribution between 0 and 1. 
print(x) #This is equvalent to np.random.uniform(size = (3,1), low = 0., high =1.)

# %% [markdown] id="QzArJwL5t1Iz"
# **NumPy arrays are assignable**
#
# This means you can do the following

# %% id="ahecoL4qt1I0"
import numpy as np
x = np.ones(shape=(2, 2))
x[0, 0] = 0. #You can assign the value of a particular entry in a numpy array
print(x)

# %% [markdown]
# However, TensorFlow tensors are __not__ assignable. Tensors in TensorFlow are constant. Trying to do this in TensorFlow will result in the error: 
# "EagerTensor object does not support item assignment."

# %% [markdown] id="bTat2vjft1I0"
# **Creating a TensorFlow variable**
#
# In order manage modifiable state in tensor flow, you need to use _variables_. To create a TensorFlow variable, you need an initial value, such as a random tensor.

# %% id="6Sg-YLxFt1I0"
v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1))) 
print(v)

# %% [markdown] id="Ge90Lt67t1I0"
# **Assigning a value to a TensorFlow variable**
#
# We can modify the state of a TensorFlow variable using the `assign` method

# %% id="XVWyCRGmt1I1"
v.assign(tf.ones((3, 1)))

# %% [markdown] id="IWxex8Att1I1"
# **Assigning a value to a subset of a TensorFlow variable**
#
# We can modify a single coefficient or any subset of coefficients. 

# %% id="H5ZH5GHGt1I1"
v[0, 0].assign(3.)

# %% [markdown] id="EPC0jVjxt1I1"
# **Using `assign_add`**
#
# We can also use `assign_add()` or `assign_sub()` to add or subtract a tensor to a variable. 

# %% id="9nLNm1kTt1I1"
v.assign_add(tf.ones((3, 1)))

# %% [markdown] id="VGzBD_Est1I1"
# #### Tensor operations: Doing math in TensorFlow
#
# Just like NumPy, TensorFlow has a built-in collection of tensor operations to express mathematical formulas. For example:

# %% [markdown] id="zpDJH9DVt1I2"
# **A few basic math operations**

# %% id="RGv0NIhzt1I2"
a = tf.ones((2, 2))
b = tf.square(a) #Take the square (element-wise)
c = tf.sqrt(a) #Take the square root (element-wise)
d = b + c #Add two tensors (element-wise)
e = tf.matmul(a, b) #Take the product of two tensors (matrix multiply)
e *= d #Multiply two tensors (element-wise)

# %% [markdown]
# Each of these operations get executed on the fly, and you can print out what the current result is at any point. This is called _eager execution_. 

# %% [markdown] id="0zhM_v03t1I2"
# #### A second look at the GradientTape API
#
# TensorFlow can compute the gradient of any differentiable expression with respect to one of its inputs. 
#
# The GradientTape "records" relevant operations executed inside the context of a tf.GradientTape onto a "tape". TensorFlow then uses that tape to compute the gradients of a "recorded" computation using reverse mode differentiation.

# %% [markdown] id="HzfMpFFrt1I2"
# **Using the `GradientTape`**

# %% id="xwKCNz2ut1I2"
input_var = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
   result = tf.square(input_var)
gradient = tape.gradient(result, input_var)
print(gradient)

# %% [markdown] id="UFV7eHUnt1I2"
# **Using `GradientTape` with constant tensor inputs**
#
# The previous example showcases how GradientTape can track the operations on a TensorFlow variable in order to perform automatic differentiation. In fact, it is possible for the input to be any arbitrary tensor. However, TensorFlow only tracks _trainable variables_ by default. With a constant vector, you have to manually mark it as being tracked by calling `tape.watch()` on it. 

# %% id="LDn0mWcVt1I3"
input_const = tf.constant(3.)
with tf.GradientTape() as tape:
   tape.watch(input_const)
   result = tf.square(input_const)
gradient = tape.gradient(result, input_const)

# %% [markdown]
# The reason this is designed this way is because it would be too expensive to preemptively store all the information required to compute a gradient for every single tensor, the automatic storing of this information is made default for the _variable_ tensors only as that is the most common use case for gradient calculations.

# %% [markdown] id="bcRns81Nt1I3"
# **Using nested gradient tapes to compute second-order gradients**
#
# You can also compute higher order derivatives by using multiple GradientTapes. Here, the outer tape computes the gradient of the gradient from the inner tape. In this case, it is the same as differentiating with respect to `time` twice, so the output is just `9.8`. 

# %% id="F2ryIwwlt1I3"
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position =  4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)

# %% [markdown] id="B5gkOq8Xt1I-"
# #### An end-to-end example: A linear classifier in pure TensorFlow
#
# Now you know about tensors, variables, tensor operations and gradients. This is enough to build a machine learning model. 
#
# Let's come up with _linear classifier_ model which tries to classify between two classes of random points. They are each generated via sampling from a multivariate normal distribution and will have a specific covariance matrix and specific mean. The covariance matrix specifies the shape or spread of the points around the mean point. We will use the same covariance matrix for both classes of points, but different mean values, leading to two distinct "point clouds" with different positions. They can be easily identified by eye, especially if we color coded them. 

# %% [markdown] id="0-kLQ6Iut1I_"
# **Generating two classes of random points in a 2D plane**
#
# - We generate 1000 random 2D points for the first class with the covariance matrix being `cov = [[1, 0.5], [0.5,1]]` which corresponds to an oval-like point cloud oriented from bottom left to top right.
# - We do the same for the second class but choose a different mean value so that these clouds of points are relatively separated. 

# %% id="UEawWzL8t1I_"
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)

# %% [markdown] id="1UhEHnect1I_"
# **Stacking the two classes into an array with shape (2000, 2)**
#
# We can mush the two classes together into one stack.

# %% id="x-p77dDDt1I_"
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

# %% [markdown] id="okZrYE6Ft1I_"
# **Generating the corresponding targets (0 and 1)**
#
# We can create a _targets_ array with the same dimension as the stack with values 0 corresponding to the first 1000 points of the first class and values 1 to the next 1000 points of the second class. These will act as labels for our points so we can color-code them when plotting.

# %% id="pmjdoFEtt1JA"
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

# %% [markdown] id="MnYlxPirt1JA"
# **Plotting the two point classes**

# %% id="mzq6_mFqt1JA"
import matplotlib.pyplot as plt
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0]) #We color-code the points using our targets array.
plt.show()

# %% [markdown] id="sXEOUtkDt1JA"
# Now we can create a linear classifier model learns to separate between the two blobs. A linear classifier is an affine transformation `(prediction = W * input + b)` trained to minimize the square of the difference between predictions and the targets. 
#
# **Creating the linear classifier variables**
#
# Lets create our variables `W` and `b`:

# %% id="3ov2bz2Kt1JA"
input_dim = 2 #The inputs will be 2D points
output_dim = 1 #The output predictions will be a single score per sample (close 0 if predicted to be in class 0, and close to 1 if predicted to be in class 1)
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


# %% [markdown] id="gubcQiHnt1JA"
# **The forward pass function**
#
# Now we define our forward pass function.

# %% id="wQumodFWt1JB"
def model(inputs):
    return tf.matmul(inputs, W) + b


# %% [markdown]
# Essentially, if the input is `(x,y)` and `W = [[w1],[w2]]`, our prediction output is just:
# `prediction = w1 * x + w2 * y + b`

# %% [markdown] id="AHLSV1DQt1JB"
# **The mean squared error loss function**
#
# This will be our loss function. The mean of squares of the differences between the target value and predictions. 

# %% id="URvcQicAt1JB"
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)


# %% [markdown] id="Q4lIHSVmt1JB"
# **The training step function**
#
# Next is the training step which receives some training data and updates the weights `W` and `b` so as to minimize the loss on the data.

# %% id="2Fg5lX1dt1JB"
learning_rate = 0.1 # Defines how much the weights are updated at each step (i.e. the step-size of the gradient descent).

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs) # Our forward pass function pred = W * inputs + b
        loss = square_loss(targets, predictions) # Our loss function
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b]) #The gradient of the loss with respect to our weights
    W.assign_sub(grad_loss_wrt_W * learning_rate) #Update the weights
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss


# %% [markdown]
# For simplicity, we are doing _batch_ training instead of _mini-batch_ training. So we will be processing our entire batch of data at once instead of breaking it up into smaller batches. This takes longer per learning cycle, but on the other hand, the gradient is more effective at reducing the loss per training step, so we can use a larger learning rate of 0.1

# %% [markdown] id="yS3d4bkNt1JC"
# **The batch training loop**
#
# Let's do 40 steps of training

# %% id="6Ms6jaOst1JC"
for step in range(40):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

# %% [markdown]
# Our loss has stabilized around 0.028. Let's plot the predictions by color coding values >0.5 as corresponding to the class with target = 1 and values less then 0.5 corresponding to the class with target = 0.

# %% id="rfDj6IjVt1JC"
predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

# %% [markdown]
# Recall that our prediction equation is just a linear equation of x and y. 
#
# The class 1 is defined when `w1 * x + w2 * y + b > 0.5` 
#
# The class 0 is defined when `w1 * x + w2 * y + b < 0.5` 
#
# Then the line: `w1 * x + w2 * y + b = 0.5` is the line that separates the two classes.
#
# We can re-write this into an equation y(x) and plot the line.

# %% id="Ti4GP_8_t1JC"
x = np.linspace(-1, 4, 100)
y = - W[0] /  W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)

# %% [markdown] id="TzzcUP-Pt1JC"
# ## Anatomy of a neural network: Understanding core Keras APIs
#
# We can now create a toy model from scratch. Now let us move onto a more robust system: Keras. 

# %% [markdown] id="E_5KAxTYt1JC"
# ### Layers: The building blocks of deep learning
#
# - The fundamental building block in neural networks is the _layer_.
# - A _layer_ is a data processing module that takes one or more tensors as input and gives one or more tensors as output.
# - Sometimes layers are stateless, but most of the time they do have a state: the layer's _weights_, one of several tensors which are learned with stochastic gradient descent that compose the network's knowledge.
# - Different types of layers are appropriate for different tensor formats and different types of data processing.
#     - Simple vector data are stored in rank-2 tensors with shape `(samples, features)` is often processed by _densely connected_ layers, also called _fully connected_ or _dense_ layers which is the `Dense` class in Keras.
#     - Sequence data, stored in rank-3 tensors of shape `(samples, timesteps, features)` is typically processed by _recurrent_ layers such as an `LSTM` layer, or 1D convolution layers (`Conv1D`). Image data, stored in rank-4 tensors, is usually processed by 2D convolution layers (`Conv2D`)

# %% [markdown] id="sl3t0h8Dt1JD"
# #### The base Layer class in Keras
#
# Everything in Keras is either a `Layer` or something that closely interacts with a `Layer`. 
#
# A `Layer` encapsulates some state (weights) + some computation (a forward pass). The weights are typically defined in a `build()`, although they can be created in the constructor `__init__()`, and the computation is defined in the `call()` method.
#
# In Chapter 2, we implemented a `NaiveDense` class that contained two weights `W` and `b` and applied the computation:
#
# `output = activation(dot(input,W)+b)`
#
# This same layer would be implemented in Keras as follows.

# %% [markdown] id="NFvV-GB8t1JF"
# **A `Dense` layer implemented as a `Layer` subclass**

# %% id="8g-A5Q08t1JF"
from tensorflow import keras

class SimpleDense(keras.layers.Layer): #Every Keras layer inherits the base `Layer` class

    def __init__(self, units, activation=None): #We initialize the data objects beloning to the class
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape): #We create the initial weights in the build() method
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal") 
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")
"""
add_weight() is a shortcut for adding weights, you can also create a standalone variable and assign them as layer attributes
For example: self.W = tf.Variable(tf.random.uniform(w_shape))
"""
    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

# %% [markdown]
# We will undertand the purpose of using the `build()` and `call()` functions later.
# Once instantiated, a layer like this can be used just like a function, taking as input a TensorFlow tensor:

# %% id="JydVQij_t1JF"
my_dense = SimpleDense(units=32, activation=tf.nn.relu) #Instantiate our layer using the SimpleDense class we just created
input_tensor = tf.ones(shape=(2, 784)) # Create some test inputs
output_tensor = my_dense(input_tensor) #Call the layer on the inputs, just like a function
print(output_tensor.shape)

# %% [markdown]
# So why did we have to implement `call()` and `build()`, since we ended up just using our layer by just calling it, i.e. by just using its call method?
# It is because we want to be able to create the state just in time. Let's see how that works.

# %% [markdown] id="5dL0Thqot1JF"
# #### Automatic shape inference: Building layers on the fly
#
# You can "clip" together layer that are compatible, i.e. a layer which can accept input tensors that is of the shape of the output tensor of another. Consider the example:

# %% id="pJLrUcQtt1JF"
from tensorflow.keras import layers
layer = layers.Dense(32, activation="relu")

# %% [markdown]
# This layer returns a tensor where the first dimension has been transformed to be 32 units. It can only be connected to a downstream layer that expects a 32 dimensional vector as its input. 
#
# Most of the time, you don't need to worry about layers being size compatible, as they are dynamyically built to match the shape of the incoming layer. For example, if you write:

# %% id="cxS7Olnwt1JG"
from tensorflow.keras import models
from tensorflow.keras import layers
model = models.Sequential([
    layers.Dense(32, activation="relu"),
    layers.Dense(32)
])

# %% [markdown]
# The layers didn't receive any information about the shape of their inputs, instead they automatically inferred the input shape as being the first inputs they see. Note that the `32` refers to the dimension of the output tensor, not the input. 
#
# In Chapter 2, there was this example of NaiveDense where we had to specify the input and output tensor lengths for each layer. Now, with SimpleDense, you don't have to do that. Each layer can take a tensor of whatever input size as follows.
#

# %% id="Yq8V1EPbt1JG"
model = keras.Sequential([
    SimpleDense(32, activation="relu"),
    SimpleDense(64, activation="relu"),
    SimpleDense(32, activation="relu"),
    SimpleDense(10, activation="softmax")
])


# %% [markdown]
# The `build()` method provides a dedicated way of creating state on the fly which receipes as an argument the input shape seen by the layer. The `build()` method is _automatically_ called the first time the layer is called (i.e. via its `__call()__` method). This is separate from the `call()` method which we defined separately to handle computation (the `__call()__` method will call on the `call` method also at the very end. 
#
# The `__call()__` method of the Keras base layer looks like this

# %%
def __call__(self, inputs):
    if not self.built:
        self.build(inputs.shape)
        self.built = True
    return self.call(inputs)


# %% [markdown]
# Long story short: Put your forward pass in the `call()` method you define.

# %% [markdown] id="wsUrxNcFt1JG"
# ### From layers to models
#
# A deep learning __model__ is a graph of layers. In Keras, that's the `Model` class. Until now, you have only seen `Sequential` models, which are simple stacks of layers, mapping a single input to a single output. However, there are more complex topologies available. Some common topologies include:
#
# - Two-branch networks
# - Multihead networks
# - Residual connections
#
# See Figure 3.9 to see a Transformer model. It is quite involved, and you will understand it in later chapters. 
#
# There are two ways of building models in Keras:
#
# - Use the `Model` subclass directly
# - Use the Functional API which allows you to do more with less code
#
# Both approaches will be covered in Chapter 7.
#
# The topology of a model defines a _hypothesis space_. Machine learning searches for useful representations of some input data within a predefined _space of possibilities_, using guidance from a feedback signal. 
#
# By choosing a network topology, you constrain your space of possibilities (i.e. your hypothesis space).
#
#
# To learn from data, you have to make certain assumptions. These assumptions define what can be learned. As such, the architecture of your model is extremely important. It encodes the assumptions you make about your problem, the prior knowlege that your model starts with. 
#
# For ex. 
#
# - If you are working on a two class classification problem with a model made of a single `Dense` layer with no activation (a pure affine transformation), you are assumming your two classes are linearly separable.
#
# - Choosing the right network architecutre is more of an art than a science

# %% [markdown] id="3qsDn4u0t1JG"
# ### The "compile" step: Configuring the learning process
#
# Once the model architecture is defined, you still have to choose three more things:
#
# - _Loss function_ (_objective function_) - which is what you wish to minimize during training
# - _Optimizer_ - which is how the network will be updated based on the loss function. It implements a specific variant of stochastic gradient descent (SGD)
# - _Metrics_ - The measures of success you wish to monitor during training and validation, such as classification accuracy for example. Unlike the loss, training will not directly optimize for these metrics, so they need not be differentiable!
#
#
# `compile()` configures the training process and takes in the optimizer, loss, and metrics. 

# %% id="PiWrZWo8t1JH"
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer="rmsprop",
              loss="mean_squared_error",
              metrics=["accuracy"])

# %% id="FY9pkvNBt1JH"
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

# %% [markdown]
# You can supply your own custom losses or metrics if you want!

# %% [markdown]
# ### List of Built in Options
#
# #### Optimizers
#
# - `SGD` (with or without momentum)
# -  `RMSprop`
# -  `Adam`
# -  `Adagrad`
# -  Etc.
#
# #### Losses
# - `CategoricalCrossentropy`
# - `SparseCategoricalCrossentropy`
# - `BinaryCrossentropy`
# - `MeanSquaredError`
# - `KLDivergence`
# - `CosineSimilarity`
# - Etc.
#
# #### Metrics
# - `CategoricalAccuracy`
# - `SparseCategoricalAccuracy`
# - `BinaryAccuracy`
# - `AUC`
# - `Precision`
# - `Recall`
# - Etc.

# %% [markdown] id="xLMdXHFgt1JH"
# ### Picking a loss function
#
# Picking a loss function can be tricky and must be chosen wisely. 
#
# There are some common guidelines for picking a loss function in simple examples. 
#
# - Use `BinaryCrossentropy` for a two-class classification problem
# - Use `CategoricalCrossentropy` for a many-class classification problem
# - And so on
#
# When you are working on truly new research problems, you will have to define your own loss functions (see next few chapters). 
#

# %% [markdown] id="EbM7yyxXt1JH"
# ### Understanding the fit() method
#
# After `compile()` comes `fit()`. The `fit() method implements the training loop itself. These are the key arguments
#
# - The _data_ (inputs and targets to train on). It will typically be passed either in the form of Numpy arrays or TensorFlow `Dataset` objects. You'll learn more about the `Dataset` API in the next chapters.
# - The number of _epochs_ to train for: how many times the training loop should iterate over the data passed.
# - The _batch_size_ to use within each epoch of the mini-batch gradient descent: the number of training examples considered to compute the gradients for one weight update step.

# %% [markdown] id="DVBTpmwRt1JH"
# **Calling `fit()` with NumPy data**

# %% id="IVYsuKbFt1JI"
history = model.fit(
    inputs, # inputs come in as a numpy array
    targets, # the trainings come in as a numpy array
    epochs=5, # the training loop will iterate over the data 5 times
    batch_size=128 #The training loop will iterate over the data in batches of 128 examples
)

# %% [markdown]
# The call to `fit()` returns a `History` object which contains a `history` field which is a dictionary mapping keys such as "`loss`" or specific metric names to the list of their per-epoch values. 

# %% id="3qZgURv-t1JI"
history.history

# %% [markdown] id="x5yWP393t1JI"
# ### Monitoring loss and metrics on validation data
#
# To keep an eye on how the model does on new data, it's standard practice to reserve a subset of the training data as _validation data_ to test the model against. This is done to prevent the model from "memorizing" a mapping between the training samples and their targets, i.e. overfitting.
#
# This `validation_data` argument in `fit()` takes Numpy arrays or TensorFlow dataset objects. 

# %% [markdown] id="6qKyTTLMt1JI"
# **Using the `validation_data` argument**

# %% id="pxKjs2ixt1JJ"
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

#To avoid having samples from only one class of data, we shuffle the inputs using a random permutation
indices_permutation = np.random.permutation(len(inputs)) 
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

# This block reserves 30% of the inputs for validation purposes
num_validation_samples = int(0.3 * len(inputs))   
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]

#We make our model using training data that is used to update the model weights, and validation data that is used only to monitor the validation loss/metrics
model.fit(
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=16,
    validation_data=(val_inputs, val_targets)
)

# %% [markdown]
# The value of the loss on the validation data is called the _validation loss_ to distinguish it from the training loss. If you want to compute the validation loss and metrics after the training is complete, you can call the `evaluate()` method:

# %%
loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size = 128)

# %% [markdown]
# `evaluate()` will iterate in batches (of size = `batch_size`) over the data passed and returns a list of scalars, whre the first entry is the validation loss and the following entries are validation metrics. If the model has no metrics, only the validation loss is returned (instead of a list).

# %% [markdown] id="jHFisLlNt1JJ"
# ### Inference: Using a model after training
#
# Once you've trained your model, you want to use it to make predictions on new data; this is called _inference_. A naive way approach would be to simply call:
#
# `predictions = model(new_inputs)`
#
# However, this will process all inputs in `new_inputs` at once which may not be feasible if you're looking at a lot of data (you may not have enough VRAM on your GPU).

# %% [markdown]
# A better way to do inference is to use the `predict()` method. It will iterate over the data in small batches and return a NumPy array of predictions. And unlike `call()`, it can also process TensorFlow `Dataset` objects.
#
# We can use `predict()` on some of our validation data from the linear model we trained earlier. We get scalar scores that correspond to the model's prediction for each input sample. 

# %% id="ymjuUVb_t1JJ"
predictions = model.predict(val_inputs, batch_size=128)
print(predictions[:10])

# %% [markdown] id="9ZU3JQPjt1JJ"
# ## Summary

# %% [markdown]
# We learned the basics of how to train a model and perform inference using Keras.

# %%
