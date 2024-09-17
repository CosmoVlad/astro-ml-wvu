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
# ## Getting started

# %% [markdown]
# ### Resources about machine learning (ML)
#
# Some ways to get started with ML:
# - Coursera:
#   - Andrew Ng's courses/specializations, e.g. *Machine Learning*, *Neural Networks and Deep Learning*, *Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization*.
#   - [DeepLearning.ai](deeplearning.ai)
# - YouTube:
#   - e.g. [TensorFlow tutorials](https://www.youtube.com/watch?v=5Ym-dOS9ssA&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb)
# - Books:
#   - e.g. [About keras and tensorflow (keras is now a part of tensorflow)](https://github.com/manjunath5496/Best-Keras-Books-of-All-Time/blob/master/README.md)

# %% [markdown]
# ### Preparing a python environment
#
# Different projects may require different sets of python packages with different versions. A good practice is to confine each project to its own python environment. It is also helpful that any software issues within an environment do not affect the system-wide python installation. That is, *what happens in a python environment stays in the python environment*.
#
# #### Linux/Mac
#
# To create an environment with some name (I use *testml* below) in a directory: 
# ```
# cd directory/
# python3 -m venv testml
# ```
# To activate the environment and install a set of basic packages inside it:
# ```
# source testml/bin/activate
# pip install --upgrade pip
# pip install --upgrade setuptools
# pip install numpy scipy matplotlib jupyter jupytext
# ```
# To launch a `jupyter` notebook from within the environment:
# ```
# jupyter-notebook
# ```
# To deactivate the environment:
# ```
# deactivate
# ```
#
# #### Windows
#
# 1. To create a venv:
# ```
# python -m venv C:\path\to\new\virtual\environment
# ```
# For example, if we want to create a venv called *testml* in the current location,
# ```
# python -m venv testml
# ```
# 2. To activate the venv:
# ```
# C:\path\to\new\virtual\environment\Scripts\activate
# ```
# To activate the venv from our previous example,
# ```
# testml\Scripts\activate
# ```
# When the venv is activated, its name appears in the beginning of the command line:
# ```
# (testml) C:\>
# ```
# 3. To install packages,
# ```
# python -m pip install --upgrade pip
# python -m pip install numpy scipy matplotlib jupyter
# ```
#    - Any additional packages can be installed by using `python -m pip install` followed by the name of a package.
# 4. To launch a jupyter notebook,
# ```
# jupyter notebook
# ```
# If that doesn't work, try
# ```
# python -m notebook
# ```
# 5. To deactivate the venv,
# ```
# deactivate
# ```
#

# %% [markdown]
# ### Plan
#
# - installations;
# - run-of-the-mill perceptron:
#   - Problem (suggestion): range of a projectile as a function of $v_0$ and $\alpha$;
#   - Forward propagation

# %%
