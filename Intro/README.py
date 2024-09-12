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
# - YouTube:
#   - e.g. [TensorFlow tutorials](https://www.youtube.com/watch?v=5Ym-dOS9ssA&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb)
# - Books:
#   - e.g. [About keras and tensorflow (keras is now a part of tensorflow)](https://github.com/manjunath5496/Best-Keras-Books-of-All-Time/blob/master/README.md)

# %% [markdown]
# ### Preparing a python environment
#
# Different projects may require different sets of python packages with different versions. A good practice is to confine each project to its own python environment. It is also helpful that any software issues within an environment do not affect the system-wide python installation. That is, *what happens in a python environment stays in the python environment*.
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

# %%
