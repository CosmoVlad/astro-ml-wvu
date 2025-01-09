## Overview
This is a collection of Jupyter notebooks from machine-learning (ML) coworking sessions held by the members of [Prof. Sean McWilliam](https://physics.wvu.edu/directory/faculty/sean-mcwilliams)'s group at West Virginia University. 
Participants (alphabetical):
- Anuj Kankani,
- Jordan O'Kelley,
- Matthew Cerep,
- Sean McWilliams,
- Siddharth Mahesh,
- Suchindram Dasgupta,
- Vladimir Strokov (discussion leader).

The .py files are [jupytext](https://jupytext.readthedocs.io/en/latest/) versions of Jupyter notebooks. `jupytext` prevents Git from tracking any changes in outputs of a notebook and can be installed together with other necessary packages (see below). 
- to open a jupytext-enabled .py file, right-click on it in a Jupyter server and choose Open With -> Notebook;
- to generate a .py version of a new notebook, choose File -> Jupytext -> Pair Notebook with Percent Format (the Jupytext menu should be there if `jupytext` has been installed).

## Getting started
### Resources about machine learning (ML)

Some ways to get started with ML:
- Coursera:
  - Andrew Ng's courses/specializations, e.g. *Machine Learning*, *Neural Networks and Deep Learning*, *Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization*.
  - [DeepLearning.ai](deeplearning.ai)
- YouTube:
  - e.g. [TensorFlow tutorials](https://www.youtube.com/watch?v=5Ym-dOS9ssA&list=PLhhyoLH6IjfxVOdVC1P1L5z5azs0XjMsb)
- Books:
  - e.g. [About keras and tensorflow (keras is now a part of tensorflow)](https://github.com/manjunath5496/Best-Keras-Books-of-All-Time/blob/master/README.md)
  - François Chollet's [*Deep Learning with Python*](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff)

### Preparing a python environment

Different projects may require different sets of python packages with different versions. A good practice is to confine each project to its own python environment. It is also helpful that any software issues within an environment do not affect the system-wide python installation. That is, *what happens in a python environment stays in the python environment*.

#### Linux/Mac

To create an environment with some name (I use *testml* below) in a directory: 
```
cd directory/
python3 -m venv testml
```
To activate the environment and install a set of basic packages inside it:
```
source testml/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install numpy scipy matplotlib jupyter jupytext
```
To launch a `jupyter` notebook from within the environment:
```
jupyter-notebook
```
To deactivate the environment:
```
deactivate
```

#### Windows

1. To create a venv:
```
python -m venv C:\path\to\new\virtual\environment
```
For example, if we want to create a venv called *testml* in the current location,
```
python -m venv testml
```
2. To activate the venv:
```
C:\path\to\new\virtual\environment\Scripts\activate
```
To activate the venv from our previous example,
```
testml\Scripts\activate
```
When the venv is activated, its name appears in the beginning of the command line:
```
(testml) C:\>
```
3. To install packages,
```
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib jupyter jupytext
```
   - Any additional packages can be installed by using `python -m pip install` followed by the name of a package.
4. To launch a jupyter notebook,
```
jupyter notebook
```
If that doesn't work, try
```
python -m notebook
```
5. To deactivate the venv,
```
deactivate
```
