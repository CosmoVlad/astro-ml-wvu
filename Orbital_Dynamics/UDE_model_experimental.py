# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Here we will start reproducing the paper [arXiv: 2102.12695](https://arxiv.org/pdf/2102.12695) on learning the GR orbital dynamics from gravitational-wave signal.

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
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

fig.tight_layout()
fig.savefig('test.pdf')

# %% [markdown]
# ## Paper summary
#
# ### Flowchart
#
# ![image](./arXiv210212695/Flowchart2.png)
#
# ### Equations
#
# #### Eqs. (5)
# \begin{eqnarray}
# 	\dot \phi
# 	&=&
# 	\frac{(1+e\cos(\chi))^2}{Mp^{3/2}}
# 	\left(
# 		1 + \mathcal{F}_1(\cos(\chi),p,e)
# 	\right),\\
# 	\dot \chi
# 	&=&
# 	\frac{(1+e\cos(\chi))^2}{Mp^{3/2}}
# 	\left(
# 		1 + \mathcal{F}_2(\cos(\chi),p,e)
# 	\right),\\
# 	\dot p
# 	&=&
# 	\mathcal{F}_3(p,e), \\
# 	\dot e
# 	&=&
# 	\mathcal{F}_4(p,e),
# \end{eqnarray}
# with $\mathbf{x}(0) = (\phi_0,\chi_0,p_0,e_0)$.
#
# #### Eqs. (7)
# \begin{eqnarray}
#     r(t) &=& p(t) M / (1+e(t) \cos\chi(t)) \,, \\
# 	\mathbf{r}_1(t)
# 	&=&
# 	\frac{r(t) m_2}{M}\left(\cos(\phi(t)),\sin(\phi(t)), 0 \right) \,,
# 	\\	
#     \mathbf{r}_2(t)
# 	&=&
# 	-\frac{r(t) m_1}{M}\left(\cos(\phi(t)),\sin(\phi(t)), 0 \right) 
# 	\,.
# \end{eqnarray}
#
# #### Eqs. (8)
# $$
# h^{22}(t) =\dfrac{1}{r}\sqrt{\dfrac{4\pi}{5}}\left(\ddot{\mathcal{I}}_{xx}
# - 2 i\ddot{\mathcal{I}}_{xy}-\ddot{\mathcal{I}}_{yy}\right)\,
# $$
#
# #### Eqs. (A10) and (A11)
# $$
# {\mathcal{I}}^{ab} 
# = I^{ab} - \frac{1}{3} \delta^{ab}\delta_{cd}I^{cd} \,,
# $$
# $\delta^{ab}$ is the Kronecker delta. 
#
# \begin{eqnarray}
# 	I^{xx} & =& \int d^3x \rho x^2 = m_1 x_1(t)^2 + m_2 x_2(t)^2  \,, \\	
#     I^{yy} & =& \int d^3x \rho y^2 = m_1 y_1(t)^2 + m_2 y_2(t)^2 \,, \\
# 	I^{xy} & =& \int d^3x \rho xy = m_1 x_1(t)y_1(t) + m_2 x_2(t)y_2(t) \,,
# \end{eqnarray}
# and by symmetry $I^{xy} = I^{yx}$.
#
# #### Eq. (9)
# \begin{equation}
# 	\mathcal{J}(\mathbf{x})
# 	=
# 	\langle J(\mathbf{x},\cdot) \rangle
# 						=
# 	\frac{1}{T}
# 	\int_{0}^{T}
# 			J(\mathbf{x},t)
# 	\,
# 	\mathrm{d}t
# 	,
# \end{equation}
# where $J(\mathbf{x},t) = \sum_{k} \big( w_k - w(t) \big)^2\delta(t-t_k)$
# and bracket notation, $\langle \cdot \rangle$, denotes 
# denotes averaging over the time interval.

# %%
"""Defines neural network layers for a Universal Differential Equation (UDE) model.

This module contains:
    - Fblock: A feedforward block with three dense layers.
    - UDEcell: A neural network-based ODE solver.
    - UDE: A full neural network-based differential equation solver.
"""

from typing import List

#Todo: compile the documentation to create a github.io page for the repo!
class Fblock(keras.layers.Layer):
    """Feedforward block with three dense layers."""

    def __init__(self, units_list: List[int]) -> None:
        """Initializes the Fblock layer.

        Args:
            units_list (List[int]): List of three integers specifying the number of units in each dense layer.
        """
        super().__init__()
        if len(units_list) != 3:
            raise ValueError("units_list must have exactly three elements.")
        self.dense1 = layers.Dense(units_list[0], activation='tanh')
        self.dense2 = layers.Dense(units_list[1], activation='tanh')
        self.dense3 = layers.Dense(units_list[2], activation='tanh')

    def call(self, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass for the Fblock layer.

        Args:
            input_tensor (tf.Tensor): Input tensor.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Output tensor after passing through the three dense layers.
        """
        x = self.dense1(input_tensor)
        x = self.dense2(x)
        x = self.dense3(x)
        x0 , x1 = tf.unstack(x,axis = -1)
        x2 = tf.zeros_like(x0)
        if x0.shape != x2.shape:
            print(x)
            print(x0)
            print(x2)
        return tf.transpose(tf.convert_to_tensor([x1,x1,x2,x2]))
        #return x
        

class UDEcell(keras.layers.Layer):
    """Runge-Kutta single-step evolver for neural network-based ODEs."""

    def __init__(self, partial_units_list: List[int], timestep: float, q: float) -> None:
        """Initializes the UDEcell layer.

        Args:
            partial_units_list (List[int]): List of integers specifying the hidden layer sizes.
            timestep (float): Time step for numerical integration.
            q (float): Mass ratio of the system.
        """
        super().__init__()
        self.timestep = timestep
        self.partial_units_list = partial_units_list
        self.q = q

    def build(self, input_shape: tf.TensorShape) -> None:
        """Builds the Fblock using the input shape.

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor.
        """
        #testing purely to evolve chi and phi
        units_list = [*self.partial_units_list, 2]
        #units_list = [*self.partial_units_list, 4]
        self.fblock = Fblock(units_list)

    def call(self, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass for UDEcell using a Runge-Kutta solver.

        Args:
            input_tensor (tf.Tensor): Input batch of primitive variables.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Batch of primitive variables numerically evolved by one timestep.
        """
        
        k1 = self.fiducial_rhs(input_tensor)
        k2 = self.fiducial_rhs(input_tensor + self.timestep * k1 / 2.0)
        k3 = self.fiducial_rhs(input_tensor + self.timestep * k2 / 2.0)
        k4 = self.fiducial_rhs(input_tensor + self.timestep * k3)
        dy = self.timestep / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        k1 = self.q*self.fblock(input_tensor)
        k2 = self.q*self.fblock(input_tensor + self.timestep * k1 / 2.0)
        k3 = self.q*self.fblock(input_tensor + self.timestep * k2 / 2.0)
        k4 = self.q*self.fblock(input_tensor + self.timestep * k3)
        dy_correction = self.timestep / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        return dy + dy_correction
    #Todo: use a decorator so that it can be called outside of the class! 
    def fiducial_rhs(self,y: tf.Tensor) -> tf.Tensor:
        """Fiducial(Newtonian) right hand side for UDE.

        Args:
            y (tf.Tensor): Input batch of primitive variables.

        Returns:
            tf.Tensor: Output batch of Newtonian right hand sides
        """
        phi,chi,p,e = tf.unstack(y, axis=-1)
        phi_dot = (1 + e*tf.math.cos(chi))**2 / p**1.5
        chi_dot = (1 + e*tf.math.cos(chi))**2 / p**1.5
        p_dot = tf.zeros(tf.shape(phi_dot))
        e_dot = tf.zeros(tf.shape(phi_dot))
        return tf.transpose(tf.convert_to_tensor([phi_dot,chi_dot,p_dot,e_dot]))
        

class UDE(keras.Model):
    """Neural network-based universal differential equation solver."""

    def __init__(
        self,
        partial_units_list: List[int],
        num_step: int,
        timestep: float,
        q: float,
        mean: float,
        stdev: float,
        use_real: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the UDE model.

        Args:
            partial_units_list (List[int]): Hidden layer sizes for Fblock.
            num_step (int): Number of integration steps.
            timestep (float): Time step for numerical integration.
            q (float): Mass ratio given by m_smaller/m_larger.
            mean (float): Global mean of the waveform.
            stdev (float): Global standard deviation of the waveform.
            use_real (bool, optional): Whether to use real part of the waveform. Defaults to True.
            **kwargs: Additional arguments for keras.Model.
        """
        super().__init__(**kwargs)
        self.timestep = timestep
        self.num_step = num_step
        self.q = q
        self.mean = mean
        self.stdev = stdev
        self.use_real = use_real
        self.udecell = UDEcell(partial_units_list, timestep, q)

    def call(self, init_conditions_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Integrate UDE and returns the waveform.

        Args:
            init_conditions_tensor (tf.Tensor): Initial condition tensor.
            training (bool, optional): Whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: Normalized waveform data.
        """
        y = init_conditions_tensor
        sol = [y]
        for _ in range(1,self.num_step):
            dy = self.udecell(y, training=training)
            y += dy
            sol.append(y)
        sol = tf.convert_to_tensor(sol)
        self.sol = sol
        #phi, chi, p, e = tf.unstack(sol, axis=-1)


        waveform = self.h22(sol)
        if self.use_real:
            real_part = tf.math.real(waveform)
            return tf.transpose((real_part - self.mean) / self.stdev)
        imag_part = tf.math.imag(waveform)
        return tf.transpose((imag_part - self.mean) / self.stdev)
        
    def h22(
        self,
        dynamics: tf.Tensor
    ) -> tf.Tensor:
        """Compute the waveform.

        Args:
            dynamics (tf.Tensor): Batch of integrated UDE solutions.
            
        Returns:
            tf.Tensor: Normalized batch of strains (real part if self.use_real is True, complex otherwise).
        """

        phi,chi,p,e = tf.unstack(dynamics, axis=-1)
        r = p / (1 + e*tf.math.cos(chi))
        x1 = r * self.q/(1+self.q) * tf.math.cos(phi)
        y1 = r * self.q/(1+self.q) * tf.math.sin(phi)
        x2 = -r * 1/(1+self.q) * tf.math.cos(phi)
        y2 = -r * 1/(1+self.q) * tf.math.sin(phi)
        Ixx = x1**2 + self.q*x2**2
        Iyy = y1**2 + self.q*y2**2
        Ixy = x1*y1 + self.q*x2*y2
        trace = Ixx + Iyy
        r = p / (1 + e*tf.math.cos(chi))
        Jxx = Ixx - trace/3
        Jyy = Iyy - trace/3
        ddJxx = (Jxx[2:] - 2*Jxx[1:-1] + Jxx[:-2]) / self.timestep**2
        ddJyy = (Jyy[2:] - 2*Jyy[1:-1] + Jyy[:-2]) / self.timestep**2
        const = 1/r[1:-1] * tf.math.sqrt(4*np.pi/5)
        real_part = const * (ddJxx - ddJyy)
        if self.use_real:
            return real_part
        Jxy = Ixy
        ddJxy = (Jxy[2:] - 2*Jxy[1:-1] + Jxy[:-2]) / self.timestep**2
        imag_part = const * (- 2*ddJxy)
        return tf.complex(real_part, imag_part)
    
    def change_num_step(
        self,
        num_step: float,
    ) -> None:
        """
        Change the number of timesteps over which RK integration is done.

        Args:
            num_step (float): Number of timesteps.
        """
        self.num_step = num_step

test_UDE_infrastructure = True
if test_UDE_infrastructure:
    phi0 = 0.
    chi0 = np.pi
    p0 = 100.
    e0 = 0.5
    tinit = 0.
    tfin = 1e+4
    num_tsteps = 51
    times = np.linspace(tinit, tfin, num_tsteps)
    dt = times[1] - times[0]
    y0 = np.array(
        [phi0,chi0,p0,e0]
    )
    batch_size = 17
    input_tensor = np.repeat([y0], batch_size, axis=0)
    timestep = times[1] - times[0]
    q = 0.01
    partial_units_list = [4,4]
    model = UDE(
        partial_units_list=partial_units_list,
        num_step=num_tsteps,
        timestep=timestep,
        q=q,
        mean=0,
        stdev=1,
    )
    waveform = model(input_tensor)
    print(f"Expected input shape: ({batch_size},4), Actual input shape = {input_tensor.shape}")
    print(f"Expected output shape: ({batch_size},{num_tsteps-2}), Actual output shape = {waveform.shape}")
    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))
    for wf in waveform:
        ax.plot(times[1:-1], wf)
    #ax.plot(x1,y1)
    #ax.plot(x2,y2)
    
    ax.grid(True,linestyle=':',linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)
    
    # ax.set_xlabel('time (s)')
    # ax.set_ylabel('$h_{22}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


# %%
class GR():
    """General relativity waveform solver."""

    def __init__(
        self,
        num_step: int,
        timestep: float,
        q: float,
        use_real: bool = True,
    ) -> None:
        """Initializes the GR model.

        Args:
            num_step (int): Number of integration steps.
            timestep (float): Time step for numerical integration.
            q (float): Mass ratio given by m_smaller/m_larger.
            use_real (bool, optional): Whether to use real part of the waveform. Defaults to True.
        """
        self.timestep = timestep
        self.num_step = num_step
        self.q = q
        self.use_real = use_real

    def __call__(self, init_conditions_tensor: tf.Tensor) -> tf.Tensor:
        """Integrate UDE and returns the waveform.

        Args:
            init_conditions_tensor (tf.Tensor): Initial condition tensor.

        Returns:
            tf.Tensor: Normalized waveform data.
        """
        y = init_conditions_tensor
        sol = [y]
        for _ in range(1,self.num_step):
            y += self.evolve(y)
            sol.append(y)
        sol = tf.convert_to_tensor(sol)
        self.sol = sol
        #phi, chi, p, e = tf.unstack(sol, axis=-1)


        waveform = self.h22(sol)
        if self.use_real:
            real_part = tf.math.real(waveform)
            mean = 0#tf.math.reduce_mean(real_part, axis=0, keepdims=True)
            std = 1#tf.math.reduce_std(real_part, axis=0, keepdims=True)
            return tf.transpose(real_part)
        imag_part = tf.math.imag(waveform)
        #mean = tf.math.reduce_mean(imag_part, axis=0, keepdims=True)
        #std = tf.math.reduce_std(imag_part, axis=0, keepdims=True)
        return tf.transpose(imag_part)
        
    def GR_rhs(self,y: tf.Tensor) -> tf.Tensor:
        """GR right hand side for ODE.

        Args:
            y (tf.Tensor): Input batch of primitive variables.

        Returns:
            tf.Tensor: Output batch of Newtonian right hand sides
        """
        phi,chi,p,e = tf.unstack(y, axis=-1)
        phi_dot = (1 + e*tf.math.cos(chi))**2 / p**1.5 * (p - 2 - 2*e*tf.math.cos(chi)) / tf.math.sqrt((p-2)**2 - 4*e**2)
        chi_dot = (1 + e*tf.math.cos(chi))**2 / p**2 * (p - 2 - 2*e*tf.math.cos(chi)) *\
                    tf.math.sqrt( (p - 6 - 2*e*tf.math.cos(chi)) / ((p-2)**2 - 4*e**2))
        p_dot = tf.zeros(shape=p.shape,dtype=p.dtype)
        e_dot = tf.zeros(shape=e.shape,dtype=e.dtype)
        return tf.transpose(tf.convert_to_tensor([phi_dot,chi_dot,p_dot,e_dot]))
        
    def evolve(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Evolve using a Runge-Kutta solver.

        Args:
            input_tensor (tf.Tensor): Input batch of primitive variables.
            
        Returns:
            tf.Tensor: Batch of primitive variables numerically evolved by one timestep.
        """
        
        k1 = self.GR_rhs(input_tensor)
        k2 = self.GR_rhs(input_tensor + self.timestep * k1 / 2.0)
        k3 = self.GR_rhs(input_tensor + self.timestep * k2 / 2.0)
        k4 = self.GR_rhs(input_tensor + self.timestep * k3)
        dy = self.timestep / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        return dy
    
    def h22(
        self,
        dynamics: tf.Tensor
    ) -> tf.Tensor:
        """Compute the waveform.

        Args:
            dynamics (tf.Tensor): Batch of integrated UDE solutions.
            
        Returns:
            tf.Tensor: Normalized batch of strains (real part if self.use_real is True, complex otherwise).
        """

        phi,chi,p,e = tf.unstack(dynamics, axis=-1)
        r = p / (1 + e*tf.math.cos(chi))
        x1 = r * self.q/(1+self.q) * tf.math.cos(phi)
        y1 = r * self.q/(1+self.q) * tf.math.sin(phi)
        x2 = -r * 1/(1+self.q) * tf.math.cos(phi)
        y2 = -r * 1/(1+self.q) * tf.math.sin(phi)
        Ixx = x1**2 + self.q*x2**2
        Iyy = y1**2 + self.q*y2**2
        Ixy = x1*y1 + self.q*x2*y2
        trace = Ixx + Iyy
        r = p / (1 + e*tf.math.cos(chi))
        Jxx = Ixx - trace/3
        Jyy = Iyy - trace/3
        ddJxx = (Jxx[2:] - 2*Jxx[1:-1] + Jxx[:-2]) / self.timestep**2
        ddJyy = (Jyy[2:] - 2*Jyy[1:-1] + Jyy[:-2]) / self.timestep**2
        prefac = tf.math.sqrt(tf.constant(4.*np.pi/5.,dtype=tf.float64))
        const = 1/r[1:-1] * prefac
        real_part = const * (ddJxx - ddJyy)
        if self.use_real:
            return real_part
        Jxy = Ixy
        ddJxy = (Jxy[2:] - 2*Jxy[1:-1] + Jxy[:-2]) / self.timestep**2
        imag_part = const * (- 2*ddJxy)
        return tf.complex(real_part, imag_part)

test_GR_infrastructure = True
if test_GR_infrastructure:
    batch_size = 100
    model_sample = np.linspace(0.,1.,batch_size)
    phi0 = 2*np.pi*model_sample
    np.random.shuffle(phi0)
    chi0 = 2*np.pi*model_sample
    np.random.shuffle(chi0)
    p0 = 50. + 100.*model_sample
    np.random.shuffle(p0)
    e0 = 0.5*model_sample
    np.random.shuffle(e0)
    q = 0.01
    tinit = 0.
    #select tfin so that at least 1 orbit is sampled in each case
    a = p0 / (1 - e0**2)
    Tkep = 2 * np.pi * np.sqrt(a**3)
    tfin = Tkep.max()
    len_sol = 252
    times = np.linspace(tinit, tfin, len_sol)
    dt = times[1] - times[0]
    y0 = np.c_[phi0,chi0,p0,e0]
    input_tensor = tf.convert_to_tensor(y0)
    model_GR = GR(
        num_step=len_sol,
        timestep=dt,
        q=q
    )
    GR_waveform = model_GR(input_tensor)
    sol = model_GR.sol
    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))
    true_wf = GR_waveform[0]
    ax.plot((times[1:-1]),true_wf)
    ax.scatter((times[1:-1])[::1],true_wf[::1], s=10, color='black')
    ax.grid(True,linestyle=':',linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    
    #ax.set_aspect('equal')    

# %%
single_traj_training = False
if single_traj_training:
    rng = np.random.default_rng()
    gr_dynamics = sol[:,0]
    gr_waveforms = GR_waveform[0]
    gr_mean = tf.math.reduce_mean(gr_waveforms)
    gr_stdev = tf.math.reduce_std(gr_waveforms)
    gr_waveforms_norm = (gr_waveforms - gr_mean)/gr_stdev
    orbit_len = len(gr_dynamics)
    slice_len = min(10,orbit_len//5) #20% of orbit length
    train_size = 100
    indices = rng.integers(low=0, high=orbit_len-slice_len, size=train_size)
    x_train = tf.gather(gr_dynamics, indices)
    y_train = tf.stack([gr_waveforms_norm[i:i+slice_len-2] for i in indices])
    #print(y_train.shape)
    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))
    
    ax.plot((times[1:-1]),gr_waveforms_norm)
    for h_slice,i in zip(y_train,indices):
        ax.scatter((times[i:i+slice_len-2]),h_slice, s=10,)
    
    ax.grid(True,linestyle=':',linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

else:
    orbit_len = len(sol)
    slice_len = (orbit_len//10) #~10% of orbit length
    num_cases = 10
    train_size = batch_size*num_cases
    rng = np.random.default_rng()
    gr_mean = tf.math.reduce_mean(GR_waveform)
    gr_stdev = tf.math.reduce_std(GR_waveform)
    gr_waveforms_norm = (GR_waveform - gr_mean)/gr_stdev
    orbit_len = len(sol)
    indices = rng.integers(low=0, high=orbit_len-slice_len, size=num_cases)
    x_train_noflat = tf.gather(sol,indices)
    x_train = tf.reshape(x_train_noflat,(train_size,4))
    y_train_noflat = tf.convert_to_tensor([gr_waveforms_norm[:,i:i+slice_len-2] for i in indices])
    y_train = tf.reshape(y_train_noflat,(train_size,slice_len - 2))
    print(y_train_noflat.shape)
    print(y_train.shape)
    print(y_train[3])
    fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))
    i_plot = rng.integers(low=0,high=batch_size)
    ax.plot((times[1:-1]),gr_waveforms_norm[i_plot])
    for i in range(len(indices)):
        h_slice = y_train[(i)*batch_size + i_plot]
        ax.scatter((times[indices[i]:indices[i]+slice_len-2]),h_slice, s=10,)
    
    ax.grid(True,linestyle=':',linewidth='1.')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')


# %%
timestep = times[1] - times[0]
q = 0.01
partial_units_list = [12,25]
gr_mean = tf.cast(gr_mean,dtype=tf.float32)
gr_stdev = tf.cast(gr_stdev,dtype=tf.float32)
model = UDE(partial_units_list,slice_len,model_GR.timestep,q,gr_mean,gr_stdev)
y_pred = model(x_train)
print(y_pred.shape)
i_plot = rng.integers(low=0,high=batch_size)
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))
ax.plot((times[1:-1]),gr_waveforms_norm[i_plot])
for i in range(len(indices)):
    idx = indices[i]
    h_slice = y_pred[i*batch_size + i_plot]
    ax.scatter((times[idx:idx+slice_len-2]),h_slice, s=10,)
ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')


# %%
model.change_num_step(slice_len)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5,clipnorm=.01),
    loss=keras.losses.MeanSquaredError(),
)

# training
history = model.fit(
    x_train, y_train, 
    batch_size=100, epochs=300, validation_split=0.2, verbose=1,
)

# %%
quality_dict = history.history
plt.semilogy(quality_dict['loss'])
plt.semilogy(quality_dict['val_loss'])

# %%
phi0_p = 0.
chi0_p = np.pi
p0_p = rng.uniform(50.,150.)
e0_p = rng.uniform(0.,0.5)
#q_p = 0.01*(1 + .01)

y_0 = np.array(
    [phi0_p,chi0_p,p0_p,e0_p]
)

input_tensor_p = np.array(
    [[phi0_p,chi0_p,p0_p,e0_p]]
)

ground_truth_waveform = (tf.cast(model_GR(input_tensor_p),dtype = tf.float32)[0] - tf.cast(gr_mean,dtype = tf.float32))/tf.cast(gr_stdev,dtype = tf.float32)

model.change_num_step(model_GR.num_step)
waveform_ude = model.predict(input_tensor_p)
predicted_waveform = waveform_ude[0]
step_size = model.timestep
step_count = model.num_step
#times = np.array([i*step_size for i in range(step_count)])
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))
#print(waveform)

ax.plot((times[1:-1]),ground_truth_waveform)
ax.scatter((times[1:-1]),predicted_waveform, s=10,)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
<<<<<<< HEAD

# %%
model.save("sids_emri_training_model.keras")

# %% [markdown]
# ### Strategies:
#
# 1. $q$ -> This is only used for waveform generation so make this a model parameter in the GR and UDE classes
# 2. Make the NN rhs function of cosine chi instead of chi.
# 3. Add a function to change the number of timesteps in both GR and UDE classes.
# 4. GEnerate GR data for multiple mass ratios
# 5. Instead of using normalized waveforms, try using the mean squared percentage error so that predicted waveforms have the correct scaling.

# %%
init_cell = UDEcell(partial_units_list, timestep)

init_state = tf.convert_to_tensor([y0], dtype=tf.float32)

units_list = [*partial_units_list, 4]
fblock_rhs = Fblock(units_list)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(init_state)
    total_rhs = fiducial_rhs(init_state) + fblock_rhs(init_state)
    final_rhs = total_rhs[0]

init_jac = tape.jacobian(final_rhs, init_state)
init_jac = tf.squeeze(init_jac, axis=1)

init_jac.shape

# %%
tf.linalg.eigvals(init_jac)

# %%
timestep

# %%

# %%
=======
>>>>>>> f5e05cfdd3eb81a7f9037e1550e877d0ab4d7ea1
