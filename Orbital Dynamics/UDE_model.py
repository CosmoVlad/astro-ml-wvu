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
def fiducial_rhs(y):     # t -> t/M:   G=c=1
                        # M -> GM/c^3

    # y -> (None, 4)
    
    phi,chi,p,e = tf.unstack(y, axis=-1)

    phi_dot = (1 + e*tf.math.cos(chi))**2 / p**1.5
    chi_dot = (1 + e*tf.math.cos(chi))**2 / p**1.5
    p_dot = tf.zeros(tf.shape(phi_dot))
    e_dot = tf.zeros(tf.shape(phi_dot))

    return tf.transpose(tf.convert_to_tensor([phi_dot,chi_dot,p_dot,e_dot]))   # (None, 4)

def h22(phi,chi,p,e, q, dt):

    r = p / (1 + e*tf.math.cos(chi))    # (num_tsteps, None)

    x1 = r * q/(1+q) * tf.math.cos(phi)
    y1 = r * q/(1+q) * tf.math.sin(phi)

    x2 = -r * 1/(1+q) * tf.math.cos(phi)
    y2 = -r * 1/(1+q) * tf.math.sin(phi)

    Ixx = x1**2 + q*x2**2
    Iyy = y1**2 + q*y2**2
    Ixy = x1*y1 + q*x2*y2

    trace = Ixx + Iyy

    r = p / (1 + e*tf.math.cos(chi))

    Jxx = Ixx - trace/3
    Jxy = Ixy
    Jyy = Iyy - trace/3

    ddJxx = (Jxx[2:] - 2*Jxx[1:-1] + Jxx[:-2]) / dt**2
    ddJxy = (Jxy[2:] - 2*Jxy[1:-1] + Jxy[:-2]) / dt**2
    ddJyy = (Jyy[2:] - 2*Jyy[1:-1] + Jyy[:-2]) / dt**2

    const = 1/r[1:-1] * tf.math.sqrt(4*np.pi/5)
    real_part = const * (ddJxx - ddJyy)
    imag_part = const * (- 2*ddJxy)

    return tf.complex(real_part, imag_part)  # (num_tsteps, None)


# %%

class Fblock(keras.Layer):

    def __init__(self, units_list):
        super(Fblock, self).__init__()
        self.dense1 = layers.Dense(units_list[0], activation='relu')
        self.dense2 = layers.Dense(units_list[1], activation='relu')
        self.dense3 = layers.Dense(units_list[2])

    def call(self, input_tensor, training=False):

        x = self.dense1(input_tensor)
        x = self.dense2(x)
        x = self.dense3(x)

        return x

class UDEcell(keras.Layer):
    
    def __init__(self, partial_units_list, timestep):
        
        super(UDEcell, self).__init__()
        self.timestep = timestep
        self.partial_units_list = partial_units_list

    # partial_units_list = [32,32]
    # [*partial_units_list, 4]  -> [32,32,4]

    def build(self, input_dim):    # (None, 4)
        units_list = [*self.partial_units_list, input_dim[-1]]
        self.fblock = Fblock(units_list)

    def call(self, input_tensor, training=False):  # input_tensor -> (None, 4)


        k1 = fiducial_rhs(input_tensor)
        k2 = fiducial_rhs(input_tensor + self.timestep * k1/2.)
        k3 = fiducial_rhs(input_tensor + self.timestep * k2/2.)
        k4 = fiducial_rhs(input_tensor + self.timestep * k3)

        correction = self.timestep * self.fblock(input_tensor)
        #correction = 0.

        return self.timestep/6. * (k1 + 2*k2 + 2*k3 + k4) + correction   # dy

        

class UDE(keras.Layer):

    def __init__(self, partial_units_list, num_tsteps, timestep, q, use_real=True):

        super(UDE, self).__init__()
        self.num_tsteps = num_tsteps
        self.timestep = timestep
        self.q = q
        self.use_real = use_real

        self.udecell = UDEcell(partial_units_list,timestep)


    def call(self, init_conditions_tensor, training=False):

        # input_tensor -> (None, 4)
        # sol -> (None, 4, int(tinterval/timestep))

        # sol is obtained from integration(input_tensor)

        y = init_conditions_tensor
        sol = [y]
        
        for i in range(self.num_tsteps):

            y += self.udecell(y, training=training)
            sol.append(y)

        sol = tf.convert_to_tensor(sol)  # (num_tsteps, None, 4)

        ##########################################

        phi,chi,p,e = tf.unstack(sol, axis=-1)
        phi = tf.keras.ops.nan_to_num(phi, nan=100.)
        chi = tf.keras.ops.nan_to_num(chi, nan=100.)
        p = tf.keras.ops.nan_to_num(p, nan=100.)
        e = tf.keras.ops.nan_to_num(e, nan=100.)
        
        waveform = h22(phi,chi,p,e, self.q, self.timestep)    # (num_tsteps - 2, None)

        if self.use_real:
            real_part = tf.math.real(waveform)
            mean = tf.math.reduce_mean(real_part, axis=0, keepdims=True)
            std = tf.math.reduce_std(real_part, axis=0, keepdims=True)
            
            return tf.transpose((real_part - mean) / std)     # (None, num_tsteps - 2)

        imag_part = tf.math.imag(waveform)
        mean = tf.math.reduce_mean(imag_part, axis=0, keepdims=True)
        std = tf.math.reduce_std(imag_part, axis=0, keepdims=True)
        
        return tf.transpose((imag_part - mean) / std)     # (None, num_tsteps - 2)



# %%
# tinit = 0.
# tfin = 2*np.pi

# times = np.linspace(tinit, tfin, 100)

# y0 = np.array(
#     [[0.,0.,1.,0.2]]
# )

phi0 = 0.
chi0 = np.pi
p0 = 100.
e0 = 0.5


tinit = 0.
tfin = 6e+4

times = np.linspace(tinit, tfin, 252)
dt = times[1] - times[0]

y0 = np.array(
    [phi0,chi0,p0,e0]
)

batch_size = 32

input_tensor = np.repeat([y0], batch_size, axis=0)

timestep = times[1] - times[0]
num_tsteps = int(times[-1]/timestep)

input_tensor.shape

q = 0.01
partial_units_list = [32,32]

ude_layer = UDE(partial_units_list, num_tsteps, timestep, q)

test_wforms = np.array(ude_layer(input_tensor))

test_wforms.shape

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

for wf in test_wforms[:10]:
    ax.plot(times[1:-1], wf)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time (s)')
ax.set_ylabel('$h_{22}$')

# %%
from scipy.integrate import odeint

def GR_rhs(y,t):     # t -> t/M:   G=c=1
                        # M -> GM/c^3

    phi,chi,p,e = y

    phi_dot = (1 + e*np.cos(chi))**2 / p**1.5 * (p - 2 - 2*e*np.cos(chi)) / np.sqrt((p-2)**2 - 4*e**2)
    chi_dot = (1 + e*np.cos(chi))**2 / p**2 * (p - 2 - 2*e*np.cos(chi)) *\
                np.sqrt( (p - 6 - 2*e*np.cos(chi)) / ((p-2)**2 - 4*e**2))
    p_dot = 0.
    e_dot = 0.

    return np.array([phi_dot,chi_dot,p_dot,e_dot])


# %%
phi0 = 0.
chi0 = np.pi
p0 = 100.
e0 = 0.5

q = 0.01


tinit = 0.
#tfin = 2*np.pi * 100 * np.sqrt(p0**3)
tfin = 6e+4

times = np.linspace(tinit, tfin, 2000)
dt = times[1] - times[0]

y0 = np.array(
    [phi0,chi0,p0,e0]
)

sol = odeint(GR_rhs, y0, times)
sol = tf.convert_to_tensor(sol, dtype=tf.float32)

phi,chi,p,e = tf.unstack(sol, axis=-1)

true_wf = tf.math.real(h22(phi,chi,p,e, q, dt))
mean = tf.math.reduce_mean(true_wf)
std = tf.math.reduce_std(true_wf)

true_wf = (true_wf - mean) / std


fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot((times[1:-1]),true_wf)
ax.scatter((times[1:-1])[::8],true_wf[::8], s=10, color='black')

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

#ax.set_aspect('equal')

# %%
x_element = np.array(
    [phi0,chi0,p0,e0]
)
y_element = true_wf[::8]

train_size = 10000

x_train = tf.repeat([x_element], train_size, axis=0)
y_train = tf.repeat([y_element], train_size, axis=0)

x_train.shape, y_train.shape, type(x_train), type(y_train)


# %%
new_ude_layer = UDE(partial_units_list, 251, 7*dt, q)

new_ude_layer(x_train[:64]).shape

# %%
#tf.compat.v1.enable_eager_execution()

model = keras.Sequential(
    [new_ude_layer]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanSquaredError()
)

# training
history = model.fit(
    x_train, y_train, 
    batch_size=32, epochs=20, validation_split=0.2, verbose=1,
)

# %%
new_ude_layer = UDE(partial_units_list, 251, 7*dt, q)

new_ude_layer(x_train[:2])

# %%
