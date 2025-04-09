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
    p_dot = tf.zeros(tf.shape(phi_dot), dtype=y.dtype)
    e_dot = tf.zeros(tf.shape(phi_dot), dtype=y.dtype)

    return tf.transpose(tf.convert_to_tensor([phi_dot,chi_dot,p_dot,e_dot], dtype=y.dtype))   # (None, 4)

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

    print(type(r))

    Jxx = Ixx - trace/3
    Jxy = Ixy
    Jyy = Iyy - trace/3

    ddJxx = (Jxx[2:] - 2*Jxx[1:-1] + Jxx[:-2]) / dt**2
    ddJxy = (Jxy[2:] - 2*Jxy[1:-1] + Jxy[:-2]) / dt**2
    ddJyy = (Jyy[2:] - 2*Jyy[1:-1] + Jyy[:-2]) / dt**2

    const = 1/r[1:-1] * tf.math.sqrt(4*tf.constant(np.pi, dtype=r.dtype)/5)
    real_part = const * (ddJxx - ddJyy)
    imag_part = const * (- 2*ddJxy)

    return tf.complex(real_part, imag_part)  # (num_tsteps, None)


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
dtype = tf.float64

phi0 = 0.
chi0 = np.pi
p0 = 100.
e0 = 0.5

q = 0.01


tinit = 0.
tfin = 1e+4

times = np.linspace(tinit, tfin, 200)
dt = times[1] - times[0]

y0 = np.array(
    [phi0,chi0,p0,e0]
)

sol = odeint(GR_rhs, y0, times)   # (num_steps, 4)
sol = tf.convert_to_tensor(sol, dtype=dtype)

phi,chi,p,e = tf.unstack(sol, axis=-1)

true_wf = tf.math.real(h22(phi,chi,p,e, q, dt))
mean = tf.math.reduce_mean(true_wf)
std = tf.math.reduce_std(true_wf)

true_wf = (true_wf - mean) / std


fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot((times[1:-1]),true_wf)
ax.scatter((times[1:-1]),true_wf, s=5, color='black')

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

#ax.set_aspect('equal')

# %%
rng = np.random.default_rng()

def generate_slices(orbit_len, slice_len_traj, train_size=10000):

    traj_indices = rng.integers(low=0, high=orbit_len-slice_len_traj, size=train_size)
    
    # N - slice_len - 1 + slice_len - 2 = N - 3
    
    x_train = tf.gather(sol, traj_indices)
    y_train = tf.stack([true_wf[i:i+slice_len_traj-2] for i in traj_indices])

    return x_train,y_train,traj_indices

slice_len_traj = 50
orbit_len = len(sol)

x_train,y_train,traj_indices = generate_slices(orbit_len, slice_len_traj=slice_len_traj)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

################### plotting ###############

ax.plot((times[1:-1]),true_wf, zorder=0)

for h_slice,i in zip(y_train[:10], traj_indices[:10]):
    ax.scatter(times[1:-1][i:i+slice_len_traj-2], h_slice, s=10, zorder=1)

print(h_slice.shape)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time')
ax.set_ylabel('${\\rm Re\,}h_{22}$')


# %%
class WaveformGRU(keras.Model):
    def __init__(self, units, num_steps, **kwargs):
        super(WaveformGRU, self).__init__(**kwargs)
        self.units = units
        self.num_steps = num_steps
        self._dtype = kwargs.get('dtype', tf.float32)
        
        self.gru_cell = layers.GRUCell(units, dtype=self._dtype)
        self.wf_dense = layers.Dense(1, dtype=self._dtype)

    def build(self, input_dim):
        super(WaveformGRU, self).build(input_dim)
        self.traj_dense = layers.Dense(input_dim[-1], dtype=self._dtype)

    def call(self, input_tensor, training=False):
        
        y_output = []
        batch_size = tf.shape(input_tensor)[0]
        state = tf.zeros(
            shape=(batch_size, self.units)
        )
        
        output, state = self.gru_cell(input_tensor, state)
        waveform_output = self.wf_dense(output)
        output = self.traj_dense(output)
        y_output = [waveform_output]
        
        for i in range(self.num_steps-1):
            output, state = self.gru_cell(output, state)
            waveform_output = self.wf_dense(output)
            output = self.traj_dense(output)
            y_output.append(waveform_output)
        
        y_output = tf.stack(y_output, axis=1)  # Shape: (batch_size, num_steps, 1)
        return tf.squeeze(y_output, axis=-1)  # Shape: (batch_size, num_steps)

units=128
batch_size=32

x_batch = x_train[:batch_size]
y_batch = y_train[:batch_size] # (batch, slice_len)
_,num_steps = y_batch.shape

model = WaveformGRU(units, num_steps)

y_output = model(x_batch)
y_output.shape

# %%
# timestep = times[1] - times[0]

# q = 0.01
# partial_units_list = [32,32]

# model = UDE(partial_units_list, slice_len, timestep, q, mean, std)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanSquaredError(),
)

# training
history = model.fit(
    x_train, y_train, 
    batch_size=128, epochs=200, validation_split=0.2, verbose=1,
)

# %%
val_acc = history.history['val_loss']
train_acc = history.history['loss']

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.semilogy(1 + np.arange(len(val_acc)), train_acc, label='training')
ax.semilogy(1 + np.arange(len(val_acc)), val_acc, label='validation')

ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)


ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

################### plotting ###############

ax.plot((times[1:-1]),true_wf, zorder=0)


y_pred = model(x_train[10:12])

predict_len_traj = slice_len_traj

for h_slice,i in zip(y_pred, traj_indices[10:12]):
    ax.scatter(times[1:-1][i:i+predict_len_traj-2], h_slice, s=10, zorder=1)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time')
ax.set_ylabel('${\\rm Re} h_{22}$')

# %%
y_pred.shape, slice_len_traj

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

start_id = np.argmin(traj_indices)
x_start = x_train[start_id][tf.newaxis,...]

predict_len_traj = 1000
print(predict_len_traj)

################### plotting ###############

ax.plot((times[1:-1]),true_wf, zorder=0)


y_pred = model(x_start, num_steps=predict_len_traj)
print(y_pred.shape)

ax.scatter(times[:predict_len_traj][1:-1], y_pred[0], s=10, zorder=1)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time')
ax.set_ylabel('${\\rm Re} h_{22}$')

# %%
