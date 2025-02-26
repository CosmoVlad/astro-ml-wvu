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

    print(type(r))

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

    def __init__(self, units_list, **kwargs):
        super(Fblock, self).__init__(**kwargs)
        self.dense1 = layers.Dense(units_list[0], activation='tanh', kernel_initializer='zeros')
        self.dense2 = layers.Dense(units_list[1], activation='tanh', kernel_initializer='zeros')
        self.dense3 = layers.Dense(units_list[2], activation='tanh', kernel_initializer='zeros')

    def call(self, input_tensor, training=False):

        x = self.dense1(input_tensor, training=training)
        x = self.dense2(x, training=training)
        x = self.dense3(x, training=training)

        return x

class UDEcell(keras.Layer):
    
    def __init__(self, partial_units_list, timestep, **kwargs):
        
        super(UDEcell, self).__init__(**kwargs)
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

        dy = self.timestep/6. * (k1 + 2*k2 + 2*k3 + k4)

        k1 = self.fblock(input_tensor, training=training)
        k2 = self.fblock(input_tensor + self.timestep * k1/2., training=training)
        k3 = self.fblock(input_tensor + self.timestep * k2/2., training=training)
        k4 = self.fblock(input_tensor + self.timestep * k3, training=training)

        dy_correction = self.timestep/6. * (k1 + 2*k2 + 2*k3 + k4)

        return  dy + dy_correction

        

class UDE(keras.Model):

    def __init__(self, partial_units_list, timestep, q, mean, std, use_real=True, **kwargs):

        super(UDE, self).__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.timestep = timestep
        #self.num_steps = num_steps
        self.q = q
        self.use_real = use_real

        self.udecell = UDEcell(partial_units_list,timestep)

    def call(self, init_conditions_tensor, num_steps, training=False):

        # input_tensor -> (None, 4)
        # sol -> (None, 4, int(tinterval/timestep))

        # sol is obtained from integration(input_tensor)

        y = init_conditions_tensor
        sol = [y]
        
        for i in range(num_steps-1):

            y += self.udecell(y, training=training)
            sol.append(y)

        sol = tf.convert_to_tensor(sol)  # (num_tsteps, None, 4)

        ##########################################

        phi,chi,p,e = tf.unstack(sol, axis=-1)
        # phi = tf.keras.ops.nan_to_num(phi, nan=0.)
        # chi = tf.keras.ops.nan_to_num(chi, nan=np.pi)
        # p = tf.keras.ops.nan_to_num(p, nan=100.)
        # e = tf.keras.ops.nan_to_num(e, nan=0.5)

        #r = p / (1 + e*tf.math.cos(chi))    # (num_tsteps, None)

        # x1 = r * q/(1+q) * tf.math.cos(phi)
        # y1 = r * q/(1+q) * tf.math.sin(phi)

        # x2 = -r * 1/(1+q) * tf.math.cos(phi)
        # y2 = -r * 1/(1+q) * tf.math.sin(phi)
        
        waveform = h22(phi,chi,p,e, self.q, self.timestep)    # (num_tsteps - 2, None)

        if self.use_real:
            real_part = tf.math.real(waveform)
            # mean = tf.math.reduce_mean(real_part, axis=0, keepdims=True)
            # std = tf.math.reduce_std(real_part, axis=0, keepdims=True)

            
            return tf.transpose((real_part - self.mean) / self.std)     # (None, num_tsteps - 2)
            #return x1,y1,x2,y2

        imag_part = tf.math.imag(waveform)
        # mean = tf.math.reduce_mean(imag_part, axis=0, keepdims=True)
        # std = tf.math.reduce_std(imag_part, axis=0, keepdims=True)
        
        return tf.transpose((imag_part - self.mean) / self.std)     # (None, num_tsteps - 2)

    @tf.function
    def train_step(self, batch, slice_len):
        X,Y = batch
        
        with tf.GradientTape() as tape:
            Y_pred = self(X, num_steps=slice_len+2, training=True)   # (None, num_steps-2)
            train_loss = tf.reduce_mean((Y_pred-Y)**2)


        gradients = tape.gradient(train_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        return train_loss
        
    @tf.function
    def test_step(self, batch, slice_len):
        X,Y = batch

        Y_pred = self(X, num_steps=slice_len+2, training=False)   # (None, num_steps-2)
        test_loss = tf.reduce_mean((Y_pred-Y)**2)

        return test_loss



# %%

# phi0 = 0.
# chi0 = np.pi
# p0 = 100.
# e0 = 0.5


# tinit = 0.
# tfin = 1e+4

# times = np.linspace(tinit, tfin, 51)
# dt = times[1] - times[0]

# y0 = np.array(
#     [phi0,chi0,p0,e0]
# )

# batch_size = 2

# input_tensor = np.repeat([y0], batch_size, axis=0)

# timestep = times[1] - times[0]
# num_tsteps = int(times[-1]/timestep)

# input_tensor.shape

# q = 0.01
# partial_units_list = [4,4]

# model = UDE(partial_units_list, num_tsteps, timestep, q)

# test_wforms = np.array(model(input_tensor))

# # test_wforms.shape

# #x1,y1,x2,y2 = np.array(model(input_tensor))

# %%
# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# for wf in test_wforms[:10]:
#     ax.plot(times[1:-1], wf)
#     #ax.plot(x1,y1)
#     #ax.plot(x2,y2)

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# # ax.set_xlabel('time (s)')
# # ax.set_ylabel('$h_{22}$')
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')

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
tfin = 1e+4

times = np.linspace(tinit, tfin, 1000)
dt = times[1] - times[0]

y0 = np.array(
    [phi0,chi0,p0,e0]
)

sol = odeint(GR_rhs, y0, times)   # (num_steps, 4)
sol = tf.convert_to_tensor(sol, dtype=tf.float32)

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
# x_element = np.array(
#     [phi0,chi0,p0,e0]
# )
# y_element = true_wf[::1]

# train_size = 10000

# x_train = tf.repeat([x_element], train_size, axis=0)
# y_train = tf.repeat([y_element], train_size, axis=0)

# x_train.shape, y_train.shape


# %%
rng = np.random.default_rng()

slice_len_traj = 10
orbit_len = len(sol)
train_size = 10000

traj_indices = rng.integers(low=0, high=orbit_len-slice_len_traj, size=train_size)

# N - slice_len - 1 + slice_len - 2 = N - 3

x_train = tf.gather(sol, traj_indices)
y_train = tf.stack([true_wf[i:i+slice_len_traj-2] for i in traj_indices])

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
timestep = times[1] - times[0]
q = 0.01
partial_units_list = [16,8]

model = UDE(partial_units_list, timestep, q, mean, std, dtype=tf.float32)

y_pred = model(x_train[:10], num_steps=slice_len_traj)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

################### plotting ###############

ax.plot((times[1:-1]),true_wf, zorder=0)

for h_slice,i in zip(y_pred, traj_indices[:10]):
    ax.scatter(times[1:-1][i:i+slice_len_traj-2], h_slice, s=10, zorder=1)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# %%
# timestep = times[1] - times[0]

# q = 0.01
# partial_units_list = [32,32]

# model = UDE(partial_units_list, slice_len, timestep, q, mean, std)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanSquaredError(),
)

# # training
# history = model.fit(
#     x_train, y_train, 
#     batch_size=128, epochs=200, validation_split=0.2, verbose=1,
# )

# %%
from tqdm.notebook import tqdm


batch_size=128
num_epochs=100
steps_per_epoch=int(len(x_train)/batch_size)

indices=np.arange(x_train.shape[0])

rng.shuffle(indices)

custom_history = {}
custom_history['loss'] = []
custom_history['val_loss'] = []

for epoch in range(num_epochs):

    train_loss = 0.
    val_loss = 0.

    tepoch = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch+1}/{num_epochs}"
    )

    for idx in tepoch:

        batch_ids = rng.choice(indices, size=batch_size, replace=False)
        batch = (
            tf.gather(x_train, batch_ids), tf.gather(y_train, batch_ids)
        )

        slice_len = y_train.shape[1]

        loss = model.train_step(batch, slice_len=slice_len)
        train_loss += loss.numpy()
        tepoch.set_postfix_str("batch={:d}, train loss={:.4f}".format(idx+1, train_loss/(idx+1)))

    custom_history['loss'].append(train_loss/(idx+1))

    rng.shuffle(indices)

        
    for step in range(int(steps_per_epoch/4)):

        batch_ids = rng.choice(indices, size=batch_size, replace=False)
        batch = (
            tf.gather(x_train, batch_ids), tf.gather(y_train, batch_ids)
        )

        loss = model.test_step(batch, slice_len=slice_len)
        val_loss += loss.numpy()

    val_loss /= step + 1
    print("val loss={:.4f}".format(val_loss))

    custom_history['val_loss'].append(val_loss)

    rng.shuffle(indices)

# %%

plt.semilogy(custom_history['loss'])
plt.semilogy(custom_history['val_loss'])

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

################### plotting ###############

ax.plot((times[1:-1]),true_wf, zorder=0)

predict_len_traj = slice_len_traj + 100

y_pred = model(x_train[10:20], num_steps=predict_len_traj)

for h_slice,i in zip(y_pred, traj_indices[10:20]):
    ax.scatter(times[1:-1][i:i+predict_len_traj-2], h_slice, s=10, zorder=1)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

# %% [markdown]
# ### Unstable ODE integration?
#
# The network weights are initialized randomly. It is then possible that sometimes those weights are such that they lead to exponentionally diverging solutions. Let us linearize the r.h.s. of the UDE around the initial state and look at the eigenvalues of the matrix.

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
type(tf.shape(y_train)[1].numpy())

# %%
batch[1]

# %%
