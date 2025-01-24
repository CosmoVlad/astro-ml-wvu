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

# %% [markdown]
# ### Newtonian trajectories

# %%
from scipy.integrate import odeint

def Newton_rhs(y,t):     # t -> t/M:   G=c=1
                        # M -> GM/c^3

    phi,chi,p,e = y

    phi_dot = (1 + e*np.cos(chi))**2 / p**1.5
    chi_dot = (1 + e*np.cos(chi))**2 / p**1.5
    p_dot = 0.
    e_dot = 0.

    return np.array([phi_dot,chi_dot,p_dot,e_dot])

def trajectory(phi,chi,p,e):

    r = p/(1 + e*np.cos(chi))

    return np.array([r*np.cos(phi),r*np.sin(phi)])


# %%
tinit = 0.
tfin = 2*np.pi

times = np.linspace(tinit, tfin, 100)

y0 = np.array(
    [0.,0.,1.,0.2]
)

sol = odeint(Newton_rhs, y0, times)
sol.shape

# %%
x,y = trajectory(*sol.T)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(x,y)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')


# %% [markdown]
# ### GR trajectories ("test particle")

# %%
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
p0 = 100.
e0 = 0.5

tinit = 0.
tfin = 2*np.pi * 100 * np.sqrt(p0**3)
#tfin = 100000

times = np.linspace(tinit, tfin, 100000)

y0 = np.array(
    [0.,0.,p0,e0]
)

sol = odeint(GR_rhs, y0, times)

x,y = trajectory(*sol.T)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(x,y)

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_aspect('equal')


# %% [markdown]
# ### Quadrupole tensor

# %%
def Jtensor(phi,chi,p,e, q):

    r = p / (1 + e*np.cos(chi))

    x1 = r * q/(1+q) * np.cos(phi)
    y1 = r * q/(1+q) * np.sin(phi)

    x2 = -r * 1/(1+q) * np.cos(phi)
    y2 = -r * 1/(1+q) * np.sin(phi)

    Ixx = x1**2 + q*x2**2
    Iyy = y1**2 + q*y2**2
    Ixy = x1*y1 + q*x2*y2

    trace = Ixx + Iyy

    return np.array(
        [
            [Ixx - trace/3, Ixy],
            [Ixy, Iyy - trace/3]   
        ]
    )


def h22(phi,chi,p,e, q, dt):

    J = Jtensor(phi,chi,p,e, q)
    r = p / (1 + e*np.cos(chi))

    Jxx,Jxy,Jyy = J[0,0],J[0,1],J[1,1]

    ddJxx = (Jxx[2:] - 2*Jxx[1:-1] + Jxx[:-2]) / dt**2
    ddJxy = (Jxy[2:] - 2*Jxy[1:-1] + Jxy[:-2]) / dt**2
    ddJyy = (Jyy[2:] - 2*Jyy[1:-1] + Jyy[:-2]) / dt**2

    return 1/r[1:-1] * np.sqrt(4*np.pi/5) * (ddJxx - 2*1j*ddJxy - ddJyy)


# %%
dt = times[1] - times[0]  # assuming that all timesteps are equal
q = 0.01

print(sol.T.shape)

Jtensor(*sol.T, q).shape

# %%
waveform = h22(*sol.T, q,dt)

wf_norm = (waveform - np.mean(waveform)) / np.std(waveform)

fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

ax.plot(times[1:-1], np.real(wf_norm), label='Re')
ax.plot(times[1:-1], np.imag(wf_norm), label='Im')
ax.legend()

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xlabel('time (s)')
ax.set_ylabel('$h_{22}$')



# %%
