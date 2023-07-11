# Post-processing code for SP1
# compare the numerical results with the "exact solution" frame by frame
# also can be used to compare tilde_phi_exact with phi_numerical

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os.path

def load_text(file_path, file_name):
    """load data from text files (.csv)"""
    file = os.path.join(file_path, file_name)
    xi, h, phis, phib = np.loadtxt(file, usecols=(0,1,2,3), unpack=True)
    return xi, h, phis, phib

def load_binary(file_path, file_name):
    """load data from binary files (.npy)"""
    file = os.path.join(file_path, file_name)
    data_array = np.load(file)
    xi = data_array[:,0]
    h  = data_array[:,1]
    phis = data_array[:,2]
    phib = data_array[:,3]
    return xi, h, phis, phib

#------ User Input ------#
g = 9.81
A = 0.1
H0= 1.0
e = 0.01
mu= e*e

Lx = 400
x0 = -0.5*Lx
x1 = 0.5*Lx
dx = 0.5

data_path ='data/SP1_test/TC1_SV'
gif_name='SP1_TC1.gif'
plot_phi=False

interval=2
#------------------------#

a_eta=A*H0*e
a_phi=e*H0*np.sqrt(g*H0/mu)
c1 = 2*np.sqrt(4*e*A/3)

path=os.path.join(data_path,'figures')
try: 
    os.mkdir(path) 
except OSError as error: 
    print(error) 

# get t from the energy file
energy_file = os.path.join(data_path,'energy.csv')
t_steps_full = np.loadtxt(energy_file, usecols=0)
t_steps = t_steps_full[::interval]

fig, (ax1, ax2) = plt.subplots(2,figsize=(7,7),constrained_layout=True)

line1e, = ax1.plot([], [], 'b-',label='Exact')
line1n, = ax1.plot([], [], 'r--',label='Numerical')
line2, = ax2.plot([], [], 'k-',label='$z=0$')
line3, = ax2.plot([], [], 'c--',label='$z=H_0+\eta$')
time_text=ax1.text( 0.8, 0.9, '', transform=ax1.transAxes, fontsize='large',
                   bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)) )
ax1.grid()
ax2.grid()

def init():
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(-0.1*a_eta, 1.1*a_eta)
    ax1.set_title('$\eta (x,t)$',fontsize='x-large')
    ax1.set_xlabel('$x$',fontsize='large')
    ax1.set_ylabel('$\eta$',fontsize='large')
    
    ax1.legend(loc='upper left',fontsize='large')
    
    ax2.set_xlim(x0, x1)
    #ax2.set_ylim(-0.1*c1*a_phi, 1.1*c1*a_phi)
    ax2.set_ylim(-0.3, 0.1)
    ax2.set_title(r'$\tilde{\phi}(x,z,t)$',fontsize='x-large')
    ax2.set_xlabel('$x$',fontsize='large')
    ax2.set_ylabel(r'$\tilde{\phi}$',fontsize='large')
    
    ax2.legend(loc='upper left',fontsize='large')
    return line1e,line1n,line2,line3,

def animate(t):
    # numerical results
    tt   = format(t,'.3f')
    fname = tt+'.npy'
    #xn, hn, phisn, phibn = load_data(data_path, fname)
    xn, hn, phisn, phibn = load_binary(data_path, fname)

    # exact solution of KPE
    xc = (1+0.5*A*e)*np.sqrt(g*H0)*t
    if xc < x1:
        xi = (np.sqrt(3*e*A)/(2*H0)) * (xn - (1+0.5*A*e)*np.sqrt(g*H0)*t)
    else:
        xi = (np.sqrt(3*e*A)/(2*H0)) * (xn - (1+0.5*A*e)*np.sqrt(g*H0)*t + Lx) 
    
    eta = a_eta*np.power(1/np.cosh(xi),2)
    
    if plot_phi:
        Phi = np.sqrt(4*e*A/3)*(np.tanh(xi)+1)
        Phi_2x = -(A*e/mu) * np.sqrt(3*e*A) * np.tanh(xi) * np.power(1/np.cosh(xi),2)
        Phi_4x = 3 * np.power(A*e/mu,2) * np.sqrt(3*e*A) *\
                (2*np.tanh(xi)*np.power(1/np.cosh(xi),4) - np.power(np.tanh(xi),3)*np.power(1/np.cosh(xi),2))

        z_b = 0/H0
        z_s = (H0+eta)/H0
        phi_b = a_phi*(Phi - 0.5*mu*np.power(z_b,2)*Phi_2x + (mu*mu/24)*np.power(z_b,4)*Phi_4x)
        phi_s = a_phi*(Phi - 0.5*mu*np.power(z_s,2)*Phi_2x + (mu*mu/24)*np.power(z_s,4)*Phi_4x)
    
        U0_s=(phi_s[-1]-phi_s[0])/Lx
        c0_s=phi_s[0]
        phit_ex_s=phi_s - (U0_s*xn + c0_s)
    
    # plot
    line1e.set_data(xn, eta)
    line1n.set_data(xn, hn-H0)
    line2.set_data(xn, phibn)
    line3.set_data(xn, phisn)
    #line3.set_data(xn, phit_ex_s)
    time_text.set_text('$t = %s$' %tt)

    return line1e,line1n,line2,line3,

anim = FuncAnimation(fig, animate, init_func=init, frames=t_steps, interval=200, blit=True)

anim.save(os.path.join(path, gif_name), writer='pillow',dpi=200)

plt.close()