# This code can be used for the preparation for SP1 test cases
# It makes a snapshot of one-soliton exact solution of KPE
# line 3 can also show the 'x-periodic' tilde_phi
# expressions are based on "Notes on solitary waves" section "Single-Line soliton"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def phi_expr(z):
    phi=a_phi*(Phi - 0.5*mu*np.power(z/H0,2)*Phi_2x + (mu*mu/24)*np.power(z/H0,4)*Phi_4x)
    return phi

def phi_expr_t0(z):
    xi0 = (np.sqrt(3*e*A)/(2*H0)) * ( x - (1+0.5*A*e)*np.sqrt(g*H0)*t0 ) # shift
    Phi0 = np.sqrt(4*e*A/3)*(np.tanh(xi0)+1)
    Phi_2x0 = -(A*e/mu) * np.sqrt(3*e*A) * np.tanh(xi0) * np.power(np.cosh(xi0),-2)
    Phi_4x0 = 3 * np.power(A*e/mu,2) * np.sqrt(3*e*A) *\
                (2*np.tanh(xi0)*np.power(np.cosh(xi0),-4) - np.power(np.tanh(xi0),3)*np.power(np.cosh(xi0),-2)) 
    phi0=a_phi*(Phi0 - 0.5*mu*np.power(z/H0,2)*Phi_2x0 + (mu*mu/24)*np.power(z/H0,4)*Phi_4x0)
    return phi0

def tilde_phi_expr(z):
    U0=(phi_expr_t0(z)[-1]-phi_expr_t0(z)[0])/(x1-x0)
    c0=(phi_expr_t0(z)[0]*x1-phi_expr_t0(z)[-1]*x0)/(x1-x0)
    corr=(U0*x+c0)
    tilde_phi=phi_expr(z)-corr
    #print('U0=',U0,'c0=',c0)
    return tilde_phi

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

marker = False # add markers to the plot
figure2='phi' # "phi" or "tilde_phi"

t0 = 0 # t_start

i = 30 # specify the instant
#------------------------#

a_eta=A*H0*e
a_phi=e*H0*np.sqrt(g*H0/mu)
c1 = 2*np.sqrt(4*e*A/3) # the coefficent of Phi (eq.37)

# To estimate the time step and one period.
speed = (1+0.5*A*e)*np.sqrt(g*H0) # propagation speed
dt_max = dx/speed
print('dt_max = %.4f' %dt_max)
T = Lx/speed
print('Period = %.4f' %T)

# Initialisation of plotting
fig, (ax1, ax2) = plt.subplots(2,figsize=(7,7),constrained_layout=True)
x = np.arange(x0, x1, dx)
if marker:
    line1, = ax1.plot([], [], 'b-o')
else:
    line1, = ax1.plot([], [], 'b-')
line2, = ax2.plot([], [], 'k-',  label='$z=0$')
line3, = ax2.plot([], [], 'c--', label=r'$z=H_0+\eta$')
if figure2=='phi':
    ax2r= ax2.twinx()
    line4, = ax2r.plot([], [], 'r-.',label='diff')


time_text=ax1.text( 0.8, 0.9, '', transform=ax1.transAxes, fontsize='large',
                   bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)) )

ax1.set_xlim(x0, x1)
ax1.set_ylim(-0.1*a_eta, 1.1*a_eta)
ax1.set_title(r'$\eta (x,t)$',fontsize='x-large')
ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$\eta$')
ax1.grid()

ax2.set_xlim(x0, x1)
if figure2=='phi':
    ax2.set_ylim(-0.1*c1*a_phi, 1.1*c1*a_phi)
    ax2.set_title(r'$\phi (x,z,t)$',fontsize='x-large')
    ax2.set_ylabel(r'$\phi$')
    ax2.legend(handles=[line2,line3,line4], loc='upper left', fontsize='large')
else: # figure2=='tilde_phi'
    ax2.set_ylim(-0.5*c1*a_phi, 0.5*c1*a_phi)
    ax2.set_title(r'$\tilde{\phi}(x,z,t)$',fontsize='x-large')
    ax2.set_ylabel(r'$\tilde{\phi}$')
    ax2.legend(handles=[line2,line3], loc='upper left', fontsize='large')
ax2.set_xlabel('$x$')
ax2.grid()

xi = (np.sqrt(3*e*A)/(2*H0)) * ( x - (1+0.5*A*e)*np.sqrt(g*H0)*i ) 
Phi = np.sqrt(4*e*A/3)*(np.tanh(xi)+1)
Phi_2x = -(A*e/mu) * np.sqrt(3*e*A) * np.tanh(xi) * np.power(np.cosh(xi),-2)
Phi_4x = 3 * np.power(A*e/mu,2) * np.sqrt(3*e*A) *\
                (2*np.tanh(xi)*np.power(np.cosh(xi),-4) - np.power(np.tanh(xi),3)*np.power(np.cosh(xi),-2)) 

eta = a_eta*np.power(np.cosh(xi),-2)

phi_b = phi_expr(0)
phi_s = phi_expr(H0+eta)

line1.set_data(x, eta)
if figure2=='phi':
    line2.set_data(x, phi_b)
    line3.set_data(x, phi_s)
    line4.set_data(x, phi_b-phi_s)
    ax2r.set_ylim(-1.5*np.amax(phi_b-phi_s),1.5*np.amax(phi_b-phi_s))
else:
    line2.set_data(x, tilde_phi_expr(0)) # tilde_phi at z=0
    line3.set_data(x, tilde_phi_expr(H0+eta)) # tilde_phi at z=h0

time_text.set_text('$t = %.1f$' %(i))

print(np.amax(phi_b-phi_s))

plt.show()