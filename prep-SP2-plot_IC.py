# this code is used to check whether the sympy-generated code for the ICs is correct or not
# also to make preparations for SP2 simulations

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import math

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# input parameters
epsilon  = 0.05
mu = epsilon**2
tan0 = 1/4 # specifying tan(hat_theta), BLE
hat_y2 = 20 # BLE
H0 = 20
g = 9.81

c1 = pow(9/2,1/6)
c2 = pow(4/3,1/3)
c3 = pow(32/81,1/6)

sx=c1*np.sqrt(epsilon/mu)
sy=c1**2*epsilon/np.sqrt(mu)

# Make data.
# Under the framework of BLE
k1 = -tan0*sx/sy
hat_x2 = (-k1**2*sy*hat_y2-np.log(2))/(k1*sx)
Lx_BLE = 2*hat_x2
Ly_BLE = 2*hat_y2
k4 = -k1
v_BLE = 1 + k1*k1*epsilon*np.sqrt(2)/c1 # speed under BLE
T_BLE = Lx_BLE/v_BLE
T_end_BLE = 13
dt_BLE = 0.005
Nx = 132
Ny = 480
dx_BLE = Lx_BLE/Nx
dy_BLE = Ly_BLE/Ny
dt_BLE_max = dx_BLE/v_BLE
print('k4 = %.5f' % k4)
Amp = epsilon*H0*c2*(0.5*k4**2)
max_A = 4*Amp
print('Amplitide = %.5f m' % Amp)
print('max_A = %.5f m' % max_A)
print('BL framework: Lx=%10.4f,   Ly=%10.4f,   Period=%10.4f,   T_end=%10.4f,' %(Lx_BLE, Ly_BLE, T_BLE, T_end_BLE))
print('              Δx=%10.4f,   Δy=%10.4f,   Δt_max=%10.4f,      Δt=%10.4f.' %(dx_BLE,dy_BLE,dt_BLE_max,dt_BLE))

# Under the framework of PFE
x2 = (H0/np.sqrt(mu))*hat_x2
y2 = (H0/np.sqrt(mu))*hat_y2
Lx_PFE = 2*x2
Ly_PFE = 2*y2
v_PFE = v_BLE*np.sqrt(g*H0)
T_PFE = Lx_PFE/v_PFE
dx_PFE = Lx_PFE/Nx
dy_PFE = Ly_PFE/Ny
dt_PFE_max = dx_PFE/v_PFE
T_end_PFE = H0*T_end_BLE/np.sqrt(g*H0*mu)
dt_PFE = H0*dt_BLE/np.sqrt(g*H0*mu)
print('PF framework: Lx=%10.4f m, Ly=%10.4f m, Period=%10.4f s, T_end=%10.4f s,' %(Lx_PFE, Ly_PFE, T_PFE, T_end_PFE))
print('              Δx=%10.4f m, Δy=%10.4f m, Δt_max=%10.4f s,    Δt=%10.4f s.' %(dx_PFE,dy_PFE,dt_PFE_max,dt_PFE))

Nx_code = round(Lx_PFE/dx_PFE)
Ny_code = round(Ly_PFE/dy_PFE)
print('Nx =', Nx_code, ', Ny =', Ny_code)

# ======== plotting =========
xc = np.linspace(-x2, x2, Nx+1)
yc = np.linspace(-y2, y2, Ny+1)
x, y = np.meshgrid(xc, yc)

x_12 = (-k1**2*sy*yc-np.log(2))/(k1*sx) # {x,y}
line12, = ax.plot(x_12, yc, 'g-', linewidth=2, zs=0, zdir='z')

eta_n = 4*H0*c2*epsilon*k4**2*(np.exp(k4*(c1**2*epsilon*k4*y/H0 + 2*c1*np.sqrt(epsilon)*x/H0)) + 2*np.exp(c1*np.sqrt(epsilon)*k4*x/H0) \
	  + np.exp(c1**2*epsilon*k4**2*y/H0))*np.exp(c1*np.sqrt(epsilon)*k4*x/H0)
eta_d= 4*np.exp(2*k4*(c1**2*epsilon*k4*y/H0 + c1*np.sqrt(epsilon)*x/H0)) \
       + 4*np.exp(k4*(c1**2*epsilon*k4*y/H0 + c1*np.sqrt(epsilon)*x/H0)) \
       + 4*np.exp(k4*(c1**2*epsilon*k4*y/H0 + 3*c1*np.sqrt(epsilon)*x/H0)) \
       + np.exp(4*c1*np.sqrt(epsilon)*k4*x/H0) + 2*np.exp(2*c1*np.sqrt(epsilon)*k4*x/H0) + 1
eta = eta_n/eta_d

# Plot the surface.
ax.set_box_aspect((x2,y2,2*x2))
surf = ax.plot_surface(x, y, eta, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha=0.8)


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.6, aspect=10)

plt.show()