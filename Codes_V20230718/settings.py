from firedrake import *
import numpy as np

"""
    *********************************************
    *                 Test case                 *
    *********************************************"""
def test_case():
    scheme = 'SV' # "SE": Symplectic-Euler ; "SV": Stormer-Verlet
    dim = '3D'    # "2D": b(x)=0, SP1; "3D": b(x,y), SP2/3
    save_path = 'data/SP2_test/Test_SV1'
    save_pvd = True  # Whether or not to save the 3D results into .pvd files
    nCG = 2
    if dim == '2D':
        FWF = 0 
        soliton = 'SP1'
    else:
        FWF = 0   # 0: b=0 or use mild-slope aproximation; 1: b(x,y)!=0
        soliton = 'SP2' # SP2/SP3: two- or three-soliton simulations
    return scheme, dim, save_path, save_pvd, FWF, soliton, nCG

"""
    *********************************************************************
    *                         Numerical domain                          *
    *********************************************************************"""
def domain(soliton):
    g = 9.81
    
    Amp = 0.1 # specifying the amplitude tilde_A for SP1
    tan0 = 1/4 # specifying tan(hat_theta) for SP2/3

    if soliton=='SP1':
        epsilon = 0.01
        mu = epsilon**2
        Lx = 400
        res_x = 0.5
        H0 = 1.0
        n_z = 4
        Ly = 1.0 #symbolic
        res_y = res_x #symbolic
    elif soliton=='SP2':
        H0 = 20
        n_z = 4
        epsilon = 0.05
        mu = epsilon**2
        hat_y2 = 20
        c1 = pow(9/2,1/6)
        k1 = -tan0/(c1*sqrt(epsilon))
        hat_x2 = -k1*c1*sqrt(epsilon)*hat_y2 - (ln(2)*sqrt(mu))/(k1*c1*sqrt(epsilon))
        x2 = (H0/sqrt(mu))*hat_x2
        y2 = (H0/sqrt(mu))*hat_y2
        Lx = 2*x2
        Ly = 2*y2
        res_x = 31.1432
        res_y = 33.3333

    return g, H0, Lx, Ly, res_x, res_y, n_z, epsilon, mu, Amp, tan0


"""
    **************************************************************************
    *                              Initialization                            *
    **************************************************************************"""
def IC_2D(g, H0, epsilon, mu, Amp):
    # initialising the system with 2D exact solution of KPE
    A = Amp
    a_eta = A*H0*epsilon
    a_phi = epsilon*H0*np.sqrt(g*H0/mu)
    
    def xi_expr(x,t):
        function=( (sqrt(3*epsilon*A)/(2*H0)) * (x - (1+0.5*A*epsilon)*sqrt(g*H0)*t) )
        return function
        
    def h_ex_expr(xi): # h = H0 + eta
        function=( H0 + a_eta*(pow(cosh(xi),-2)) )
        return function
        
    def Phi_expr(xi):
        function=( sqrt(4*epsilon*A/3)*(tanh(xi)+1) )
        return function

    def Phi_2x_expr(xi):
        function=( -(A*epsilon/mu) * sqrt(3*epsilon*A) * tanh(xi) * pow(cosh(xi),-2) )
        return function
    
    def Phi_4x_expr(xi):
        function=( 3 * pow(A*epsilon/mu,2) * sqrt(3*epsilon*A) *\
                             (2*tanh(xi)*pow(cosh(xi),-4)-pow(tanh(xi),3)*pow(cosh(xi),-2)) )
        return function
    
    def phi_corr(x,U0,c0):
        function=U0*(x-0)+c0
        return function
    
    def phi_ex_expr(Phi,Phi_2x,Phi_4x,z):
        function=( a_phi*(Phi - 0.5*mu*pow(z/H0,2)*Phi_2x + (mu*mu/24)*pow(z/H0,4)*Phi_4x) )
        return function

    def tilde_phi_expr(function,Phi,Phi_2x,Phi_4x,phi_corr,z):
        function.interpolate( a_phi*(Phi - 0.5*mu*pow(z/H0,2)*Phi_2x + (mu*mu/24)*pow(z/H0,4)*Phi_4x) - phi_corr )
          
    return xi_expr, h_ex_expr, Phi_expr, Phi_2x_expr, Phi_4x_expr, phi_corr, phi_ex_expr, tilde_phi_expr


def IC_3D(soliton, g, H0, epsilon, mu, tan0):
    # initialising the 3D system with 3D exact solution of KPE
    if soliton=='SP2': # k2=k3=0, k1=-k4
        c1 = pow(9/2,1/6)
        c2 = pow(4/3,1/3)
        c3 = pow(32/81,1/6)
        k1 = -tan0/(c1*sqrt(epsilon))
        k4 = -k1
        a_phi = epsilon*H0*sqrt(g*H0/mu)

        def h_ex_expr(x,y):  # h(x,y,0) = H0 + eta(x,y,0)
            function=( H0 + 4*H0*c2*epsilon*k4**2*(exp(k4*(c1**2*epsilon*k4*y/H0 + 2*c1*sqrt(epsilon)*x/H0)) + 2*exp(c1*sqrt(epsilon)*k4*x/H0) + exp(c1**2*epsilon*k4**2*y/H0))*exp(c1*sqrt(epsilon)*k4*x/H0)/(4*exp(2*k4*(c1**2*epsilon*k4*y/H0 + c1*sqrt(epsilon)*x/H0)) + 4*exp(k4*(c1**2*epsilon*k4*y/H0 + c1*sqrt(epsilon)*x/H0)) + 4*exp(k4*(c1**2*epsilon*k4*y/H0 + 3*c1*sqrt(epsilon)*x/H0)) + exp(4*c1*sqrt(epsilon)*k4*x/H0) + 2*exp(2*c1*sqrt(epsilon)*k4*x/H0) + 1) )
            return function

        def Phi_expr(x,y): # Phi(x,y,0)
            function = 2*c3*sqrt(epsilon)*k4*(exp(2*c1*sqrt(epsilon)*k4*x/H0) - 1)/(2*exp(k4*(c1**2*epsilon*k4*y/H0 + c1*sqrt(epsilon)*x/H0)) + exp(2*c1*sqrt(epsilon)*k4*x/H0) + 1)
            return function

        def Phi_2ed_expr(x,y): # Phi_xx(x,y,0)+Phi_yy(x,y,0)
            function = 4*c1**2*c3*epsilon**(3/2)*k4**3*(-c1**2*epsilon*k4**2*(2*exp(c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) - 2*exp(c1*k4*(c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) + exp(4*c1*sqrt(epsilon)*k4*x/H0) - 1)*exp(c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + (-exp(c1*k4*(c1*epsilon*k4*y + 4*sqrt(epsilon)*x)/H0) - 2*exp(c1*k4*(2*c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 2*exp(c1*k4*(2*c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) - 4*exp(3*c1*sqrt(epsilon)*k4*x/H0) + 4*exp(c1*sqrt(epsilon)*k4*x/H0) + exp(c1**2*epsilon*k4**2*y/H0))*exp(c1*sqrt(epsilon)*k4*x/H0))/(mu*(8*exp(3*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 12*exp(2*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 6*exp(c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 12*exp(2*c1*k4*(c1*epsilon*k4*y + 2*sqrt(epsilon)*x)/H0) + 12*exp(c1*k4*(c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) + 6*exp(c1*k4*(c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) + exp(6*c1*sqrt(epsilon)*k4*x/H0) + 3*exp(4*c1*sqrt(epsilon)*k4*x/H0) + 3*exp(2*c1*sqrt(epsilon)*k4*x/H0) + 1))
            return function

        def Phi_4th_expr(x,y): # Phi_xxxx(x,y,0)+Phi_yyyy(x,y,0)+2*Phi_xxyy(x,y,0)
            function = 4*c1**4*c3*epsilon**(5/2)*k4**5*(c1**4*epsilon**2*k4**4*(-8*exp(3*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 44*exp(2*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) - 22*exp(c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) - 44*exp(2*c1*k4*(c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) - 22*exp(c1*k4*(c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) + 22*exp(c1*k4*(c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) + 22*exp(c1*k4*(c1*epsilon*k4*y + 7*sqrt(epsilon)*x)/H0) + 8*exp(c1*k4*(3*c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) - exp(8*c1*sqrt(epsilon)*k4*x/H0) - 2*exp(6*c1*sqrt(epsilon)*k4*x/H0) + 2*exp(2*c1*sqrt(epsilon)*k4*x/H0) + 1)*exp(c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) - 2*c1**2*epsilon*k4**2*(8*exp(3*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) - 44*exp(2*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 22*exp(c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 44*exp(2*c1*k4*(c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) - 122*exp(c1*k4*(c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) + 122*exp(c1*k4*(c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) - 22*exp(c1*k4*(c1*epsilon*k4*y + 7*sqrt(epsilon)*x)/H0) - 8*exp(c1*k4*(3*c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) + exp(8*c1*sqrt(epsilon)*k4*x/H0) - 22*exp(6*c1*sqrt(epsilon)*k4*x/H0) + 22*exp(2*c1*sqrt(epsilon)*k4*x/H0) - 1)*exp(c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + (-44*exp(3*c1*k4*(c1*epsilon*k4*y + 2*sqrt(epsilon)*x)/H0) - 78*exp(c1*k4*(c1*epsilon*k4*y + 2*sqrt(epsilon)*x)/H0) + 78*exp(c1*k4*(c1*epsilon*k4*y + 6*sqrt(epsilon)*x)/H0) - exp(c1*k4*(c1*epsilon*k4*y + 8*sqrt(epsilon)*x)/H0) - 22*exp(c1*k4*(2*c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 138*exp(c1*k4*(2*c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) - 138*exp(c1*k4*(2*c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) + 22*exp(c1*k4*(2*c1*epsilon*k4*y + 7*sqrt(epsilon)*x)/H0) + 44*exp(c1*k4*(3*c1*epsilon*k4*y + 2*sqrt(epsilon)*x)/H0) - 8*exp(c1*k4*(4*c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) + 8*exp(c1*k4*(4*c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) - 16*exp(7*c1*sqrt(epsilon)*k4*x/H0) + 176*exp(5*c1*sqrt(epsilon)*k4*x/H0) - 176*exp(3*c1*sqrt(epsilon)*k4*x/H0) + 16*exp(c1*sqrt(epsilon)*k4*x/H0) + exp(c1**2*epsilon*k4**2*y/H0))*exp(c1*sqrt(epsilon)*k4*x/H0))/(mu**2*(32*exp(5*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 80*exp(4*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 80*exp(3*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 40*exp(2*c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 10*exp(c1*k4*(c1*epsilon*k4*y + sqrt(epsilon)*x)/H0) + 120*exp(2*c1*k4*(c1*epsilon*k4*y + 2*sqrt(epsilon)*x)/H0) + 120*exp(2*c1*k4*(c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) + 40*exp(c1*k4*(c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) + 40*exp(2*c1*k4*(c1*epsilon*k4*y + 4*sqrt(epsilon)*x)/H0) + 60*exp(c1*k4*(c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) + 40*exp(c1*k4*(c1*epsilon*k4*y + 7*sqrt(epsilon)*x)/H0) + 10*exp(c1*k4*(c1*epsilon*k4*y + 9*sqrt(epsilon)*x)/H0) + 80*exp(2*c1*k4*(2*c1*epsilon*k4*y + 3*sqrt(epsilon)*x)/H0) + 160*exp(c1*k4*(3*c1*epsilon*k4*y + 5*sqrt(epsilon)*x)/H0) + 80*exp(c1*k4*(3*c1*epsilon*k4*y + 7*sqrt(epsilon)*x)/H0) + exp(10*c1*sqrt(epsilon)*k4*x/H0) + 5*exp(8*c1*sqrt(epsilon)*k4*x/H0) + 10*exp(6*c1*sqrt(epsilon)*k4*x/H0) + 10*exp(4*c1*sqrt(epsilon)*k4*x/H0) + 5*exp(2*c1*sqrt(epsilon)*k4*x/H0) + 1))
            return function

        def phi_ex_expr(Phi,Phi_2,Phi_4,z): # phi(x,y,z,0), here z is z_bt
            function=( a_phi*(Phi - 0.5*mu*pow(z/H0,2)*Phi_2 + (mu*mu/24)*pow(z/H0,4)*Phi_4) )
            return function

        def phi_corr(x,U0,c0):
            function = U0*x + c0
            return function

        def tilde_phi_expr(function, Phi,Phi_2,Phi_4,phi_corr,z):
            function.interpolate( a_phi*(Phi - 0.5*mu*pow(z/H0,2)*Phi_2 + (mu*mu/24)*pow(z/H0,4)*Phi_4) - phi_corr )

    return h_ex_expr, Phi_expr, Phi_2ed_expr, Phi_4th_expr, phi_ex_expr, phi_corr, tilde_phi_expr


"""
    ***********************************
    *               Time              *
    ***********************************"""
def set_time(soliton, g, H0, mu):
    if soliton=='SP1':
        T0=0.0
        dt=0.1
        Tend=T0+1
    elif soliton=='SP2':
        dt_BLE = 0.005
        T_BLE  = 13
        # ==== PFE ====
        dt = H0*dt_BLE/sqrt(g*H0*mu)         # time step
        T0 = 0.0                             # Initial time
        #Tend = T0 + H0*T_BLE/sqrt(g*H0*mu)     # Final time 
        Tend=T0+200*dt
    
    t = T0                  # Temporal variable 
    dt_save = 10*dt          # saving time step
    return T0, t, dt, Tend, dt_save
