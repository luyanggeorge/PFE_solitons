#import pdb
import time
import numpy as np
import os
import sympy as sp
from sympy.integrals.quadrature import gauss_lobatto

from firedrake import *
from firedrake.petsc import PETSc

from savings import *
import solvers_nob as SP_solvers
from settings import *

start_time = time.perf_counter()

"""
    ****************************************
    *               Settings               *
    **************************************** """

scheme, dim, save_path, save_pvd, FWF, soliton, nCG = test_case()
g, H0, Lx, Ly, res_x, res_y, n_z, epsilon, mu, Amp, tan0 = domain(soliton)
if soliton=='SP1':
    xi_expr, h_ex_expr, Phi_expr, Phi_2x_expr, Phi_4x_expr, phi_corr, phi_ex_expr, tilde_phi_expr\
    = IC_2D(g, H0, epsilon, mu, Amp)
else:
    h_ex_expr, Phi_expr, Phi_2ed_expr, Phi_4th_expr, phi_ex_expr, phi_corr, tilde_phi_expr \
    = IC_3D(soliton, g, H0, epsilon, mu, tan0)
T0, t, dt, Tend, dt_save = set_time(soliton, g, H0, mu)

PETSc.Sys.Print('...settings loaded for %s!' % soliton)

# Create the directory to store output files
try:
    os.makedirs(save_path, exist_ok = True)
    PETSc.Sys.Print("Directory created successfully!")
except OSError as error:
    PETSc.Sys.Print("Directory can not be created!")

PETSc.Sys.Print('Setting up mesh across %d process(es)...' % COMM_WORLD.size)
    
"""
    ****************************************************
    *              Construction of the mesh            *
    **************************************************** """

#_________________ Vertical discretization ________________#
Nz = n_z+1         # Number of points in one vertical element

#________________ Horizontal discretization _______________#
Nx = round(Lx/res_x)    # Number of elements in x (round to the nearest integer)
Ny = round(Ly/res_y)    # Number of elements in y

#___________________________ Mesh _________________________#
if dim=="2D":
    # Generate a periodic mesh of an interval [0,Lx].
    hor_mesh = PeriodicIntervalMesh(Nx,Lx)
    # Change mesh coordinates: [0,Lx] => [-Lx/2,Lx/2]
    hor_mesh.coordinates.dat.data[:] -= 0.5*Lx
else:
    # Generate a periodic rectangular mesh
    hor_mesh = PeriodicRectangleMesh(Nx, Ny, Lx, Ly, direction='x', quadrilateral=True)
    # Change mesh coordinates: [0,Lx]*[0,Ly] => [-Lx/2,Lx/2]*[-Ly/2,Ly/2]
    hor_mesh.coordinates.dat.data[:,0] -= 0.5*Lx
    hor_mesh.coordinates.dat.data[:,1] -= 0.5*Ly

x = SpatialCoordinate(hor_mesh)

PETSc.Sys.Print('...mesh created!')

PETSc.Sys.Print('Assembling z-matrices...')

"""
    ************************
    * Compute the matrices *
    ************************ """
#_______ Initialization ______#
A = np.zeros((Nz,Nz))
B = np.zeros((Nz,Nz)) # FWF
C = np.zeros((Nz,Nz)) # FWF
M = np.zeros((Nz,Nz))
D = np.zeros((Nz,Nz))
S = np.zeros((Nz,Nz))

# construction of Lagrange polynomials
varphi=[]
z = sp.Symbol('z', positive = True)
#-------------GLL--------------
points, weights = gauss_lobatto(Nz, 20)
nodes=[(x_i+1)*0.5*H0 for x_i in points]
nodes.reverse()
z_k=nodes.copy()
#------------------------------
for i in range(Nz):
    index = list(range(Nz))
    index.pop(i)
    varphi.append(sp.prod([(z-z_k[j])/(z_k[i]-z_k[j]) for j in index]))

#____ Filling the matrices ___#
for i in range(0,Nz):
    for j in range(0,Nz):
        expr_A = sp.diff(varphi[i],z)*sp.diff(varphi[j],z)
        expr_M = varphi[i]*varphi[j]
        expr_D = z*varphi[i]*sp.diff(varphi[j],z)
        expr_S = z*z*sp.diff(varphi[i],z)*sp.diff(varphi[j],z)
        A[i,j] = sp.integrate(expr_A, (z,0,H0))
        M[i,j] = sp.integrate(expr_M, (z,0,H0))
        D[i,j] = sp.integrate(expr_D, (z,0,H0))
        S[i,j] = sp.integrate(expr_S, (z,0,H0))

#________ Submatrices ________#
A11 = A[0,0]
A1N = as_tensor(A[0,1:])
AN1 = as_tensor(A[1:,0])
ANN = as_tensor(A[1:,1:])

M11 = M[0,0]
M1N = as_tensor(M[0,1:])
MN1 = as_tensor(M[1:,0])
MNN = as_tensor(M[1:,1:])

D11 = D[0,0]
D1N = as_tensor(D[0,1:])
DN1 = as_tensor(D[1:,0])
DNN = as_tensor(D[1:,1:])

S11 = S[0,0]
S1N = as_tensor(S[0,1:])
SN1 = as_tensor(S[1:,0])
SNN = as_tensor(S[1:,1:])

# yl added: full weak forms
if FWF==1:
    for i in range(0,Nz):
        for j in range(0,Nz):
            expr_B = varphi[i]*sp.diff(varphi[j],z)
            expr_C = z*sp.diff(varphi[i],z)*sp.diff(varphi[j],z)
            B[i,j] = sp.integrate(expr_B, (z,0,H0))
            C[i,j] = sp.integrate(expr_C, (z,0,H0))

B11 = B[0,0]
B1N = as_tensor(B[0,1:])
BN1 = as_tensor(B[1:,0])
BNN = as_tensor(B[1:,1:])

C11 = C[0,0]
C1N = as_tensor(C[0,1:])
CN1 = as_tensor(C[1:,0])
CNN = as_tensor(C[1:,1:])

PETSc.Sys.Print('... matrices ready!')

PETSc.Sys.Print('Create functions...')

"""
    *************************************************
    *       Definition of the function spaces       *
    ************************************************* """
#___________________ For h and psi_1 ___________________#
V = FunctionSpace(hor_mesh, "CG", nCG)
#_____________________ For hat_psi _____________________#
Vec = VectorFunctionSpace(hor_mesh, "CG", nCG, dim=n_z)
#_________________ Mixed function space ________________#
V_mixed = V*Vec # to solve simultaneous weak formulations

"""
    ******************************************************
    *            Definition of the functions             *
    ****************************************************** """

if scheme=="SE": #_________ Symplectic-Euler scheme _________#
    #______________________ At time t^n _____________________#
    h_n0 = Function(V)                                   # h^n
    psi_1_n0 = Function(V)                     # tilde_psi_1^n
    hat_psi_n0 = Function(Vec)               # hat_tilde_psi^n
    
    #________________ At time t^{n+1} and t^* _______________#
    psi_1_n1 = Function(V)                       # psi_1^{n+1}
    w_n1 = Function(V_mixed)              # h^{n+1}, hat_psi^*
    h_n1, hat_psi_star = split(w_n1) 
    hat_psi_n1 = Function(Vec)                 # hat_psi^{n+1}
else: #________________ Stormer-Verlet scheme _______________#
    #______________________ At time t^n _____________________#
    h_n0 = Function(V)                                   # h^n
    psi_1_n0 = Function(V)                           # psi_1^n
    hat_psi_n0 = Function(Vec)                     # hat_psi^n

    #_______________ At time t^{n+1/2} and t^* ______________#
    w_half = Function(V_mixed)      # psi_1^{n+1/2}, hat_psi^*
    psi_1_half, hat_psi_star = split(w_half)  

    #_______________ At time t^{n+1} and t^** _______________#
    psi_1_n1 = Function(V)                       # psi_1^{n+1}
    w_n1 = Function(V_mixed)           # h^{n+1}, hat_psi^{**}
    h_n1, hat_psi_aux = split(w_n1)    
    hat_psi_n1 = Function(Vec)                 # hat_psi^{n+1}

#______________ Bea bed topography b(x,y)____________________#
b = Function(V)                                       # b(x,y)

#_______________________ Depth at rest ______________________#
H = Function(V)                                         # H(x)

#______________________ Trial functions _____________________#
psi_1 = TrialFunction(V)      # psi_1^{n+1} for linear solvers
hat_psi = TrialFunction(Vec)# hat_psi^{n+1} for linear solvers

#_______________________ Test functions _____________________#
delta_h = TestFunction(V)                         # from dH/dh
delta_hat_psi = TestFunction(Vec)           # from dH/dhat_psi
w_t = TestFunction(V_mixed)                # from dH/dpsi_1...
delta_psi, delta_hat_star = split(w_t)    # ...and dH/dhat_psi
if scheme=="SV": 
    w_t_sv = TestFunction(V_mixed)
    delta_h_sv, delta_hat_psi_sv = split(w_t_sv)

# ------SE------
# step1: use delta_psi and delta_hat_star to solve simultaneously for h^n+1 and psi_hat^*
# step2: use delta_h to solve for psi_1^(n+1)
# step3: use delta_hat_psi to update psi_hat^(n+1) using Laplace Eq.
# ------SV------
# step1: use delta_h_sv and delta_hat_psi_sv to solve simultaneously for psi_1^half and psi_hat^*
# step2: use delta_psi and delta_hat_star to solve simultaneously for h^n+1 and psi_hat^**
# step3: use delta_h to solve for psi_1^(n+1)
# step4: use delta_hat_psi to update psi_hat^(n+1) using Laplace Eq.

#______ correction functions for soliton simulations ________#
corr_nz  = Function(V)    # U_0*x+c_0 at the free surface
hat_corr = Function(Vec)  # U_0*x+c_0 underwater
U0_nz  = Function(V)      # U_0 at the free surface
c0_nz  = Function(V)
hat_U0 = Function(Vec)    # U_0 underwater

# _______ functions used for setting initial values _________#
phii_z = Function(V)  # initialising phi underwater
z_bt = Function(V)    # z = (z*/H0)*h before transformation

PETSc.Sys.Print('...functions created!')

PETSc.Sys.Print('Initalisation of the functions...')
"""
    ***********************************************************************************
    *                          Initialisation of the Functions                        *
    ***********************************************************************************"""
#---------------------------- geometry of the Fluid domain  ----------------------------#             
b.interpolate(Constant(0))
H.assign(H0-b)                  

#----------------------------------------------------------------------------------------#
#                               Set up initial conditions                                #
#----------------------------------------------------------------------------------------#

if soliton == 'SP1':
    #____________________________ Initialization of Depth _______________________________#
    xi = xi_expr(x[0],t) # a UFL expression
    h_n0.interpolate(h_ex_expr(xi)) # initialise h^n
    w_n1.sub(0).assign(h_n0) # initialise h^{n+1}

    #______________ Velocity potential at the free surface: phi(x,y,z=h;t) ______________#
    Phi   = Phi_expr(xi) # UFL expressions
    Phi_2x= Phi_2x_expr(xi)
    Phi_4x= Phi_4x_expr(xi)

    xi_x1 = xi_expr(-0.5*Lx,t) # a constant
    xi_x2 = xi_expr(0.5*Lx,t)  # a constant

    Phi_x1    = Phi_expr(xi_x1) # constants
    Phi_2x_x1 = Phi_2x_expr(xi_x1)
    Phi_4x_x1 = Phi_4x_expr(xi_x1)
    h_x1 = h_ex_expr(xi_x1)

    Phi_x2    = Phi_expr(xi_x2) # constants
    Phi_2x_x2 = Phi_2x_expr(xi_x2)
    Phi_4x_x2 = Phi_4x_expr(xi_x2)
    h_x2 = h_ex_expr(xi_x2)

    U_0 = (phi_ex_expr(Phi_x2, Phi_2x_x2, Phi_4x_x2, h_x2)-phi_ex_expr(Phi_x1, Phi_2x_x1, Phi_4x_x1, h_x1))/Lx
    c_0 = 0.5*(phi_ex_expr(Phi_x2, Phi_2x_x2, Phi_4x_x2, h_x2)-phi_ex_expr(Phi_x1, Phi_2x_x1, Phi_4x_x1, h_x1))
    phi_corr_exp = phi_corr(x[0],U_0,c_0) # corr at z* = H0, a UFL expression

    corr_nz.interpolate(phi_corr_exp) # corr at z* = H0
    U0_nz.interpolate(Constant(U_0))  # U0 at z* = H0
    c0_nz.interpolate(Constant(c_0))
    
    tilde_phi_expr(psi_1_n0, Phi, Phi_2x, Phi_4x, phi_corr_exp, h_n0) # initialise psi_1^n

    if scheme=='SV': w_half.sub(0).assign(psi_1_n0)

    #_____________________ Velocity pot. in depth: phi(x,y,z<h;t) _____________________#
    for i in range(0,n_z):    
        z_star=float(z_k[i+1]) # after coordinate transformation z*, a constant float number
        z_bt.assign((z_star/H0)*h_n0) # before coordinate transformation z, a function of x
        
        z_x1 = (z_star/H0)*h_ex_expr(xi_x1)  # z_bt.at(-Lx/2)
        z_x2 = (z_star/H0)*h_ex_expr(xi_x2)  # z_bt.at(Lx/2)

        U_0_i = (phi_ex_expr(Phi_x2, Phi_2x_x2, Phi_4x_x2, z_x2)-phi_ex_expr(Phi_x1, Phi_2x_x1, Phi_4x_x1, z_x1))/Lx
        c_0_i = 0.5*(phi_ex_expr(Phi_x2, Phi_2x_x2, Phi_4x_x2, z_x2)-phi_ex_expr(Phi_x1, Phi_2x_x1, Phi_4x_x1, z_x1))
        phi_corr_exp = phi_corr(x[0], U_0_i, c_0_i) # correction function at z*_i, UFL expression
        
        tilde_phi_expr(phii_z, Phi, Phi_2x, Phi_4x, phi_corr_exp, z_bt) # store this layer's in phii_z
    
        hat_corr.dat.data[:,i] = interpolate(phi_corr_exp,V).dat.data[:]
        hat_U0.dat.data[:,i] = interpolate(Constant(U_0_i),V).dat.data[:]
        
        hat_psi_n0.dat.data[:,i] = phii_z.dat.data[:]   # initialise psi_i^n
        w_n1.sub(1).dat.data[:,i] = phii_z.dat.data[:]  # initialise psi_i^{*}

        if scheme=='SV': w_half.sub(1).dat.data[:,i] = phii_z.dat.data[:]

else: # soliton == 'SP2' or 'SP3'
    #____________________________ Initialization of Depth _______________________________#
    h_n0.interpolate(h_ex_expr(x[0],x[1]))
    w_n1.sub(0).assign(h_n0)
    
    #____________________ Velocity pot. at the surface: phi(x,y,z=h;t0) _________________#
    Phi_1 = Phi_expr(x[0],x[1]) # UFL expressions
    Phi_2 = Phi_2ed_expr(x[0],x[1])
    Phi_3 = Phi_4th_expr(x[0],x[1])
    
    xl = -Lx*0.5
    xr = Lx*0.5

    Phi_1_x1 = Phi_expr(xl,x[1]) # UFL expressions
    Phi_2_x1 = Phi_2ed_expr(xl,x[1])
    Phi_3_x1 = Phi_4th_expr(xl,x[1])
    h_x1 = h_ex_expr(xl,x[1])

    Phi_1_x2 = Phi_expr(xr,x[1])
    Phi_2_x2 = Phi_2ed_expr(xr,x[1])
    Phi_3_x2 = Phi_4th_expr(xr,x[1])
    h_x2 = h_ex_expr(xr,x[1])

    # U_0, c_0 and phi_corri are ufl expressions
    U_0 = (phi_ex_expr(Phi_1_x2,Phi_2_x2,Phi_3_x2,h_x2)-phi_ex_expr(Phi_1_x1,Phi_2_x1,Phi_3_x1,h_x1))/Lx
    c_0 = 0.5*(phi_ex_expr(Phi_1_x1,Phi_2_x1,Phi_3_x1,h_x1)+phi_ex_expr(Phi_1_x2,Phi_2_x2,Phi_3_x2,h_x2))
    phi_corr_exp = phi_corr(x[0],U_0,c_0)
    
    corr_nz.interpolate(phi_corr_exp) # corr at z*=H0, or z=h0
    U0_nz.interpolate(U_0)
    c0_nz.interpolate(c_0)

    tilde_phi_expr(psi_1_n0, Phi_1, Phi_2, Phi_3, phi_corr_exp, h_n0)

    if scheme=='SV': w_half.sub(0).assign(psi_1_n0)

    #_____________________ Velocity pot. in depth: phi(x,y,z<h;t) _____________________#
    for i in range(0,n_z):
        z_star = float(z_k[i+1])     # after coordinate transformation z*, a constant
        z_bt.assign((z_star/H0)*h_n0) # before coordinate transformation z, a function of x,y
        
        z_x1 = (z_star/H0)*h_ex_expr(xl,x[1])  # z_bt(x1,y)
        z_x2 = (z_star/H0)*h_ex_expr(xr,x[1])  # z_bt(x2,y)

        U_0_i = (phi_ex_expr(Phi_1_x2,Phi_2_x2,Phi_3_x2,z_x2)-phi_ex_expr(Phi_1_x1,Phi_2_x1,Phi_3_x1,z_x1))/Lx
        c_0_i = 0.5*(phi_ex_expr(Phi_1_x1,Phi_2_x1,Phi_3_x1,z_x1)+phi_ex_expr(Phi_1_x2,Phi_2_x2,Phi_3_x2,z_x2))
        phi_corr_exp = phi_corr(x[0], U_0_i, c_0_i) # correction function at z*_i, UFL expression
        
        tilde_phi_expr(phii_z, Phi_1, Phi_2, Phi_3, phi_corr_exp, z_bt)
    
        hat_corr.dat.data[:,i] = interpolate(phi_corr_exp,V).dat.data[:]
        hat_U0.dat.data[:,i] = interpolate(U_0_i,V).dat.data[:]
        
        hat_psi_n0.dat.data[:,i] = phii_z.dat.data   # psi_i^n
        w_n1.sub(1).dat.data[:,i] = phii_z.dat.data  # psi_i^{*}

        if scheme=='SV': w_half.sub(1).dat.data[:,i]= phii_z.dat.data


PETSc.Sys.Print('...functions initialised!')

#pdb.set_trace()

PETSc.Sys.Print('Initialisation of the solvers...')

"""
    ************************************************************************************************************************
    *                                                   Weak Formulations                                                  *
    ************************************************************************************************************************ """

if scheme=="SE": #_____________________________________________ Symplectic-Euler ______________________________________________#
    #------------------------ Step 1 : Update h at time t^{n+1} and psi_i at time t^* simulataneously: ------------------------#
    WF_h_psi = SP_solvers.WF_h_SE(dim, n_z, g, H0, dt, b, FWF, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_n0, hat_psi_star,
                                  C11, CN1, CNN, B11, B1N, BN1, BNN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN,
                                  U0_nz, hat_U0, corr_nz, hat_corr)

    #----------------------------------------- Step 2 : Update psi_1 at time t^{n+1}: -----------------------------------------#
    A_psi_s, L_psi_s = SP_solvers.WF_psi_SE(dim, g, H0, H, dt, b, FWF, delta_h, psi_1, psi_1_n0, hat_psi_star, h_n1, 
                                            C11, CN1, CNN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN,
                                            U0_nz, hat_U0, corr_nz, hat_corr)

    #----------------------------------------- Step 3 : Update psi_i at time t^{n+1}: -----------------------------------------#
    A_hat, L_hat = SP_solvers.WF_hat_psi_SE_SV(dim, H0, n_z, dt, b, FWF, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                                               MN1, MNN, D1N, DN1, DNN, B1N, BN1, BNN, SN1, SNN, AN1, ANN, CN1, CNN,
                                               U0_nz, hat_U0, corr_nz, hat_corr)

elif scheme=="SV":#______________________________________________ Stormer-Verlet ______________________________________________#
    #--------------------------------------- Step 1 : Update psi_1^{n+1/2} and psi_i^*: ---------------------------------------#
    WF_psi_star = SP_solvers.WF_psi_half_SV(dim, n_z, g, H, H0, dt, b, FWF, delta_h_sv, delta_hat_psi_sv, psi_1_n0, psi_1_half, hat_psi_star, h_n0, 
                                            C11, CN1, CNN, B11, B1N, BN1, BNN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN,
                                            U0_nz, hat_U0, corr_nz, hat_corr)

    #----------------------------- Step 2 : Update h^{n+1} and psi_i at time t^** simulataneously: ----------------------------#
    WF_h_psi = SP_solvers.WF_h_SV(dim, n_z, H0, g, dt, b, FWF, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_half, hat_psi_star, hat_psi_aux, 
                                  C11, CN1, CNN, B11, B1N, BN1, BNN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, 
                                  U0_nz, hat_U0, corr_nz, hat_corr)

    #----------------------------------------- Step 3 : Update psi_1 at time t^{n+1}: -----------------------------------------#
    a_psi_1, L_psi_1 = SP_solvers.WF_psi_n1_SV(dim, H0, H, g, dt, b, FWF, delta_h, psi_1_half, psi_1, hat_psi_aux, h_n1, 
                                               C11, CN1, CNN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN,
                                               U0_nz, hat_U0, corr_nz, hat_corr)

    #----------------------------------------- Step 4 : Update psi_i at time t^{n+1}: -----------------------------------------#
    A_hat, L_hat = SP_solvers.WF_hat_psi_SE_SV(dim, H0, n_z, dt, b, FWF, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                                               MN1, MNN, D1N, DN1, DNN, B1N, BN1, BNN, SN1, SNN, AN1, ANN, CN1, CNN,
                                               U0_nz, hat_U0, corr_nz, hat_corr)
"""
    **************************************************************************************
    *                                 Define the solvers                                 *
    ************************************************************************************** """

#____________________________________ Solvers parameters ____________________________________#

lines_parameters = {'ksp_type': 'gmres',
                    'ksp_monitor': None,
                    'ksp_converged_reason': None,
                    'pc_type': 'python',
                    'pc_python_type': 'firedrake.ASMStarPC',
                    'snes_lag_preconditioner_persists': None,
                    'snes_lag_preconditioner': 5,
                    'star_construct_dim': 2, # 1
                    'star_sub_sub_pc_type': 'lu',
                    'star_sub_sub_pc_factor_mat_ordering_type': 'rcm'}

# default
param_h       = {'ksp_converged_reason':None}               
param_psi     = {'ksp_converged_reason':None}
param_hat_psi = {'ksp_converged_reason':None}

# https://www.firedrakeproject.org/solving-interface.html#solving-linear-systems
# 'ksp_view': None => print the solver parameters for linear solves
# 'snes_view': None => print the solver options for nonlinear solves

#--------------------------------------------------------------------------------------------#
#                                      Symplectic-Euler                                      #
#____________________________________________________________________________________________#
if scheme=="SE":
    #_______________________ Variational solver for h (and hat_psi^*) _______________________#
    h_problem = NonlinearVariationalProblem(WF_h_psi, w_n1)
    h_solver = NonlinearVariationalSolver(h_problem, options_prefix="h_dt_imp", solver_parameters=param_h)

    #_____________________________ Variational solver for psi_1 _____________________________#
    psi_problem = LinearVariationalProblem(A_psi_s, L_psi_s, psi_1_n1)
    # yl comment:
    # In this linear solver the trial function is psi_1.
    # psi_1_n1 is a function holding the solution, or we place the solution in psi_1_n1.
    psi_solver = LinearVariationalSolver(psi_problem, options_prefix="psi1_dt_exp", solver_parameters=param_psi)

    #____________________________ Variational solver for hat_psi ____________________________#
    hat_psi_problem = LinearVariationalProblem(A_hat, L_hat, hat_psi_n0)
    hat_psi_solver = LinearVariationalSolver(hat_psi_problem, options_prefix="hat_psi_exp", solver_parameters=param_hat_psi)

#--------------------------------------------------------------------------------------------#
#                                       Stormer-Verlet                                       #
#____________________________________________________________________________________________#
if scheme=="SV":
    #_______________________ Variational solver for psi_1^{n+1/2} (and hat_psi^*) _______________________#
    psi_half_problem = NonlinearVariationalProblem(WF_psi_star, w_half)
    psi_half_solver = NonlinearVariationalSolver(psi_half_problem, options_prefix="psi1_dt2_imp", solver_parameters=param_h)
    
    #____________________________ Variational solver for h^{n+1} (and hat_psi^**) _______________________#
    h_problem = NonlinearVariationalProblem(WF_h_psi, w_n1)
    h_solver = NonlinearVariationalSolver(h_problem, options_prefix="h_dt_imp", solver_parameters=param_h)
    
    #_______________________ Variational solver for psi_1^{n+1} _______________________#
    psi_n1_problem = LinearVariationalProblem(a_psi_1, L_psi_1, psi_1_n1)
    psi_n1_solver = LinearVariationalSolver(psi_n1_problem, options_prefix="psi1_dt_exp", solver_parameters=param_psi)
    
    #____________________________ Variational solver for hat_psi ____________________________#
    hat_psi_problem = LinearVariationalProblem(A_hat, L_hat, hat_psi_n0)
    hat_psi_solver = LinearVariationalSolver(hat_psi_problem, options_prefix="hat_psi_exp", solver_parameters=param_hat_psi)

PETSc.Sys.Print('...solvers initialised!')

"""
    *************************************************************
    *               Preparation for saving results              *
    ************************************************************* """
if save_pvd:
    outfile = File(os.path.join(save_path, 'soliton.pvd'))
energy_file = os.path.join(save_path, 'energy.csv')
energy_data = np.empty((0,2)) # create an empty 2d array for saving E(t)


PETSc.Sys.Print('Update of the solutions:')
""" *********************************************************************************
    *                                   Time loop                                   *
    ********************************************************************************* """
t_save = t
smallfac = 1e-10

#pdb.set_trace()

while t<=Tend+smallfac:
    """ *****************************************************************************
        *                                Save Results                               *
        ***************************************************************************** """
    if t_save-smallfac < t:

        progress = format(100*(t-T0)/(Tend-T0), '.3f')+' %'
        tt = format(t, '.3f')
        PETSc.Sys.Print('t= %s, Progress: %s' % (tt, progress))
        #or PETSc.Sys.Print('Progress(%%): ', 100*t/Tend) or PETSc.Sys.Print('Progress', 100*t/Tend,' %%')
        
        #------ Track the total energy of the system and save it ------#
        energy = save_energy(dim, g, H0, H, b, FWF, h_n0, psi_1_n0, hat_psi_n0, U0_nz, hat_U0, corr_nz, hat_corr,
                             C11, CN1, CNN, B11, B1N, BN1, BNN, A11, AN1, ANN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN)
        
        energy_data = np.append(energy_data,np.array([[t,energy]]),axis=0)
        
        '''
        #------ Store the field data to a numpy array ------#
        field_data = np.empty((Nx+1,4))
        for row_i in range(Nx+1):
            x_i = -0.5*Lx+row_i*(Lx/Nx) # x coordinates
            field_data[row_i,:]=np.array([x_i, h_n0.at(x_i), psi_1_n0.at(x_i), hat_psi_n0.at(x_i)[-1]])

        #------ Save the array to a binary file (.npy) ------#
        op_file = os.path.join(save_path, tt)
        np.save(op_file, field_data)
        
        #------ Save the array to a text file (.csv) ------#
        #op_file_name = tt + '.csv'
        #op_file = os.path.join(save_path, op_file_name)
        #np.savetxt(op_file, field_data, fmt='%10.3f '+' %.18e '*3)
        '''
        if save_pvd:
            outfile.write(h_n0,psi_1_n0,time=t)
    
        t_save+=dt_save
        
    """ *********************************************************************
        *                          Main time loop                           *
        ********************************************************************* """

    t_half = t+0.5*dt
    t += dt
    
    #_____________ Solve the weak formulations ______________#
    if scheme=="SE":                     # 1st-order SE scheme
        h_solver.solve()           # get h^{n+1} and hat_psi^*
        psi_solver.solve()                     # get psi^{n+1}
    elif scheme=="SV":                   # 2nd-order SV scheme
        psi_half_solver.solve() # get psi^{n+1/2} and hat_psi^*
        h_solver.solve()        # get h^{n+1} and hat_psi^{**}
        psi_n1_solver.solve()                  # get psi^{n+1}
    
    #_________________ Update the solutions _________________#
    h_out, hat_psi_out = w_n1.subfunctions
    h_n0.assign(h_out)
    psi_1_n0.assign(psi_1_n1)
    hat_psi_n0.assign(hat_psi_out)

    #_______ Update the interior veolicty potentials ________#
    hat_psi_solver.solve() 

np.savetxt(energy_file, energy_data, fmt='%10.3f '+' %.18e ')

comp_time = time.perf_counter()-start_time
jours = int(comp_time/(24*3600))
heures = int((comp_time-jours*24*3600)/3600)
minutes = int((comp_time-jours*24*3600-heures*3600)/60)
secondes = comp_time -jours*24*3600-heures*3600 - minutes*60
README_file = open(os.path.join(save_path, 'readme.txt'), 'w')
save_README(README_file, dim, soliton, Amp, tan0, epsilon, mu, Lx, Ly, H0, res_x, res_y, Nx, Ny, Nz,
            nCG, scheme, T0, t, dt, dt_save, jours, heures, minutes, secondes, comp_time)
