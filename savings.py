# All the terms related to the wave maker motion are deleted, ie, U, V, W, etc.
# Set b=0 in 2D WFs but retain b in 3D WFs.
# For 3D WFs, to implement MSA: FWF=0 and Bij=Cij=0.

from firedrake import *
from pyop2.mpi import COMM_WORLD

""" *******************************************************
    *                  Compute the energy                 *
    ******************************************************* """

def save_energy(dim, g, H0, H, b, FWF, h_n0, psi_1_n0, hat_psi_n0, U0_nz, hat_U0, corr_nz, hat_corr,
                C11, CN1, CNN, B11, B1N, BN1, BNN, A11, AN1, ANN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN):

    if dim =="3D":
        # checked against notes. 2023.6.26
        energy = assemble(( 0.5*h_n0 * ( (psi_1_n0.dx(0)+U0_nz)**2*M11 \
                             +dot((hat_psi_n0.dx(0)+hat_U0),         (2.0*MN1*(psi_1_n0.dx(0)+U0_nz)        +dot(MNN,(hat_psi_n0.dx(0)+hat_U0)))) )\
                           +0.5*h_n0 * ( ((psi_1_n0.dx(1)+corr_nz.dx(1))**2)*M11 \
                             +dot((hat_psi_n0.dx(1)+hat_corr.dx(1)), (2.0*MN1*(psi_1_n0.dx(1)+corr_nz.dx(1))+dot(MNN,(hat_psi_n0.dx(1)+hat_corr.dx(1))))) )\
                         -h_n0.dx(0) * ( (psi_1_n0.dx(0)+U0_nz)*(D11*(psi_1_n0+corr_nz) + dot(D1N,(hat_psi_n0+hat_corr))) \
                                           +dot((hat_psi_n0.dx(0)+hat_U0), (DN1*(psi_1_n0+corr_nz) + dot(DNN,(hat_psi_n0+hat_corr)))) ) \
                         -h_n0.dx(1) * ( (psi_1_n0.dx(1)+corr_nz.dx(1))*(D11*(psi_1_n0+corr_nz) + dot(D1N,(hat_psi_n0+hat_corr))) \
                                           +dot((hat_psi_n0.dx(1)+hat_corr.dx(1)), (DN1*(psi_1_n0+corr_nz) + dot(DNN,(hat_psi_n0+hat_corr)))) ) \
                         -H0*b.dx(0) * ( (psi_1_n0.dx(0)+U0_nz)*(B11*(psi_1_n0+corr_nz) + dot(B1N,(hat_psi_n0+hat_corr))) \
                                           +dot((hat_psi_n0.dx(0)+hat_U0), (BN1*(psi_1_n0+corr_nz) + dot(BNN, (hat_psi_n0+hat_corr)))) ) \
                         -H0*b.dx(1) * ( (psi_1_n0.dx(1)+corr_nz.dx(1))*(B11*(psi_1_n0+corr_nz) + dot(B1N,(hat_psi_n0+hat_corr))) \
                                           +dot((hat_psi_n0.dx(1)+hat_corr.dx(1)), (BN1*(psi_1_n0+corr_nz) + dot(BNN, (hat_psi_n0+hat_corr))))) \
                     +0.5*(1.0/h_n0) * (S11*(psi_1_n0+corr_nz)**2 + dot((hat_psi_n0+hat_corr), (2.0*SN1*(psi_1_n0+corr_nz)+dot(SNN,(hat_psi_n0+hat_corr)))))*\
                                       (h_n0.dx(0)**2 + h_n0.dx(1)**2)
                 +(H0*H0/(2.0*h_n0)) * (A11*(psi_1_n0+corr_nz)**2 + dot((hat_psi_n0+hat_corr), (2.0*AN1*(psi_1_n0+corr_nz)+dot(ANN,(hat_psi_n0+hat_corr)))))*\
                                       (FWF*(b.dx(0)**2+b.dx(1)**2) + 1.0)\
                          +(H0/h_n0) * (C11*(psi_1_n0+corr_nz)**2 + dot((hat_psi_n0+hat_corr), (2.0*CN1*(psi_1_n0+corr_nz)+dot(CNN,(hat_psi_n0+hat_corr)))))*\
                                       (b.dx(0)*h_n0.dx(0) + b.dx(1)*h_n0.dx(1))\
                            + g*H0*h_n0*(0.5*h_n0-H) )*dx) 
    elif dim=="2D":
        energy = assemble(( 0.5*h_n0 * ( (psi_1_n0.dx(0)+U0_nz)**2*M11 \
                                        +dot((hat_psi_n0.dx(0)+hat_U0), (2.0*MN1*(psi_1_n0.dx(0)+U0_nz)+dot(MNN,(hat_psi_n0.dx(0)+hat_U0)))) ) \
                         -h_n0.dx(0) * ( (psi_1_n0.dx(0)+U0_nz)*(D11*(psi_1_n0+corr_nz) + dot(D1N,(hat_psi_n0+hat_corr)))\
                                            + dot((hat_psi_n0.dx(0)+hat_U0),(DN1*(psi_1_n0+corr_nz)+dot(DNN,(hat_psi_n0+hat_corr)))) )\
          + 0.5*(h_n0.dx(0)**2/h_n0) * (S11*(psi_1_n0+corr_nz)**2 + dot((hat_psi_n0+hat_corr), (2.0*SN1*(psi_1_n0+corr_nz)+dot(SNN,(hat_psi_n0+hat_corr)))))\
                  + 0.5*(H0*H0/h_n0) * (A11*(psi_1_n0+corr_nz)**2 + dot((hat_psi_n0+hat_corr), (2.0*AN1*(psi_1_n0+corr_nz)+dot(ANN,(hat_psi_n0+hat_corr)))))\
                           + g*H0*h_n0*(0.5*h_n0-H0) )*dx)       
    return energy


def save_README(README_file, dim, soliton, Amp, tan0, epsilon, mu, Lx, Ly, H0, res_x, res_y, Nx, Ny, Nz,
                nCG, scheme, T0, t, dt, dt_save, jours, heures, minutes, secondes, comp_time):
    README_file.write('======================================\n')
    README_file.write('    Summary of the %s test case       \n' % soliton)
    README_file.write('======================================\n')

    README_file.write('\n-------- Soliton information ---------\n')
    README_file.write('ε = %g, µ = %g\n' %(epsilon,mu))
    if soliton=='SP1':
        Amp_eta = epsilon*H0*Amp
        README_file.write('Amplitude: %g [m] (A = %g)\n' %(Amp_eta,Amp))
    elif soliton=='SP2':
        c1 = pow(9/2,1/6)
        c2 = pow(4/3,1/3)
        k4 = tan0/(c1*sqrt(epsilon))
        Amp_eta = epsilon*H0*c2*(0.5*k4**2)
        max_A = 4*Amp_eta
        README_file.write('k4 = -k1 = %.4f\n' %(k4))
        README_file.write('Far-field amplitude: %.4f [m] \n' %(Amp_eta))
        README_file.write('Maximum amplitude: %.4f [m] \n' %(max_A))

    README_file.write('\n------ Dimensions of the domain ------\n')
    if dim=="2D":
        README_file.write('Length Lx = %g [m]\n' %(Lx))
        README_file.write('Depth H0 = %g [m]\n' %(H0))
    elif dim=="3D":
        README_file.write('[x1,x2] = [%.2f, %.2f] (%.3f [m])\n' %(-Lx/2,Lx/2,Lx))
        README_file.write('[y1,y2] = [%.2f, %.2f] (%.3f [m])\n' %(-Ly/2,Ly/2,Ly))
        README_file.write('Depth H0 = %g [m]\n' %(H0))

    README_file.write('\n----------- Mesh resolution ----------\n')
    README_file.write('Δx = %g [m] (%d elements, CG%d)\n' % (res_x, Nx, nCG))
    if dim=="3D":
        README_file.write('Δy = %g [m] (%d elements)\n' % (res_y,Ny))
    README_file.write('In z-direction: %d layers\n' % (Nz))

    README_file.write('\n--------------- Solver ---------------\n')
    if scheme=="SE":
        README_file.write('1st order Symplectic-Euler scheme\n')
    else:
        README_file.write('2nd order Störmer-Verlet scheme\n')

    README_file.write('\n---------------- Time ----------------\n')
    README_file.write('Start at t = %.3f [s]\n' %(T0))
    README_file.write('End   at t = %.3f [s]\n' %(t)) # note that this is larger than T_end
    README_file.write('Time resolution: Δt = %g [s]\n' %(dt))
    README_file.write('Data were saved every %g [s]\n\n' %(dt_save))
    README_file.write('Computational time on %d process(es):\n' % COMM_WORLD.size)
    README_file.write('\t%d [d] %d [h] %d [min] %.1f [s]\n' %(jours, heures, minutes, secondes))
    README_file.write('That is in total: %.2f [s]\n' %(comp_time))