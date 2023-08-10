# All the terms related to the wave maker motion are deleted, ie, U, V, W, etc.
# Flat bottom in 2D and 3D WFs.
# Remove C, B, FWF, b.

from firedrake import *

"""
    ************************************************************************************************************************
    *                                  Weak Formulations for the symplectic Euler scheme                                   *
    ************************************************************************************************************************ """
#--------------------------------------------------------------------------------------------------------------------------#
#                         Step 1 : Update h at time t^{n+1} and psi_i at time t^* simulataneously:                         #
#__________________________________________________________________________________________________________________________#

# dim, n_z, g, H0, dt: constant
# U0_nz, hat_U0, corr_nz, hat_corr: for x-periodicity
# delta_psi, delta_hat_star: test functions [shape(delta_psi)=(), ufl.indexed.Indexed; shape(delta_hat_star)=(n_z,), ufl.tensors.ListTensor]
# h_n0, psi_1_n0: at n time level, known [shape(psi_1_n0.dx(0))=(), ufl.indexed.Indexed]
# >>> h_n1, hat_psi_star = split(w_n1): unknown [shape(hat_pasi_star)=(n_z,), len(hat_psi_star)=n_z]

def WF_h_SE(dim, n_z, g, H0, dt, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_n0, hat_psi_star,
            M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN,
            U0_nz, hat_U0, corr_nz, hat_corr):
    
    if dim == "3D":
        # checked against notes 2023.6.12

        WF_h = (H0*delta_psi*(h_n1-h_n0)/dt \
                -( h_n1*(        (psi_1_n0.dx(0)+U0_nz)*M11 +         dot((hat_psi_star.dx(0)+hat_U0),MN1))*delta_psi.dx(0)\
                  +h_n1*((psi_1_n0.dx(1)+corr_nz.dx(1))*M11 + dot((hat_psi_star.dx(1)+hat_corr.dx(1)),MN1))*delta_psi.dx(1)\
                  -h_n1.dx(0)* (delta_psi.dx(0)*(D11*(psi_1_n0+corr_nz) + dot(D1N,(hat_psi_star+hat_corr))) \
                                     +delta_psi*((psi_1_n0.dx(0)+U0_nz)*D11 + dot((hat_psi_star.dx(0)+hat_U0),DN1)))\
                  -h_n1.dx(1)* (delta_psi.dx(1)*(D11*(psi_1_n0+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                     +delta_psi*((psi_1_n0.dx(1)+corr_nz.dx(1))*D11 + dot((hat_psi_star.dx(1)+hat_corr.dx(1)),DN1)))\
                  +(delta_psi/h_n1)*(  (S11*(psi_1_n0+corr_nz)+dot(SN1,(hat_psi_star+hat_corr))) * (h_n1.dx(0)**2 + h_n1.dx(1)**2)\
                                     + H0*H0*(A11*(psi_1_n0+corr_nz)+dot(AN1,(hat_psi_star+hat_corr))) ) ))*dx 
        
        WF_hat_psi_star =   h_n1 * elem_mult(delta_hat_star.dx(0),(MN1*(psi_1_n0.dx(0)+U0_nz)        +dot(MNN,(hat_psi_star.dx(0)+hat_U0))))\
                          + h_n1 * elem_mult(delta_hat_star.dx(1),(MN1*(psi_1_n0.dx(1)+corr_nz.dx(1))+dot(MNN,(hat_psi_star.dx(1)+hat_corr.dx(1)))))\
                          - h_n1.dx(0)*( elem_mult(delta_hat_star,((psi_1_n0.dx(0)+U0_nz)*D1N + dot(DNN.T,(hat_psi_star.dx(0)+hat_U0))))\
                                        +elem_mult(delta_hat_star.dx(0),(DN1*(psi_1_n0+corr_nz) + dot(DNN,(hat_psi_star+hat_corr)))))\
                          - h_n1.dx(1)*( elem_mult(delta_hat_star,((psi_1_n0.dx(1)+corr_nz.dx(1))*D1N + dot(DNN.T,(hat_psi_star.dx(1)+hat_corr.dx(1)))))\
                                        +elem_mult(delta_hat_star.dx(1),(DN1*(psi_1_n0+corr_nz) + dot(DNN,(hat_psi_star+hat_corr)))))\
                 + (1.0/h_n1) * elem_mult(delta_hat_star,(SN1*(psi_1_n0+corr_nz)+dot(SNN,(hat_psi_star+hat_corr)))) * (h_n1.dx(0)**2 + h_n1.dx(1)**2)\
               + (H0*H0/h_n1) * elem_mult(delta_hat_star,(AN1*(psi_1_n0+corr_nz)+dot(ANN,(hat_psi_star+hat_corr))))
    
    elif dim == "2D":
        # check against the notes 2023.5.25

        WF_h = (H0*delta_psi*(h_n1-h_n0)/dt \
              -( h_n1*((psi_1_n0.dx(0)+U0_nz)*M11 + dot((hat_psi_star.dx(0)+hat_U0),MN1))*delta_psi.dx(0) \
                -h_n1.dx(0)*( delta_psi.dx(0)*(D11*(psi_1_n0 +corr_nz) + dot(D1N,(hat_psi_star+hat_corr))) \
                             +delta_psi*(D11*(psi_1_n0.dx(0)+U0_nz) + dot((hat_psi_star.dx(0)+hat_U0),DN1)))\
                +(h_n1.dx(0)**2/h_n1)*((psi_1_n0+corr_nz)*S11 + dot((hat_psi_star+hat_corr),SN1))*delta_psi\
                        +(H0*H0/h_n1)*((psi_1_n0+corr_nz)*A11 + dot((hat_psi_star+hat_corr),AN1))*delta_psi ))*dx

        WF_hat_psi_star= ( h_n1*elem_mult(delta_hat_star.dx(0),(MN1*(psi_1_n0.dx(0)+U0_nz)+dot(MNN,(hat_psi_star.dx(0)+hat_U0))))\
                          -h_n1.dx(0)*(elem_mult(delta_hat_star, ((psi_1_n0.dx(0)+U0_nz)*D1N+ dot(DNN.T,hat_psi_star.dx(0)+hat_U0))) \
                                       + elem_mult(delta_hat_star.dx(0),(DN1*(psi_1_n0+corr_nz)+dot(DNN,hat_psi_star+hat_corr)))) \
                          +(h_n1.dx(0)**2/h_n1)*elem_mult(delta_hat_star,(SN1*(psi_1_n0+corr_nz)+ dot(SNN,hat_psi_star+hat_corr)))\
                                  +(H0*H0/h_n1)*elem_mult(delta_hat_star,(AN1*(psi_1_n0+corr_nz)+ dot(ANN,hat_psi_star+hat_corr))))
    
    WF_h_psi = WF_h + sum((WF_hat_psi_star[ind])*dx for ind in range(0,n_z))

    return WF_h_psi



#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 2 : Update psi_1 at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# dim, g, H0, dt: constant
# U0_nz, hat_U0, corr_nz, hat_corr: for x-periodicity
# delta_h: test function
# psi_1_n0: at n time level, known
# h_n1, hat_psi_star: obtained from step1, known
# >>> psi_1: trial function

def WF_psi_SE(dim, g, H0, dt, delta_h, psi_1, psi_1_n0, hat_psi_star, h_n1, 
              M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN,
              U0_nz, hat_U0, corr_nz, hat_corr):
    
    if dim=="3D":
        # checked against notes. 2023.6.12
        # G1(corr_nz) is omitted in the first term on both sides

        A_psi_s = (H0*delta_h*psi_1)*dx
        
        L_psi_s = (H0*delta_h*psi_1_n0 \
                 -dt*( 0.5*delta_h*( ((psi_1_n0.dx(0)+U0_nz)**2)*M11 \
                                  +dot((hat_psi_star.dx(0)+hat_U0),         (2.0*MN1*(psi_1_n0.dx(0)+U0_nz)        +dot(MNN,(hat_psi_star.dx(0)+hat_U0))))\
                                     +((psi_1_n0.dx(1)+corr_nz.dx(1))**2)*M11 \
                                  +dot((hat_psi_star.dx(1)+hat_corr.dx(1)), (2.0*MN1*(psi_1_n0.dx(1)+corr_nz.dx(1))+dot(MNN,(hat_psi_star.dx(1)+hat_corr.dx(1))))) )\
                    -delta_h.dx(0)*( (psi_1_n0.dx(0)+U0_nz)        *(D11*(psi_1_n0+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                       +dot((hat_psi_star.dx(0)+hat_U0),         (DN1*(psi_1_n0+corr_nz) + dot(DNN, (hat_psi_star+hat_corr)))))\
                    -delta_h.dx(1)*( (psi_1_n0.dx(1)+corr_nz.dx(1))*(D11*(psi_1_n0+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                       +dot((hat_psi_star.dx(1)+hat_corr.dx(1)), (DN1*(psi_1_n0+corr_nz) + dot(DNN, (hat_psi_star+hat_corr)))))\
                       +(1.0/h_n1)* ( delta_h.dx(0)*h_n1.dx(0) + delta_h.dx(1)*h_n1.dx(1) - (delta_h/(2.0*h_n1))*(h_n1.dx(0)**2+h_n1.dx(1)**2) )* \
                                    ( (psi_1_n0+corr_nz)**2*S11 + 2.0*dot((hat_psi_star+hat_corr),SN1)*(psi_1_n0+corr_nz) \
                                                                    + dot((hat_psi_star+hat_corr),dot(SNN,(hat_psi_star+hat_corr))) )\
   -(0.5*delta_h*H0*H0/(h_n1**2)) * ( (psi_1_n0+corr_nz)**2*A11 + 2.0*dot((hat_psi_star+hat_corr),AN1)*(psi_1_n0+corr_nz) \
                                                                    + dot((hat_psi_star+hat_corr),dot(ANN,(hat_psi_star+hat_corr))) ) \
                       + g*H0*delta_h*(h_n1 - H0) ) )*dx

    elif dim=="2D":
        
        A_psi_s = (H0*delta_h*psi_1)*dx
        
        L_psi_s = (H0*delta_h*psi_1_n0 \
                  -dt * (delta_h*0.5*(((psi_1_n0.dx(0)+U0_nz)**2)*M11 + dot(hat_psi_star.dx(0)+hat_U0, (2.0*MN1*(psi_1_n0.dx(0)+U0_nz)\
                                                                      + dot(MNN,(hat_psi_star.dx(0)+hat_U0)))) )\
                         -delta_h.dx(0)*( (psi_1_n0.dx(0)+U0_nz)*(D11*(psi_1_n0+corr_nz) + dot(D1N,(hat_psi_star+hat_corr))) \
                                         + dot((hat_psi_star.dx(0)+hat_U0), (DN1*(psi_1_n0+corr_nz) + dot(DNN, (hat_psi_star+hat_corr)))))\
                         +(h_n1.dx(0)/h_n1)*(delta_h.dx(0)-(delta_h/h_n1)*h_n1.dx(0)*0.5)* ((psi_1_n0+corr_nz)**2*S11 \
                                    + 2.0*dot((hat_psi_star+hat_corr),SN1)*(psi_1_n0+corr_nz) + dot((hat_psi_star+hat_corr),dot(SNN,(hat_psi_star+hat_corr))))\
                         -(0.5*delta_h*H0*H0/(h_n1**2))*((psi_1_n0+corr_nz)**2*A11 \
                                    + 2.0*dot((hat_psi_star+hat_corr),AN1)*(psi_1_n0+corr_nz) + dot((hat_psi_star+hat_corr),dot(ANN,(hat_psi_star+hat_corr))))\
                         +g*H0*delta_h*(h_n1-H0) ))*dx

    return A_psi_s, L_psi_s


#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 3 : Update psi_i at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# dim, H0, n_z, dt: constants
# U0_nz, hat_U0, corr_nz, hat_corr: for x-periodicity
# delta_hat_psi: test function
# h_n0, psi_1_n0: at (n+1) time level, known
# >>> hat_psi: trial function (the solution is placed in hat_psi_n0)

def WF_hat_psi_SE_SV(dim, H0, n_z, dt, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                     MN1, MNN, D1N, DN1, DNN, SN1, SNN, AN1, ANN,
                     U0_nz, hat_U0, corr_nz, hat_corr):
    
    if dim=="3D":
        # Rearrange (C.2): separate _i' (unknown, a) from _1 (known, L) and replace h_n1 with h_n0.
        # checked against notes. 2023.6.15
        
        a_hat_psi =  h_n0*elem_mult(delta_hat_psi.dx(0),dot(MNN,hat_psi.dx(0)))\
                    +h_n0*elem_mult(delta_hat_psi.dx(1),dot(MNN,hat_psi.dx(1)))\
                    +   (1.0/h_n0) * (    h_n0.dx(0)**2 + h_n0.dx(1)**2     ) * elem_mult(delta_hat_psi,dot(SNN,hat_psi))\
                    + (H0*H0/h_n0) * elem_mult(delta_hat_psi,dot(ANN,hat_psi))\
                    -h_n0.dx(0)* ( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(0))) + elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_psi)) )\
                    -h_n0.dx(1)* ( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(1))) + elem_mult(delta_hat_psi.dx(1),dot(DNN,hat_psi)) )
        
        L_hat_psi = -( h_n0*elem_mult(delta_hat_psi.dx(0), MN1*(psi_1_n0.dx(0)+U0_nz))\
                      +h_n0*elem_mult(delta_hat_psi.dx(1), MN1*(psi_1_n0.dx(1)+corr_nz.dx(1)))\
                      +   (1.0/h_n0) * (    h_n0.dx(0)**2 + h_n0.dx(1)**2     ) * elem_mult(delta_hat_psi,SN1*(psi_1_n0+corr_nz))\
                      + (H0*H0/h_n0) * elem_mult(delta_hat_psi,AN1*(psi_1_n0+corr_nz))\
                      -h_n0.dx(0)* ( elem_mult(delta_hat_psi, D1N*(psi_1_n0.dx(0)+U0_nz))         + elem_mult(delta_hat_psi.dx(0),DN1*(psi_1_n0+corr_nz)))\
                      -h_n0.dx(1)* ( elem_mult(delta_hat_psi, D1N*(psi_1_n0.dx(1)+corr_nz.dx(1))) + elem_mult(delta_hat_psi.dx(1),DN1*(psi_1_n0+corr_nz))) )\
                    -( h_n0*elem_mult(delta_hat_psi.dx(0),dot(MNN,hat_U0))\
                      +h_n0*elem_mult(delta_hat_psi.dx(1),dot(MNN,hat_corr.dx(1)))\
                      +   (1.0/h_n0) * (    h_n0.dx(0)**2 + h_n0.dx(1)**2     ) * elem_mult(delta_hat_psi,dot(SNN,hat_corr))\
                      + (H0*H0/h_n0) * elem_mult(delta_hat_psi,dot(ANN,hat_corr))\
                      -h_n0.dx(0)* ( elem_mult(delta_hat_psi, dot(DNN.T,hat_U0))         + elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_corr)))\
                      -h_n0.dx(1)* ( elem_mult(delta_hat_psi, dot(DNN.T,hat_corr.dx(1))) + elem_mult(delta_hat_psi.dx(1),dot(DNN,hat_corr))) )

    elif dim=="2D":
        # Note that JH moved all the knowns to the RHS. 
        # Checked against the notes. Tested.

        a_hat_psi =  h_n0*elem_mult(delta_hat_psi.dx(0),dot(MNN,hat_psi.dx(0)))\
                    -h_n0.dx(0)*( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(0)))+elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_psi)) ) \
                    +(h_n0.dx(0)**2/h_n0)*elem_mult(delta_hat_psi,dot(SNN,hat_psi))\
                            +(H0*H0/h_n0)*elem_mult(delta_hat_psi,dot(ANN,hat_psi))    
            
        L_hat_psi =-(h_n0*elem_mult(delta_hat_psi.dx(0), MN1*(psi_1_n0.dx(0)+U0_nz))\
                     -h_n0.dx(0)*(elem_mult(delta_hat_psi, D1N*(psi_1_n0.dx(0)+U0_nz))+ elem_mult(delta_hat_psi.dx(0),DN1*(psi_1_n0+corr_nz))) \
                     +(h_n0.dx(0)**2/h_n0)*elem_mult(delta_hat_psi,SN1*(psi_1_n0+corr_nz))\
                            + (H0*H0/h_n0)*elem_mult(delta_hat_psi,AN1*(psi_1_n0+corr_nz)))\
                   -(h_n0*elem_mult(delta_hat_psi.dx(0),dot(MNN,hat_U0))\
                     -h_n0.dx(0)*( elem_mult(delta_hat_psi, dot(DNN.T,hat_U0))+elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_corr)) ) \
                     +(h_n0.dx(0)**2/h_n0)*elem_mult(delta_hat_psi,dot(SNN,hat_corr))\
                             +(H0*H0/h_n0)*elem_mult(delta_hat_psi,dot(ANN,hat_corr)))

    A_hat = sum((a_hat_psi[ind])*dx for ind in range(0,n_z))
    L_hat = sum((L_hat_psi[ind])*dx for ind in range(0,n_z))

    return A_hat, L_hat

"""
    ************************************************************************************************************************
    *                                   Weak Formulations for the Stormer-Verlet scheme                                    *
    ************************************************************************************************************************ """
#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 1 : Update psi_1^{n+1/2} and psi_i^*:                                    #
#______________________________________________________________________________________________________________________#

# dim, n_z, g, H0, dt: constant
# U0_nz, hat_U0, corr_nz, hat_corr: for x-periodicity
# delta_h_sv, delta_hat_psi_sv: test functions (modified by yl)
# h_n0, psi_1_n0: at n time level, known
# >>> psi_1_half, hat_psi_star = split(w_half): unknown

def WF_psi_half_SV(dim, n_z, g, H0, dt, delta_h_sv, delta_hat_psi_sv, psi_1_n0, psi_1_half, hat_psi_star, h_n0, 
                   M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN,
                   U0_nz, hat_U0, corr_nz, hat_corr):

    if dim=="3D":
        # checked against notes and SE step2. 2023.6.16

        WF_psi_s = ( (H0*delta_h_sv*(psi_1_half - psi_1_n0))/(0.5*dt)\
                    +( 0.5*delta_h_sv*( ((psi_1_half.dx(0)+U0_nz)**2)*M11 \
                          + dot((hat_psi_star.dx(0)+hat_U0),        (2.0*MN1*(psi_1_half.dx(0)+U0_nz)        +dot(MNN,(hat_psi_star.dx(0)+hat_U0))))\
                                       +((psi_1_half.dx(1)+corr_nz.dx(1))**2)*M11 \
                          + dot((hat_psi_star.dx(1)+hat_corr.dx(1)),(2.0*MN1*(psi_1_half.dx(1)+corr_nz.dx(1))+dot(MNN,(hat_psi_star.dx(1)+hat_corr.dx(1))))) )\
                      -delta_h_sv.dx(0) * (  (psi_1_half.dx(0)+U0_nz)        *(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                            +dot((hat_psi_star.dx(0)+hat_U0), (DN1*(psi_1_half+corr_nz) + dot(DNN, (hat_psi_star+hat_corr)))) )\
                      -delta_h_sv.dx(1) * (  (psi_1_half.dx(1)+corr_nz.dx(1))*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                            +dot((hat_psi_star.dx(1)+hat_corr.dx(1)), (DN1*(psi_1_half+corr_nz) + dot(DNN, (hat_psi_star+hat_corr)))) )\
                      +(1.0/h_n0) * ( delta_h_sv.dx(0)*h_n0.dx(0) + delta_h_sv.dx(1)*h_n0.dx(1) -(delta_h_sv/(2.0*h_n0))*(h_n0.dx(0)**2 + h_n0.dx(1)**2) ) *\
                                    ( (psi_1_half+corr_nz)**2*S11 + 2.0*dot((hat_psi_star+hat_corr),SN1)*(psi_1_half+corr_nz) \
                                       + dot((hat_psi_star+hat_corr),dot(SNN,(hat_psi_star+hat_corr))) )\
-(0.5*delta_h_sv*H0*H0/(h_n0**2)) * ( (psi_1_half+corr_nz)**2*A11 + 2.0*dot((hat_psi_star+hat_corr),AN1)*(psi_1_half+corr_nz) \
                                       + dot((hat_psi_star+hat_corr),dot(ANN,(hat_psi_star+hat_corr))) )\
                      + g*H0*delta_h_sv*(h_n0 - H0) ))*dx

        # checked against notes and SE step1. 2023.6.16

        WF_hat_psi_star=   h_n0 * elem_mult(delta_hat_psi_sv.dx(0),(MN1*(psi_1_half.dx(0)+U0_nz)        +dot(MNN,(hat_psi_star.dx(0)+hat_U0))))\
                         + h_n0 * elem_mult(delta_hat_psi_sv.dx(1),(MN1*(psi_1_half.dx(1)+corr_nz.dx(1))+dot(MNN,(hat_psi_star.dx(1)+hat_corr.dx(1)))))\
                    -h_n0.dx(0) * ( elem_mult(delta_hat_psi_sv,(D1N*(psi_1_half.dx(0)+U0_nz)  +dot(DNN.T,(hat_psi_star.dx(0)+hat_U0))))\
                                   +elem_mult(delta_hat_psi_sv.dx(0),(DN1*(psi_1_half+corr_nz)+dot(DNN,(hat_psi_star+hat_corr)))) )\
                    -h_n0.dx(1) * ( elem_mult(delta_hat_psi_sv,(D1N*(psi_1_half.dx(1)+corr_nz.dx(1))+dot(DNN.T,(hat_psi_star.dx(1)+hat_corr.dx(1)))))\
                                   +elem_mult(delta_hat_psi_sv.dx(1),(DN1*(psi_1_half+corr_nz)+dot(DNN,(hat_psi_star+hat_corr)))) )\
            + (1.0/h_n0) * elem_mult(delta_hat_psi_sv,(SN1*(psi_1_half+corr_nz) + dot(SNN,(hat_psi_star+hat_corr)))) * (h_n0.dx(0)**2 + h_n0.dx(1)**2)\
          + (H0*H0/h_n0) * elem_mult(delta_hat_psi_sv,(AN1*(psi_1_half+corr_nz) + dot(ANN,(hat_psi_star+hat_corr))))

    if dim=="2D":
        WF_psi_s = ( (H0*delta_h_sv*(psi_1_half - psi_1_n0))/(0.5*dt)\
                    + ( 0.5*delta_h_sv *( ((psi_1_half.dx(0)+U0_nz)**2)*M11\
                                         + dot((hat_psi_star.dx(0)+hat_U0),(2.0*MN1*(psi_1_half.dx(0)+U0_nz)+dot(MNN,(hat_psi_star.dx(0)+hat_U0)))) )\
                     -delta_h_sv.dx(0) *( (psi_1_half.dx(0)+U0_nz)* (D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                         + dot((hat_psi_star.dx(0)+hat_U0), (DN1*(psi_1_half+corr_nz) + dot(DNN, (hat_psi_star+hat_corr)))) )\
                    +(h_n0.dx(0)/h_n0) * ( delta_h_sv.dx(0) -(delta_h_sv/h_n0)*h_n0.dx(0)*0.5 )*\
                                         ( (psi_1_half+corr_nz)**2*S11 + 2.0*dot((hat_psi_star+hat_corr),SN1)*(psi_1_half+corr_nz)\
                                           + dot((hat_psi_star+hat_corr),dot(SNN,(hat_psi_star+hat_corr))))\
     -(0.5*delta_h_sv*H0*H0/(h_n0**2)) * ( (psi_1_half+corr_nz)**2*A11 + 2.0*dot((hat_psi_star+hat_corr),AN1)*(psi_1_half+corr_nz)\
                                           + dot((hat_psi_star+hat_corr),dot(ANN,(hat_psi_star+hat_corr))))\
                    + g*H0*delta_h_sv*(h_n0-H0) ))*dx

        WF_hat_psi_star =   h_n0 * elem_mult(delta_hat_psi_sv.dx(0),(MN1*(psi_1_half.dx(0)+U0_nz)+dot(MNN,(hat_psi_star.dx(0)+hat_U0))))\
                           -h_n0.dx(0) * ( elem_mult(delta_hat_psi_sv,(D1N*(psi_1_half.dx(0)+U0_nz)+dot(DNN.T,(hat_psi_star.dx(0)+hat_U0))))\
                                         + elem_mult(delta_hat_psi_sv.dx(0),(DN1*(psi_1_half+corr_nz)+dot(DNN,(hat_psi_star+hat_corr)))) ) \
                           +(h_n0.dx(0)**2/h_n0) * elem_mult(delta_hat_psi_sv,(SN1*(psi_1_half+corr_nz) + dot(SNN,(hat_psi_star+hat_corr))))\
                                  + (H0*H0/h_n0) * elem_mult(delta_hat_psi_sv,(AN1*(psi_1_half+corr_nz) + dot(ANN,(hat_psi_star+hat_corr))))
    
    WF_psi_star = WF_psi_s + sum((WF_hat_psi_star[ind])*dx for ind in range(0,n_z))

    return WF_psi_star

#--------------------------------------------------------------------------------------------------------------------------#
#                              Step 2 : Update h^{n+1} and psi_i at time t^** simulataneously:                             #
#__________________________________________________________________________________________________________________________#

# dim, n_z, g, H0, dt: constant
# U0_nz, hat_U0, corr_nz, hat_corr: for x-periodicity
# delta_psi, delta_hat_star: test functions
# h_n0: at n time level, known
# psi_1_half, hat_psi_star: obtained from step1 [= split(w_half)], known
# >>> h_n1, hat_psi_aux = split(w_n1): unknown

def WF_h_SV(dim, n_z, H0, g, dt, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_half, hat_psi_star, hat_psi_aux, 
            M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, 
            U0_nz, hat_U0, corr_nz, hat_corr):

    if dim == "3D":
        # checked against notes and SE step1a. 2023.6.23

        WF_h =( (H0*delta_psi*(h_n1-h_n0))/(0.5*dt) \
                -( h_n0*(        (psi_1_half.dx(0)+U0_nz)*M11         + dot((hat_psi_star.dx(0)+hat_U0),MN1))*delta_psi.dx(0)\
                  +h_n0*((psi_1_half.dx(1)+corr_nz.dx(1))*M11 + dot((hat_psi_star.dx(1)+hat_corr.dx(1)),MN1))*delta_psi.dx(1)\
                  - h_n0.dx(0)*( delta_psi.dx(0)*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                +delta_psi*((psi_1_half.dx(0)+U0_nz)*D11 + dot((hat_psi_star.dx(0)+hat_U0),DN1)))\
                  - h_n0.dx(1)*( delta_psi.dx(1)*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                +delta_psi*((psi_1_half.dx(1)+corr_nz.dx(1))*D11 + dot((hat_psi_star.dx(1)+hat_corr.dx(1)),DN1)))\
                  +(delta_psi/h_n0) * (  ((psi_1_half+corr_nz)*S11 + dot((hat_psi_star+hat_corr),SN1)) * (h_n0.dx(0)**2 + h_n0.dx(1)**2)\
                                 + H0*H0*((psi_1_half+corr_nz)*A11 + dot((hat_psi_star+hat_corr),AN1)) ) )\
                -( h_n1*(        (psi_1_half.dx(0)+U0_nz)*M11         + dot((hat_psi_aux.dx(0)+hat_U0),MN1))*delta_psi.dx(0)\
                  +h_n1*((psi_1_half.dx(1)+corr_nz.dx(1))*M11 + dot((hat_psi_aux.dx(1)+hat_corr.dx(1)),MN1))*delta_psi.dx(1)\
                  - h_n1.dx(0)*( delta_psi.dx(0)*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_aux+hat_corr)))\
                                +delta_psi*((psi_1_half.dx(0)+U0_nz)*D11 + dot((hat_psi_aux.dx(0)+hat_U0),DN1)))\
                  - h_n1.dx(1)*( delta_psi.dx(1)*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_aux+hat_corr)))\
                                +delta_psi*((psi_1_half.dx(1)+corr_nz.dx(1))*D11 + dot((hat_psi_aux.dx(1)+hat_corr.dx(1)),DN1)))\
                  +(delta_psi/h_n1) * (  ((psi_1_half+corr_nz)*S11 + dot((hat_psi_aux+hat_corr),SN1)) * (h_n1.dx(0)**2 + h_n1.dx(1)**2)\
                                 + H0*H0*((psi_1_half+corr_nz)*A11 + dot((hat_psi_aux+hat_corr),AN1)) ) ) )*dx
        
        # checked against notes and SE step1b. 2023.6.23

        WF_hat_psi_aux=   h_n1 * elem_mult(delta_hat_star.dx(0),(MN1*(psi_1_half.dx(0)+U0_nz)        +dot(MNN,(hat_psi_aux.dx(0)+hat_U0))))\
                        + h_n1 * elem_mult(delta_hat_star.dx(1),(MN1*(psi_1_half.dx(1)+corr_nz.dx(1))+dot(MNN,(hat_psi_aux.dx(1)+hat_corr.dx(1)))))\
                        - h_n1.dx(0) * ( elem_mult(delta_hat_star,((psi_1_half.dx(0)+U0_nz)*D1N+dot(DNN.T,(hat_psi_aux.dx(0)+hat_U0))))\
                                        +elem_mult(delta_hat_star.dx(0),(DN1*(psi_1_half+corr_nz)+dot(DNN,(hat_psi_aux+hat_corr)))) )\
                        - h_n1.dx(1) * ( elem_mult(delta_hat_star,((psi_1_half.dx(1)+corr_nz.dx(1))*D1N+dot(DNN.T,(hat_psi_aux.dx(1)+hat_corr.dx(1)))))\
                                        +elem_mult(delta_hat_star.dx(1),(DN1*(psi_1_half+corr_nz)+dot(DNN,(hat_psi_aux+hat_corr)))) )\
           + (1.0/h_n1) * elem_mult(delta_hat_star,(SN1*(psi_1_half+corr_nz)+dot(SNN,(hat_psi_aux+hat_corr)))) * (h_n1.dx(0)**2 + h_n1.dx(1)**2)\
         + (H0*H0/h_n1) * elem_mult(delta_hat_star,(AN1*(psi_1_half+corr_nz)+dot(ANN,(hat_psi_aux+hat_corr))))

    elif dim=="2D":

        WF_h =( (H0*delta_psi*(h_n1-h_n0))/(0.5*dt)\
                -(h_n0*((psi_1_half.dx(0)+U0_nz)*M11 + dot((hat_psi_star.dx(0)+hat_U0),MN1))*delta_psi.dx(0)\
                      -h_n0.dx(0)*( delta_psi.dx(0)*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_star+hat_corr)))\
                                        + delta_psi*(D11*(psi_1_half.dx(0)+U0_nz) + dot((hat_psi_star.dx(0)+hat_U0),DN1)) )\
                      +(h_n0.dx(0)**2/h_n0)*((psi_1_half+corr_nz)*S11 + dot((hat_psi_star+hat_corr),SN1))*delta_psi\
                              +(H0*H0/h_n0)*((psi_1_half+corr_nz)*A11 + dot((hat_psi_star+hat_corr),AN1))*delta_psi)\
                -(h_n1*((psi_1_half.dx(0)+U0_nz)*M11 + dot((hat_psi_aux.dx(0)+hat_U0),MN1))*delta_psi.dx(0)\
                      -h_n1.dx(0)*( delta_psi.dx(0)*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_aux+hat_corr)))\
                                        + delta_psi*(D11*(psi_1_half.dx(0)+U0_nz) + dot((hat_psi_aux.dx(0)+hat_U0),DN1)) )\
                      +(h_n1.dx(0)**2/h_n1)*((psi_1_half+corr_nz)*S11 + dot((hat_psi_aux+hat_corr),SN1))*delta_psi\
                              +(H0*H0/h_n1)*((psi_1_half+corr_nz)*A11 + dot((hat_psi_aux+hat_corr),AN1))*delta_psi) )*dx

        WF_hat_psi_aux =       h_n1 * elem_mult(delta_hat_star.dx(0),(MN1*(psi_1_half.dx(0)+U0_nz)+dot(MNN,(hat_psi_aux.dx(0)+hat_U0))))\
                         -h_n1.dx(0)*(elem_mult(delta_hat_star,((psi_1_half.dx(0)+U0_nz)*D1N+dot(DNN.T,(hat_psi_aux.dx(0)+hat_U0))))\
                                    + elem_mult(delta_hat_star.dx(0),(DN1*(psi_1_half+corr_nz)+dot(DNN,(hat_psi_aux+hat_corr)))))\
                         +(h_n1.dx(0)**2/h_n1)*elem_mult(delta_hat_star,(SN1*(psi_1_half+corr_nz)+dot(SNN,(hat_psi_aux+hat_corr))))\
                                + (H0*H0/h_n1)*elem_mult(delta_hat_star,(AN1*(psi_1_half+corr_nz)+dot(ANN,(hat_psi_aux+hat_corr))))

    WF_h_psi = WF_h + sum((WF_hat_psi_aux[ind])*dx for ind in range(0,n_z))

    return WF_h_psi

#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 3 : Update psi_1 at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# dim, g, H0, dt: constant
# U0_nz, hat_U0, corr_nz, hat_corr: for x-periodicity
# delta_h: test function
# psi_1_half: obtained from step1, known
# h_n1, hat_psi_aux: obtained from step2, known
# >>> psi_1: trial function

def WF_psi_n1_SV(dim, H0, g, dt, delta_h, psi_1_half, psi_1, hat_psi_aux, h_n1, 
                 M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN,
                 U0_nz, hat_U0, corr_nz, hat_corr):
    
    if dim=="3D":
        # checked against notes. 2023.6.26
        # G1(corr_nz) is omitted in the first term on both sides

        a_psi_1 = ( H0*delta_h*psi_1 )*dx

        L_psi_1 = ( H0*delta_h*psi_1_half\
                 -(0.5*dt)*( 0.5*delta_h*( (psi_1_half.dx(0)+U0_nz)**2*M11 \
                             +dot((hat_psi_aux.dx(0)+hat_U0),         (2.0*MN1*(psi_1_half.dx(0)+U0_nz)         + dot(MNN,(hat_psi_aux.dx(0)+hat_U0))))\
                                          +(psi_1_half.dx(1)+corr_nz.dx(1))**2*M11 \
                             +dot((hat_psi_aux.dx(1)+hat_corr.dx(1)), (2.0*MN1*(psi_1_half.dx(1)+corr_nz.dx(1)) + dot(MNN,(hat_psi_aux.dx(1)+hat_corr.dx(1))))) )\
                            -delta_h.dx(0) * ( (psi_1_half.dx(0)+U0_nz)        *(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_aux+hat_corr)))\
                                              + dot((hat_psi_aux.dx(0)+hat_U0),         (DN1*(psi_1_half+corr_nz) + dot(DNN,(hat_psi_aux+hat_corr)))) )\
                            -delta_h.dx(1) * ( (psi_1_half.dx(1)+corr_nz.dx(1))*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_aux+hat_corr)))\
                                              + dot((hat_psi_aux.dx(1)+hat_corr.dx(1)), (DN1*(psi_1_half+corr_nz) + dot(DNN,(hat_psi_aux+hat_corr)))) )\
                            +(1.0/h_n1) * ( delta_h.dx(0)*h_n1.dx(0) + delta_h.dx(1)*h_n1.dx(1) - (delta_h/(2.0*h_n1))*(h_n1.dx(0)**2 + h_n1.dx(1)**2) )*\
                                          ( (psi_1_half+corr_nz)**2*S11 + 2.0*dot((hat_psi_aux+hat_corr),SN1)*(psi_1_half+corr_nz) \
                                                                            + dot((hat_psi_aux+hat_corr),dot(SNN,(hat_psi_aux+hat_corr))) )\
         -(0.5*delta_h*H0*H0/(h_n1**2)) * ( (psi_1_half+corr_nz)**2*A11 + 2.0*dot((hat_psi_aux+hat_corr),AN1)*(psi_1_half+corr_nz) \
                                                                            + dot((hat_psi_aux+hat_corr),dot(ANN,(hat_psi_aux+hat_corr))) )\
                             +g*H0*delta_h*(h_n1 - H0) ) )*dx

    elif dim=="2D":
        
        a_psi_1 = (H0*delta_h*psi_1)*dx
        
        L_psi_1 = (H0*delta_h*psi_1_half \
                  -(0.5*dt) * (delta_h*0.5*( (psi_1_half.dx(0)+U0_nz)**2*M11 \
                                            +dot((hat_psi_aux.dx(0)+hat_U0),(2.0*MN1*(psi_1_half.dx(0)+U0_nz)+ dot(MNN,(hat_psi_aux.dx(0)+hat_U0)))))\
                              -delta_h.dx(0)*( (psi_1_half.dx(0)+U0_nz)*(D11*(psi_1_half+corr_nz) + dot(D1N,(hat_psi_aux+hat_corr)))\
                                              +dot((hat_psi_aux.dx(0)+hat_U0), (DN1*(psi_1_half+corr_nz) + dot(DNN,(hat_psi_aux+hat_corr)))) )\
                              +(h_n1.dx(0)/h_n1)*(delta_h.dx(0)-(delta_h/h_n1)*h_n1.dx(0)*0.5) * ((psi_1_half+corr_nz)**2*S11\
                                      +2.0*dot((hat_psi_aux+hat_corr),SN1)*(psi_1_half+corr_nz)+dot((hat_psi_aux+hat_corr),dot(SNN,(hat_psi_aux+hat_corr))))\
                              -(0.5*H0*H0*delta_h/(h_n1**2)) * ((psi_1_half+corr_nz)**2*A11\
                                      +2.0*dot((hat_psi_aux+hat_corr),AN1)*(psi_1_half+corr_nz)+dot((hat_psi_aux+hat_corr),dot(ANN,(hat_psi_aux+hat_corr))))\
                              + g*H0*delta_h*(h_n1-H0) ))*dx

    return a_psi_1, L_psi_1


#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 4 : Update psi_i at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# Exactly the same as SE step 3. Rename it as WF_hat_psi_SE_SV. 2023.6.26
