from firedrake import *
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLegendreQuadratureLineRule
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule
import pdb

n_z = 8
H0 = 20 
order=n_z+1 # GLL number of intergration points (include the end points)
fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), order)
zeros = fiat_rule.get_points()
z_k = [zeros.item(n_z-k) for k in range(n_z)] #k=0,1,...,n_z-1
z_k.append(0) # add the last one z_{n_z}=0 manually
z_k_H0=[H0*z_k[i] for i in range(len(z_k))]
print(z_k_H0) # obtain the list consisting of z_k's
