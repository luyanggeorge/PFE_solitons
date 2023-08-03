# Finite-element simulations on soliton interactions in the framework of potential-flow equations
## :information_source: General information
- This repository contains codes for FE simulations on soliton interactions based on PFE in an x-periodic computational domain, which consist of four `.py` files. They are put in the folders with name `Codes` plus the version number.
  - `3D_tank_periodic.py`: main code
  - `settings.py`: to specify the parameters of one simulation
  - `savings.py`: anything related to the output of numerical results
  - `solvers.py`: to generate weak formulations for SE and SV time-stepping schemes
- The pre-processing and post-processing codes for each case are also provided.
- After the end of one simulation, the results are output into three files:
  - `readme.txt`: details of the test case
  - `energy.csv`: the record of the total energy evolution
  - `soliton.pvd`: field data of $h(x,y,t)$ and $\tilde{\phi}(x,y,z=h,t)$ that can be visualised via *ParaView*

## :one: Single-soliton simulations (SP1, 2D)
- Pre-Processing code: `prep-SP1-KPE_solution.py`
- Post-Processing codes: `postp-SP1-Energy.py`, `postp-SP1-Validation.py`, `postp-SP1-Validation_gif.py`

## :two: Two-soliton interactions (SP2, 3D)
- Pre-Processing code: `prep-SP2-plot_IC.py`
- Post-Processing codes: `postp-SP2-Energy.py`, `postp-SP2-cross-section.py`

## :hash: Other
- `Gauss-Labatto_FIAT.py`: print the GLL nodes along the z-direction with the help of `FIAT.quadrature` module (can be compared with that in the main code, which uses `sympy.integrals.quadrature` instead.)
