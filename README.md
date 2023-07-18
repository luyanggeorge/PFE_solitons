# Finite-element simulations on soliton interactions in the framework of potential-flow equations
## :information_source: General information
- This repository contains codes for FE simulations on soliton interactions based on PFE in an x-periodic computational domain, which consist of four `.py` files. They are put in the folders with name `Codes` plus the version number.
  - `3D_tank_periodic.py`: main code
  - `settings.py`: to specify the parameters of one simulation
  - `savings.py`: anything related to the output of numerical results
  - `solvers.py`: to generate weak formulations for SE and SV time-stepping schemes
- The pre-processing and post-processing codes for each case are also provided.

## :one: Single-soliton simulations (SP1, 2D)
- Pre-Processing code: `prep-SP1-KPE_solution.py`
- Post-Processing codes: `postp-SP1-Energy.py`, `postp-SP1-Validation.py`, `postp-SP1-Validation_gif.py`

## :two: Two-soliton interactions (SP2, 3D)
- Pre-Processing code: `prep-SP2-plot_IC.py`

## :hash: Other
- `Gauss-Labatto_FIAT.py`: print the GLL nodes along the z-direction with the help of `FIAT.quadrature` module (can be compared with that in the main code, which uses `sympy.integrals.quadrature` instead.)
