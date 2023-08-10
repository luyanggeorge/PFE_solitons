## Codes for SP2 simulations.
- Mathematically it's equivalent to the VP-approach by Onno and Junho, instead the explicit weak formulations are used here:
  - The high-order $\mu^2$ term is ignored when setting the IC for $\phi$ and $\tilde{\phi}$;
  - $U_0$ and $c_0$ are only evaluated once at the beginning of the simulation.
- Solver parameters designed by Colin for MMP can speed up the code significantly using SV (ca 3.5 times faster)
    ```
    param_cc = {'ksp_type': 'gmres',
                'ksp_converged_reason': None,
                'pc_type': 'python',
                'pc_python_type': 'firedrake.ASMStarPC',
                'snes_lag_preconditioner_persists': None,
                'snes_lag_preconditioner': 5,
                'star_construct_dim': 2,
                'star_sub_sub_pc_type': 'lu',
                'star_sub_sub_pc_factor_mat_ordering_type':'rcm'}
    ```
  - Among the above parameters, either setting `'star_construct_dim': 2` or `'star_construct_dim': 1` works well for the first two nonlinear solvers for SV.
  - An explanation from Lawrence:
    > It affects which dimension mesh entity is looped over to construct the star patches. 0 -> vertices; 1 -> edges; 2 -> cells (in 2D). As you increase this number the patches get smaller (the star of a vertex is all the cells touching it, the star of an edge is the two cells touching it, the star of a cell is just the cell itself) which is cheaper to apply, but it might result in not having parameter robust convergence. Depends on the pde.
- All the variables related to the seabed and its derivatives are removed from all the codes, and so do the associated `FWF`, `H`, matrices B and C, which means this version only considers flat bottom.
- `par.sh` is the script for submitting the job on ARC4.
