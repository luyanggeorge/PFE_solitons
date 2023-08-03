## Codes for SP2 simulations.
- Mathematically it's equivalent to the VP-approach by Onno and Junho
  - The $\mu^2$ term is ignored when setting the IC for $\phi$ and $\tilde{\phi}$;
  - $U_0$ and $c_0$ are only evaluated once at the beginning of the simulation.
- All the terms related to the seabed and its derivatives are removed in the weak formulations.
- `par.sh` is the script for submitting the job on ARC4.
