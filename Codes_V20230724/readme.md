## Version Description

- The only change compared with the last version is the removal of all the terms related to the bottom $b(x,y)$ and its derivatives $b_x, b_y$ in the weak formulations.

## Results and Discussions

- `Test_SP2`
  - There were always some problems when running the code on ARC4.
  - No problem when using the laptop locally.
  - Fewer ripples (two appear at the lower part, compared with four). The calculation is sensitive to any slight disturbance.
- ➡️ Evaluate $U_0$ and $c_0$ at each time step
