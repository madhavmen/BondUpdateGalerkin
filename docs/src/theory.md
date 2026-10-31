# Method

## The BUG step

For `d_t psi = A psi` on a tensor train with orthogonality centre `c`, one step of the
rank-adaptive basis-update-and-Galerkin integrator (Ceruti, Kusch and Lubich; the tree form of
arXiv:2201.10291):

1. **Basis update.** The `K` half-sweep (sites `1..c-1`) and the `L` half-sweep (sites
   `c+1..L`) evolve each site's frame with its effective operator and keep the old frame plus the
   new directions: `W_k = orth([W_k^old, K_k(tau)])`, likewise `Z_k`. The new directions are
   chosen by controlled bond expansion (below).
2. **Galerkin root.** The centre core `S` is projected into the enlarged frames and evolved with
   the projected operator, `S(tau) = exp(tau W' A Z) S(0)` (or an implicit Runge-Kutta map for
   nonlinear flows).
3. **Truncation.** One root-to-leaves pass cuts every bond to the smallest rank whose discarded
   norm fits its share of `tol_trunc` (or `maxdim`).

The order-`p` step (`order = p`) closes the frames under the frontier Krylov vectors
`(tau A)^j psi`, `j < p`, so that the Galerkin root is accurate to `O(tau^(p+1))`.

## Controlled bond expansion

The directions a bond needs are the leading left singular vectors of the projected two-site
action `P_perp A Theta`. The default expansion (`DDQRExpansion`) sketches this action with a
Gaussian test matrix, `P_perp A Theta Omega`, using one fixed random stream per step, and accepts
the sketch only when a residual test bounds what it misses by the bond's share of `tol_basis`.
`exact = true` forms the full projected action instead (the reference arm).

## Quantics tensor trains

A field on `2^n_x x 2^n_y` points is a tensor train with one site per bit of the grid indices
(`QTTOrdering`: grouped or interleaved, least or most significant bit first). Shifts are carry
automata of bond dimension 2, so finite-difference operators are MPOs of small bond dimension.
Boundary conditions are explicit (`:periodic`, `:dirichlet0`, `:onesided`), never the open-chain
default of an MPS. The physical inner product carries the cell weight `dx dy` outside the cores.

Nonlinear flows (`GPFlow`) rebuild their operator `H[psi, t]` from the state at every stage. The
root of a nonlinear step solves the implicit Runge-Kutta stage equations in the frames by
Newton-GMRES on doubled-real unknowns, since `F` is only real-linear (`rk_stage_solve`).
