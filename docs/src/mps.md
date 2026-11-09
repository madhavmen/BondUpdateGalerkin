# Matrix product states

Every state the package evolves is a `SymMPS`: a tensor train of block-sparse Telum tensors
(`TLArray`). The same container holds spin chains under a symmetry and quantics tensor trains
without one. The BUG engine reads a state only through the state interface at the end of this
page, so the container described here is the only state type it currently runs on.

## The container

```julia
mutable struct SymMPS
    tensors::Vector{Any}   # one rank-3 TLArray per site
    center::Int            # the orthogonality centre
end
```

Site tensor `psi[i]` has three legs, always in this order:

| leg | meaning | itag |
|---|---|---|
| 1 | left link | `"L,i"` |
| 2 | physical site | `"S,i"` |
| 3 | right link | `"L,i+1"` |

Neighbouring tensors share a link by its tag: leg 3 of `psi[i]` and leg 1 of `psi[i+1]` both carry
`"L,i+1"`. Interior bonds are numbered `1:length(psi)-1`; bond `b` is the link between sites `b`
and `b+1`. The two boundary links (`"L,1"` and `"L,L+1"`) are one-dimensional. The left one is the
vacuum, and the right one carries the state's total charge: under `:U1` that is `2 S^z_tot`.

Every leg also has an arrow (`'+'` or `'-'`). The only arrow rule kept is that the two ends of a
bond carry opposite arrows. Which side is `'+'` depends on the direction of the last SVD, so no code
may assume it: contractions name the legs they pair. Bra tensors are made with `'`, which conjugates
and flips the arrows, and their link legs are primed to keep them apart from the ket.

`psi[i]`, `psi[i] = tensor`, `length(psi)` and `eachindex(psi)` work as for a vector. `copy(psi)`
copies the vector of tensors and **shares the tensors themselves**. That is safe because every
operation in the package replaces a site tensor (`psi[i] = ...`) and never mutates one in place.
Code that changes a tensor in place must copy it first.

## Symmetry modes

The symmetry is a global mode, set once before any state or operator is built:

```julia
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB
set_symmetry!(:U1)
```

| mode | local space | sectors | typical use |
|---|---|---|---|
| `:U1` (default) | spin-1/2, `local_space()` | `2 S^z`: `SECTOR_UP = ((1,),)`, `SECTOR_DOWN = ((-1,),)` | XX, XXZ, Hubbard, open systems |
| `:SU2` | spin-1/2 multiplets, `local_space(:SU2)` | `2S`: `SECTOR_SU2_HALF`, `SECTOR_SU2_SINGLET` | isotropic Heisenberg |
| `:Z2` | transverse-field Ising, `ising_local_space()`, in the `sigma^x` basis | spin-flip parity | TFIM |
| `:none` | one dense block | none | dense controls, quantics tensor trains |

The state and its MPO must be built in the same mode, because both read the local space of the
current mode. Each tensor carries its own symmetry, so the kernels need no mode flag;
`symmetry(psi)` returns it as a trait (`U1Symmetry()`, `SU2Symmetry()`, `Z2Symmetry()`,
`NoSymmetry()`).

`:SU2` is not just a faster `:U1`. It stores reduced elements on multiplets, so every state it can
hold has definite total spin. A Néel or domain-wall state cannot be written under SU(2), and
`product_state` refuses to build one; start from `dimer_state` instead. Under `:SU2` bond
dimensions count **multiplets**, so the same `maxdim` keeps more states than under `:U1`. To
compare runs across modes, use the state counts:

- `bond_dims(psi)`, `leg_dim(tensor, leg)`: multiplets, the unit of `maxdim` and of every budget.
- `state_bond_dims(psi)`, `state_dim(tensor, leg)`: states. These equal `bond_dims`/`leg_dim`
  under `:U1`, `:Z2` and `:none`.
- `multiplet_ratio(psi)`: the mean number of states per multiplet (1.0 for an abelian symmetry).

## Building a state

| builder | mode | state |
|---|---|---|
| `product_state(spins)` | `:U1`, `:none` | product of `:up`/`:down` spins |
| `neel_state(L)`, `domain_wall_state(L)` | `:U1`, `:none` | `↑↓↑↓…` and `↑…↑↓…↓` |
| `sector_product_state(q, sectors)` | abelian | product state of one-dimensional sectors of any local space `q` (spin-`s` from `spin_space(twoS)`) |
| `dimer_state(L)` | `:SU2` | nearest-neighbour singlets `(1,2)(3,4)…`, `L` even |
| `ising_product_state(dirs)`, `ising_polarised_state(L)`, `ising_kink_state(L)` | `:Z2`, `:none` | products of `:plus`/`:minus` in the `sigma^x` basis |
| `hub4_product_state(occ)` | `:U1` (charge and spin), `:none` | `d = 4` Hubbard sites |
| `qtt_mps(cores; center)` | `:none` | a quantics tensor train from dense `(chi_l, 2, chi_r)` cores |

Product states are built by a fusion recursion: each site fuses the incoming link with the physical
leg and keeps only the requested sector. The physical leg keeps its full space, so local operators
still contract, and every bond has dimension 1. The spin builders take `eltype`: `Float64` gives a
real state, and the default is `ComplexF64`, since real-time evolution makes amplitudes complex.

Product states are returned with the centre at the last site. `qtt_mps` gauges to `center` by exact
SVD moves.

## Canonical form

Every tensor left of `psi.center` is a left isometry and every tensor right of it a right
isometry. The gauge is changed only by exact, truncation-free SVD moves:

- `move_right!(psi, site)`, `move_left!(psi, site)`: move the centre one site; they do not update
  `psi.center`.
- `canonical!(psi, site)`: sweep the centre to `site` and update `psi.center`. Norm-preserving.
- `left_isometry_defect(tensor)`, `right_isometry_defect(tensor)`: `||A (I - A'A)||` and its
  mirror, zero exactly for an isometry and computed without cancellation, so a defect of `1e-14`
  can be told apart from `1e-8`. `left_gram`/`right_gram` give `A'A` and `AA'` with the bond left
  open.

`norm(psi)` is the Frobenius norm of the centre tensor, which equals the state norm **only in
canonical form**. For a state of unknown gauge, use `sqrt(real(overlap(psi, psi)))`.

## Inner products and sizes

- `overlap(bra, ket)` (also `dot`): `<bra|ket>`, contracted site by site from the left. The bond
  dimensions may differ. Two states with no common total charge have overlap exactly `0`; this is
  not an error.
- `tensor_inner(a, b)`: the inner product of two tensors with the same legs, also exactly zero
  when no charge blocks match.
- `bond_dims(psi)`: the interior bond dimensions. `tensor_elements(tensor)` is the number of stored
  elements, which is the memory a block-sparse tensor actually takes; use it rather than a product
  of dimensions.

## Charge sectors

A site tensor is block-sparse: only charge-conserving blocks are stored. Rank adaptation therefore
works sector by sector:

- `reachable_sectors(tensor, leg_a, leg_b)`: every charge the fused legs can carry, with its
  dimension. The dimensions sum to the product of the two leg dimensions.
- `fusion_basis(tensor, leg_a, leg_b)`: an orthonormal basis of the fused space, labelled the way
  `svd` labels a bond, so it can be combined with a frame directly.
- `perp_component(frame, directions)`, `perp_component_right`: the part of new directions outside
  a frame, used when a basis is enlarged.

The expansion needs these because a frame grown by orthogonalising `[U0 | K1]` cannot open a
charge sector the frame does not already hold. The sectors are enumerated explicitly instead of
padding every sector to full size. `fusion_basis` is memoised: the memo is cleared by
`clear_fusion_cache!`, bounded by `set_fusion_budget!` (default `2^28` elements), and counted by
`fusion_stats`.

## Observables

The observables are in `QMB`:

- `site_expval(psi, j, O)`, `sz_expectation(psi, j)`, `magnetisation(psi)`, `total_sz(psi)`.
- `bond_spectrum(psi, bond)` and `entanglement_spectrum(psi)`: Schmidt values, expanded by
  multiplet degeneracy.
- `mpo_energy(psi, mpo)` and `mpo_matrix_element(bra, mpo, ket)` (in `BUG`): unnormalised, so
  divide by `overlap(psi, psi)` for a state that is not normalised.

Local observables call `canonical!`, which changes the gauge but not the state. Pass `copy(psi)` if
the centre must stay where it is.

## The state interface

The BUG engine does not read `SymMPS` fields. It uses the interface in
`src/BUG/Core/Interfaces/state.jl`, which every state type implements:

| function | meaning |
|---|---|
| `nsites(psi)` | number of sites |
| `physical_dimension(psi, site)` | dimension of a site leg, summed over sectors |
| `bond_dimension(psi, bond)`, `bond_dimensions(psi)` | link dimensions |
| `orthogonality_center(psi)` | the centre site |
| `canonicalize!(psi, site)` | move the centre (exact) |
| `inner_product(bra, ket)` | `<bra|ket>` |
| `symmetry(psi)` | the symmetry trait |

`test/BUG/Core/Interfaces/test_state_conformance.jl` runs one suite against each state type.
