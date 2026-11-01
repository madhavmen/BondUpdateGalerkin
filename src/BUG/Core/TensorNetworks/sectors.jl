# Reachable-sector enumeration and the fusion-basis memo. Under U(1) with the opposite frame
# frozen, `orth([U0 | K1])` cannot open a new charge sector, so empty reachable sectors are
# detected explicitly instead of padding every sector to full dimension.
#
# Moved from BUG-Julia@b155550:src/BondUpdateBUG/sectors.jl. `SectorReport`, `sector_report`,
# `sector_report_right` and `missing_charges` were not moved: they called `_complement_basis`,
# removed in BUG-Julia 3024066, so they threw at b155550 (docs/dev/migration-map.md).

# ── charge arithmetic ───────────────────────────────────────────────────────

"Per-symmetry dual (charge negation) of a full sector label."
dual_charge(syms, charge) =
    ntuple(symmetry_index -> get_dualq(syms[symmetry_index], charge[symmetry_index]), length(syms))

"""
    add_charge(syms, qa, qb)

Per-symmetry fusion of two sector labels. **Abelian only** -- for a
non-Abelian symmetry the fusion of two irreps is a *set*, not a single label,
so use [`reachable_sectors`](@ref), which delegates to Telum's own
`getIdentity` and handles both cases.
"""
function add_charge(syms, qa, qb)
    return ntuple(length(syms)) do symmetry_index
        group = syms[symmetry_index]
        isabelian(group) ||
            throw(ArgumentError("add_charge is abelian-only; use reachable_sectors for $(group)"))
        # An abelian label is a one-element tuple per symmetry: fuse the two and rewrap.
        (add_qn(group, qa[symmetry_index][1], qb[symmetry_index][1]),)
    end
end

"""
    fuse_spaces(syms, space_a, dir_a, space_b, dir_b) -> Vector{Tuple{QLabel,Int}}

Charges reachable on the fused leg `a (x) b`, with dimensions, by explicit charge arithmetic in
the frame convention (incoming legs dualised, sum, dualise once more). Abelian only. A vacuum
link cannot tell the sign convention apart, so test on charged links.
"""
function fuse_spaces(syms, space_a, dir_a::Char, space_b, dir_b::Char)
    acc = Dict{Any,Int}()
    order = Any[]
    for (qa, da) in space_a, (qb, db) in space_b
        ca = dir_a == '+' ? dual_charge(syms, qa) : qa
        cb = dir_b == '+' ? dual_charge(syms, qb) : qb
        fused_charge = dual_charge(syms, add_charge(syms, ca, cb))
        haskey(acc, fused_charge) || push!(order, fused_charge)
        acc[fused_charge] = get(acc, fused_charge, 0) + da * db
    end
    return [(fused_charge, acc[fused_charge]) for fused_charge in order]
end

"""
Memo for [`fusion_basis`](@ref) and its hit/miss counters. The result depends only on the legs'
spaces, arrows, tags and prime levels, so it is memoised; the key carries itags and arrows, a copy
is returned, and the lock (held only around the Dict) makes it safe under the parallel half-sweeps.
"""
const _FUSION_CACHE = Dict{Any,Any}()
const _FUSION_LOCK = ReentrantLock()
const _FUSION_CALLS = Ref(0)
const _FUSION_MISSES = Ref(0)
const _FUSION_ELEMS = Ref(0)

"""
Element budget of the memo. The key holds the legs' exact spaces, which an adaptive run changes
nearly every step, so an unbounded memo grows with the run (dm7a 17541077, B1 L = 20 BUG, 107
steps: memo 8 GB, live heap 8.7 GB, RSS 12.3 GB unbounded; RSS 5.0 GB bounded, results bitwise
equal). An insert that would pass the budget empties the memo first; within a step the same legs
recur, so the reuse the memo exists for survives. `nothing` = unbounded (before 2026-10-27).
2^28 elements (2-4 GB) holds one step's legs at D ~ 1024, d = 4 (~200 keys of up to ~6e5
elements), so a step does not thrash it.
"""
const _FUSION_BUDGET = Ref{Union{Nothing,Int}}(2^28)

"Set the memo's element budget (`nothing` = unbounded); returns the previous one."
function set_fusion_budget!(budget::Union{Nothing,Integer})
    budget === nothing ||
        budget > 0 ||
        throw(ArgumentError("fusion budget must be positive, got $budget"))
    old = _FUSION_BUDGET[]
    _FUSION_BUDGET[] = budget === nothing ? nothing : Int(budget)
    return old
end

"Calls and misses since the last `reset_fusion_stats!`; distinct keys and stored elements now."
fusion_stats() = (
    calls = _FUSION_CALLS[],
    misses = _FUSION_MISSES[],
    keys = length(_FUSION_CACHE),
    elems = _FUSION_ELEMS[],
)

"Zero the counters (NOT the cache) so a step can be attributed on its own."
function reset_fusion_stats!()
    _FUSION_CALLS[] = 0
    _FUSION_MISSES[] = 0
    return nothing
end

"Drop the memo entirely -- for tests that switch symmetry mode under one process."
function clear_fusion_cache!()
    empty!(_FUSION_CACHE)
    _FUSION_ELEMS[] = 0
    reset_fusion_stats!()
    return nothing
end

# The memo key of one leg of `tensor`: everything `fusion_basis` reads off it (spaces, arrow, tags,
# prime level, lock), as a hashable tuple.
_leg_key(tensor, leg::Int) = (
    tensor.spaces[leg],
    tensor.inds[leg].dir,
    string(tensor.inds[leg].itags),
    tensor.inds[leg].plev,
    tensor.inds[leg].lock,
)

"""
    fusion_basis(tensor, leg_a=1, leg_b=2; tag="fused") -> TLArray

Orthonormal basis of the fused `leg_a (x) leg_b` space in `tensor`'s own leg convention: legs
`(leg_a, leg_b, bond)` with `tensor`'s arrows and the bond labelled as
`svd(tensor, (leg_a, leg_b)).U` labels it. `getIdentity` alone has flipped arrows and dual labels.
Memoised (`_FUSION_CACHE`).
"""
function fusion_basis(tensor, leg_a::Int = 1, leg_b::Int = 2; tag::AbstractString = "fused")
    key = (_leg_key(tensor, leg_a), _leg_key(tensor, leg_b), String(tag), symmetry_mode())
    hit = lock(_FUSION_LOCK) do
        _FUSION_CALLS[] += 1
        get(_FUSION_CACHE, key, nothing)
    end
    hit === nothing || return to_concrete(hit)
    lock(_FUSION_LOCK) do
        _FUSION_MISSES[] += 1
    end
    identity_map = to_concrete(getIdentity((tensor, leg_a), (tensor, leg_b); itag = tag)')
    # getIdentity's output is always (input_a, input_b, fused), whatever
    # leg numbers the inputs had, so the split is on (1,2) -- NOT (leg_a, leg_b).
    left_vectors = to_concrete(svd(identity_map, (1, 2); cutoff = 0.0).U)
    # The SVD relabels the bond to Telum's default, so `tag` is reapplied; the CBE probes need their
    # distinct tags to avoid colliding with a frame's `svdL` bond leg.
    out = to_concrete(setitag(left_vectors, 3, tag))
    # Store, then hand out a copy -- see the memo's note on caller ownership.
    element_count = sum(length(reduced_block) for reduced_block in out.RMTs; init = 0)
    lock(_FUSION_LOCK) do
        cap = _FUSION_BUDGET[]
        if cap !== nothing && _FUSION_ELEMS[] + element_count > cap
            empty!(_FUSION_CACHE)
            _FUSION_ELEMS[] = 0
        end
        # A concurrent miss may have stored this key already; count its elements once.
        haskey(_FUSION_CACHE, key) || (_FUSION_ELEMS[] += element_count)
        _FUSION_CACHE[key] = out
    end
    return to_concrete(out)
end

"""
    reachable_sectors(tensor, leg_a=1, leg_b=2) -> Vector{Tuple{QLabel,Int}}

Every charge reachable on the fused `leg_a (x) leg_b` of `tensor`, with its
dimension, labelled in `tensor`'s own bond convention (see [`fusion_basis`](@ref)).

The total dimension always equals `leg_dim(tensor, leg_a) * leg_dim(tensor, leg_b)` --
fusing regroups the product space by charge, it never loses or adds directions.
"""
function reachable_sectors(tensor, leg_a::Int = 1, leg_b::Int = 2)
    return [
        (charge, block_dim) for
        (charge, block_dim) in fusion_basis(tensor, leg_a, leg_b).spaces[end]
    ]
end

"Dimension of sector `charge` on leg `leg` of `tensor`, or 0 if the sector is absent."
function sector_dim(tensor, leg::Int, charge)
    for (sector, block_dim) in tensor.spaces[leg]
        sector == charge && return block_dim
    end
    return 0
end

"""
    align_charge(syms, charge, from_dir, to_dir)

Re-express sector label `charge`, read off a leg with arrow `from_dir`, in the
convention of a leg with arrow `to_dir`.

A leg's charge LABEL depends on its arrow: the same physical sector is `q` on
an outgoing leg and `dual(q)` on an incoming one. `U0`'s bond leg comes from
`svd`'s `U` and is `'-'`, while `K1`'s comes from `S0`'s right leg and is `'+'`
(measured, job 93412), so comparing their `spaces` without this conversion
silently mismatches every non-self-dual sector.
"""
align_charge(syms, charge, from_dir::Char, to_dir::Char) =
    from_dir == to_dir ? charge : dual_charge(syms, charge)

"""
    perp_component(frame_basis, new_directions) -> TLArray

`P_perp K = K - U (U' K)` for `U = frame_basis` (a left isometry, e.g. a frame `U0`) and
`K = new_directions` (e.g. a Krylov block `K1`), contracting over the `(link, site)` legs, so the
two bond legs (different tags and arrows) never need to match.
"""
function perp_component(frame_basis, new_directions)
    ov = contract(frame_basis', (1, 2), new_directions, (1, 2))  # (basis bond, directions bond)
    proj = contract(frame_basis, (3,), ov, (1,))                 # (link, site, directions bond)
    # The copying form, not `to_concrete!`: a Telum sum reuses `new_directions`'s block for any
    # sector `proj` lacks, and the in-place cleanup would rescale it. The result is then a fresh
    # tensor the caller owns -- wrapping it in another `to_concrete` only copies it again.
    return to_concrete(new_directions - proj)
end

# ── right-frame mirrors ─────────────────────────────────────────────────────
# V0 and L1 carry (bond, site, link): the frame leg is 1 and the fused pair is
# (2, 3). Written out rather than obtained by transposing, matching how Alice
# keeps _symmetric_augmented_right_isometry_from_l explicit -- the arrow and tag
# bookkeeping does not survive a naive transpose.

"""
    perp_component_right(frame_basis, new_directions) -> TLArray

`P_perp L = L - (L V') V` for `V = frame_basis` (a right isometry, e.g. a frame `V0`) and
`L = new_directions` (e.g. a Krylov block `L1`), contracting over the `(site, link)` legs.
Row mirror of [`perp_component`](@ref).
"""
function perp_component_right(frame_basis, new_directions)
    ov = contract(new_directions, (2, 3), frame_basis', (2, 3))  # (directions bond, basis bond)
    proj = contract(ov, (2,), frame_basis, (1,))                 # (directions bond, site, link)
    return to_concrete(new_directions - proj)
end

# ── from BUG-Julia@b155550:src/RSVDCBEBondUpdate/budget/multiplets.jl ──

# ── MULTIPLETS VERSUS STATES ──
# Under SU(2) `leg_dim` and `Nkeep` count multiplets, so `maxdim` cuts differently than under `:U1`.
# Weights are physical (`sum sigma^2 * degeneracy == norm^2`); convert via `state_dim` to compare.

"""
    state_dim(tensor, leg) -> Int

Physical dimension of leg `leg` of `tensor`: each sector's multiplet count weighted by its irrep
dimension. Equals `leg_dim(tensor, leg)` for an abelian symmetry (every irrep is one-dimensional)
and exceeds it under SU(2).

`leg_dim` is the number the code's budgets, `Nkeep` and reported bond dimensions are in;
`state_dim` is the number a U(1) or unsymmetric run would report for the same physical state.
Quote whichever the comparison needs -- but quote which one.
"""
function state_dim(tensor, leg::Int)
    symmetries = symm(tensor)
    symmetry_count = length(symmetries)
    tot = 0
    for (charge, block_dim) in tensor.spaces[leg]
        # Irrep dimension of the sector: the product over the symmetry factors.
        irrep_weight = 1
        for symmetry_index = 1:symmetry_count
            irrep_weight *= dimension(symmetries[symmetry_index], charge[symmetry_index])
        end
        tot += block_dim * irrep_weight
    end
    return tot
end

"""
    state_bond_dims(psi) -> Vector{Int}

The interior bond dimensions in STATES rather than multiplets -- the SU(2) analogue of
`bond_dims(psi)`, and the quantity comparable with a U(1) run of the same physics.

Under `:none` and `:U1` this is `bond_dims(psi)` exactly, so it is safe to call unconditionally
and is the right thing to log when a script may run in more than one mode.
"""
state_bond_dims(psi::SymMPS) = [state_dim(psi[bond], 3) for bond = 1:(length(psi)-1)]

"""
    local_dim(psi) -> Int

The largest physical dimension over the sites, in states (`state_dim` of the site leg): 2 for
spin-1/2, 4 for the d = 4 Hubbard site, under any symmetry. A frame of `r` multiplets fused with a
site of `d` states holds at most `d*r` multiplets, so this is the factor of `work_ceiling`.
"""
local_dim(psi::SymMPS) = maximum(site -> state_dim(psi[site], 2), 1:length(psi); init = 1)

"""
    multiplet_ratio(psi) -> Float64

Mean states per multiplet over the interior bonds: `1.0` under an abelian symmetry, and how much
smaller an SU(2) tensor is than the U(1) tensor carrying the same physics.

The honest way to state the SU(2) saving, and the factor by which a `maxdim` in multiplets is
more generous than the same number in states.
"""
function multiplet_ratio(psi::SymMPS)
    multiplets = bond_dims(psi)
    states = state_bond_dims(psi)
    isempty(multiplets) && return 1.0
    tot = sum(multiplets)
    return tot == 0 ? 1.0 : sum(states) / tot
end
