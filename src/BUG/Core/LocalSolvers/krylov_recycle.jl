# ── RECYCLED LOCAL KRYLOV ACROSS AUGMENTATION ROUNDS (`krylov_recycle`, an ABLATION) ──────────
# An order-p step runs p augmentation rounds: the half-sweeps
# (round 1, frontier j = 0) and one enriching frontier sweep per j = 1..p-1. Each round's local
# Krylov basis at a bond is cached; the next round transports it into its own gauge -- the frame
# leg by the map `Pm` the frontier sweep already carries the frames with, the opposite leg by the
# overlap of the old and the new tail state -- re-orthonormalises it against its seed, REFRESHES
# every image with its own `H1` (a stale `H1 v` is never reused) and continues the Arnoldi
# recurrence from that block under the same time-weighted stop. Off by default.
# `mode = :frame` is the image-free variant: the transported vectors only join the starting frame
# (each by its own weighted residual, as `_extend_frame` admits Krylov directions) and a FRESH
# `_krylov_enrich` runs on top -- no image refresh, no continued recurrence.
# `mode = :complement`: the entry keeps the solve's input
# first, and the next round reuses the transported block only where it certifies that input's
# evolution under its own `H1`, extending only the complement of the input (`_rk_complement`).

"`:complement` vectors kept per bond besides the input (a bounded set, so the work stays capped)."
const _RECYCLE_KEEP = 8

"""
    KrylovRecycle(mode = :images)

One step's recycle cache, per side: the last committed round's entries `bond => (basis, weights,
frame)`, the state whose tail the entries' opposite leg lives in, and the frame generation their
frame leg is written in. A sweep writes `pend` and its entries become usable only when its frames
are committed ([`_recycle_commit!`](@ref)). `log` has one row per recycled bond visit.
"""
mutable struct KrylovRecycle
    cur::Dict{Symbol,Any}      # side => (entries, tail, gen)
    pend::Dict{Symbol,Any}
    log::Dict{Symbol,Vector{Any}}
    mode::Symbol                # :images (refreshed block recurrence), :frame (frame seeding) or
    # :complement (certified block, complement extended)
end
function KrylovRecycle(mode::Symbol = :images)
    mode in (:images, :frame, :complement) ||
        throw(ArgumentError("recycle mode must be :images, :frame or :complement"))
    return KrylovRecycle(
        Dict{Symbol,Any}(:left => nothing, :right => nothing),
        Dict{Symbol,Any}(:left => nothing, :right => nothing),
        Dict{Symbol,Vector{Any}}(:left => Any[], :right => Any[]),
        mode,
    )
end

# Open a pending (uncommitted) round on `side`, recording the tail state and frame generation.
# The two half-sweeps (and the two halves of a frontier sweep) may run concurrently: each touches
# only its own side's slots, which `Dict` assignment to an EXISTING key does not resize.
_recycle_begin!(::Nothing, ::Symbol, tail, gen) = nothing
_recycle_begin!(rec::KrylovRecycle, side::Symbol, tail, gen::Int) =
    (rec.pend[side] = (entries = Dict{Int,Any}(), tail = tail, gen = gen); nothing)

"""
Store `bond`'s round: the vectors that carried weight above `keep` (they mattered to the
response) and the last one, the active frontier the recurrence would continue from. Under
`:complement` the solve's input comes first -- the next round extends only the complement of it --
then the frontier and the heaviest others, at most `_RECYCLE_KEEP` besides the input.
"""
_recycle_store!(::Nothing, args...) = nothing
function _recycle_store!(rec::KrylovRecycle, side::Symbol, bond::Int, st, frame, keep::Float64)
    pending = rec.pend[side]
    (pending === nothing || isempty(st.basis)) && return nothing
    basis_count = length(st.basis)
    sel = if rec.mode === :complement
        # the interior vectors above `keep`, heaviest first, behind the frontier (last vector)
        kept = sort!(
            [column for column = 2:(basis_count-1) if st.weights[column] > keep];
            by = column -> -st.weights[column],
        )
        kept = vcat(basis_count >= 2 ? [basis_count] : Int[], kept)
        vcat(1, kept[1:min(end, _RECYCLE_KEEP)])
    else
        [column for column = 1:basis_count if column == basis_count || st.weights[column] > keep]
    end
    pending.entries[bond] = (basis = st.basis[sel], weights = st.weights[sel], frame = frame)
    return nothing
end

# Promote `side`'s pending round to the committed one when `keep` (its frames were committed),
# otherwise discard it; returns `nothing`.
_recycle_commit!(::Nothing, ::Symbol, ::Bool) = nothing
function _recycle_commit!(rec::KrylovRecycle, side::Symbol, keep::Bool)
    keep && (rec.cur[side] = rec.pend[side])
    rec.pend[side] = nothing
    return nothing
end

"The committed entries of `side` usable in frames generation `gen`, or `nothing`."
_recycle_entries(::Nothing, ::Symbol, ::Int) = nothing
function _recycle_entries(rec::KrylovRecycle, side::Symbol, gen::Int)
    committed = rec.cur[side]
    return committed === nothing || committed.gen != gen ? nothing : committed
end

"""
`site_tensor` (at `site` of `site_count`) with its internal link legs moved into the private
`RB,site` namespace (as `overlap` does).
"""
function _rb(site_tensor, site::Int, site_count::Int)
    site > 1 && (site_tensor = setitag(site_tensor, 1, "RB,$site"))
    site < site_count && (site_tensor = setitag(site_tensor, 3, "RB,$(site + 1)"))
    return TLArray(site_tensor)
end

"""
    _tail_overlaps(Anew, Aold, side) -> overlaps

The overlap environments of two states with the same site legs, `Anew` the bra: `side = :left`
gives the right stack `overlaps[site]` = sites `site..site_count` with legs `(bra link_site,
ket link_site)` (both states right-canonical there), `:right` the left stack `overlaps[site]` =
sites `1..site-1`. Bra links are `RB,site`.
"""
function _tail_overlaps(Anew::SymMPS, Aold::SymMPS, side::Symbol)
    site_count = length(Anew)
    overlaps = Vector{Any}(nothing, site_count + 1)
    if side === :left
        for site = site_count:-1:3
            # ket site tensor with the environment to its right absorbed: (link, phys, bra link)
            ket_with_env =
                overlaps[site+1] === nothing ? Aold[site] :
                contract(Aold[site], (3,), overlaps[site+1], (2,))
            # close the physical and right legs against the conjugated bra site
            overlaps[site] =
                contract(_rb(Anew[site], site, site_count)', (2, 3), ket_with_env, (2, 3))
        end
    else
        for site = 1:(site_count-2)
            # the environment to the left absorbed into the ket site tensor: (bra link, phys, link)
            ket_with_env =
                overlaps[site] === nothing ? Aold[site] :
                contract(overlaps[site], (2,), Aold[site], (1,))
            # close the left and physical legs against the conjugated bra site
            overlaps[site+1] =
                contract(_rb(Anew[site], site, site_count)', (1, 2), ket_with_env, (1, 2))
        end
    end
    return overlaps
end

"""
    _recycle_transport(entry, side, Pm, Fnew, overlaps, bond) -> Vector

The entry `entry` of `bond` in the current round's local space: the frame leg through `Pm` (as
the frames themselves), the opposite leg through `<new frame + tail | old frame + tail>`. A
projection, not an isometry: what the new opposite space lacks is dropped (the vectors are
re-orthonormalised by the solver).
"""
function _recycle_transport(entry, side::Symbol, Pm, Fnew, overlaps, bond::Int)
    Fold = entry.frame
    if side === :left                       # opposite frame: (bond, site_{bond+1}, link_{bond+1})
        Ek = overlaps[bond+2]
        Fn = setitag(Fnew, 1, "RCb")
        Ek === nothing || (Fn = setitag(Fn, 3, "RB,$(bond + 2)"))
        old_with_tail = Ek === nothing ? Fold : contract(Fold, (3,), Ek, (2,))
        # (new*, old): the old opposite leg projected on the new one
        overlap_map = contract(TLArray(Fn)', (2, 3), old_with_tail, (2, 3))
        tag = Telum.inds(Fnew)[1].itags
        return map(entry.basis) do old_vector
            moved = Pm === nothing ? old_vector : contract(Pm, (2,), old_vector, (1,))
            TLArray(setitag(contract(moved, (3,), overlap_map, (2,)), 3, tag))
        end
    end
    Ek = overlaps[bond]                     # opposite frame: (link_{bond-1}, site_bond, bond)
    Fn = setitag(Fnew, 3, "RCb")
    Ek === nothing || (Fn = setitag(Fn, 1, "RB,$bond"))
    old_with_tail = Ek === nothing ? Fold : contract(Ek, (2,), Fold, (1,))
    # (new*, old): the old opposite leg projected on the new one
    overlap_map = contract(TLArray(Fn)', (1, 2), old_with_tail, (1, 2))
    tag = Telum.inds(Fnew)[3].itags
    return map(entry.basis) do old_vector
        moved = contract(overlap_map, (2,), old_vector, (1,))
        Pm === nothing || (moved = contract(moved, (3,), Pm, (1,)))
        TLArray(setitag(moved, 1, tag))
    end
end

"""
`vector` minus its projection on the orthonormal `basis`, CGS2 (`_krylov_enrich`'s
orthogonalisation).
"""
function _orth_against(basis, vector)
    for _pass = 1:2
        corr = nothing
        for basis_vector in basis
            ov = _kinner(basis_vector, vector)
            corr = corr === nothing ? ov * basis_vector : corr + ov * basis_vector
        end
        corr === nothing || (vector = _kconc(vector - corr))
    end
    return vector
end

"`phi_1(generator) e_1` from one augmented exponential."
_phi1_e1(generator::AbstractMatrix) = last(_exp_phi1_e1(generator))

"""
    _block_weight(projection, residual_gram, tau; state_seed = true) -> Float64

The weight of the defect a block recurrence leaves, in the normalisation [`_kry_weight`](@ref)
gives the fresh Arnoldi chain. `projection` is `G = B'HB` of the block `B`, `residual_gram` the
Gram matrix `[<R_j, R_k>]` of its residual block `R = HB - BG`. With `c(s) = exp(s tau G) e_1` the
defect at time `s` has norm `||R c(s)|| = sqrt(c(s)' Gram c(s))`, and the weight is
`|tau| int_0^1 ||R c(s)|| ds` ([`_residual_integral`](@ref)). With nothing recycled the block IS
the Arnoldi basis and `R` has one column, `beta_k v_{k+1}`, so this reduces to `_kry_weight`.
"""
function _block_weight(
    projection::AbstractMatrix,
    residual_gram::AbstractMatrix,
    tau::ComplexF64;
    state_seed::Bool = true,
)
    defect_norm(coefficients) =
        sqrt(max(real(dot(coefficients, residual_gram * coefficients)), 0.0))
    return _residual_integral(defect_norm, tau .* projection, tau; state_seed = state_seed)
end

"""
    _krylov_enrich_recycled(seed, applyH, frame_start, side, tau, alloc, recycled_vectors;
                            workcap, maxdepth, amp, state_seed = true) -> (frame, stats)

`_krylov_enrich` started from the block `block_basis = orth[seed, recycled_vectors]` instead of
`seed` alone. Every column of the block gets a FRESH image (`n_refresh` applications); the
residual block `Rs = H B - B (B'HB)` (`B` the block) then plays the role of the Arnoldi
remainder, and the defect it leaves is weighed by [`_block_weight`](@ref), which with
`recycled_vectors` empty IS `_kry_weight`: the stop is the production one on every generator,
damped and non-unitary normalisations included. While `amp` times that weight exceeds
`alloc.krylov` the recurrence continues along the defect `tau Rs phi_1(tau B'HB) e_1` (one new
application each). Block vectors are scored by their Galerkin coefficient (relative to the
state's growth when the propagator is not unitary), defect directions by the defect weight, and
appended to the frame by the same rule
(`_extend_frame`), or collected and ranked at the end when `alloc.krylov_rank > 0`
([`_append_ranked`](@ref)).
"""
function _krylov_enrich_recycled(
    seed,
    applyH,
    frame_start,
    side::Symbol,
    tau::ComplexF64,
    alloc::BasisAllocation,
    recycled_vectors::AbstractVector;
    workcap::Int = 0,
    maxdepth::Int = _KRY_MAXDEPTH,
    amp::Float64 = 1.0,
    state_seed::Bool = true,
)
    perp = side === :left ? perp_component : perp_component_right
    fl = side === :left ? 3 : 1
    frame = frame_start
    teff = iszero(tau) ? one(tau) : tau
    seed_norm = norm(seed)
    added, nref, nnew, eta_c, eta_d = 0, 0, 0, 0.0, 0.0
    capped, closed = false, false
    ranked = alloc.krylov_rank > 0
    candidates = Any[]                                   # the ranked rule's collected residuals
    # the block: the normalised seed, then each recycled vector that survives orthogonalisation
    block_basis, ws = Any[TLArray((1.0 / seed_norm) * seed)], Float64[amp]
    for recycled_vector in recycled_vectors
        orth_part = _orth_against(block_basis, recycled_vector)
        nx = norm(orth_part)
        nx > 1e-8 * max(norm(recycled_vector), eps()) &&
            push!(block_basis, TLArray((1.0 / nx) * orth_part))
    end
    nrec = length(block_basis) - 1
    # Score `vector` by `weight` times its part outside the current frame and append it to the
    # frame when that exceeds `alloc.krylov`; returns false once the work cap stops growth.
    function score!(vector, weight, first)
        perp_part = TLArray(perp(frame, vector))
        eta = weight * norm(perp_part)
        first ? (eta_c = max(eta_c, eta)) : (eta_d = max(eta_d, eta))
        eta > alloc.krylov || return true
        # ranked: collect against the unchanged `frame_start`, select after the recurrence
        ranked && (push!(candidates, TLArray(weight * perp_part)); return true)
        room = workcap > 0 ? workcap - leg_dim(frame, fl) : typemax(Int)
        room <= 0 && return false
        frame, nadd, fits =
            _extend_frame(frame, perp_part, vector, side, weight, alloc.krylov; room = room)
        added += nadd
        return fits
    end
    nmax = length(block_basis) + maxdepth
    Hm = zeros(ComplexF64, nmax + 1, nmax + 1)
    Rs = Any[]
    # Fill column `column` of `Hm` from a fresh image of block vector `column` and push that
    # image's residual (orthogonal to the first `column` block vectors) onto `Rs`.
    function image!(column)
        image = applyH(block_basis[column])
        for row = 1:column
            Hm[row, column] = tensor_inner(block_basis[row], image)
        end
        residual = _orth_against(block_basis[1:column], image)
        for row = 1:column                              # the CGS2 correction, folded back
            Hm[row, column] += tensor_inner(block_basis[row], residual)
        end
        push!(Rs, _orth_against(block_basis[1:column], residual))
    end
    # the block: images of the seed and the recycled vectors (refreshed, never reused)
    for column = 1:length(block_basis)
        image!(column)
        nref += 1
        # later block vectors' residuals must also be orthogonal to the earlier block vectors
    end
    for column = 1:length(block_basis), row = (column+1):length(block_basis)
        ov = tensor_inner(block_basis[row], Rs[column])
        Hm[row, column] += ov
        Rs[column] = TLArray(Rs[column] - ov * block_basis[row])
    end
    basis_count = length(block_basis)
    if nrec > 0
        # Galerkin coefficients of the block over the step score the recycled vectors
        generator = teff .* Hm[1:basis_count, 1:basis_count]
        cG = exp(generator)[:, 1]
        # relative to the state's growth when the propagator is not unitary, as the defect weight
        unitary = norm(generator + generator') <= _KRY_SKEW_TOL * norm(generator)
        growth = unitary || !state_seed ? 1.0 : max(norm(cG), floatmin(Float64))
        for column = 2:basis_count
            coeff_weight = amp * abs(cG[column]) / growth
            push!(ws, coeff_weight)
            score!(block_basis[column], coeff_weight, column == 2) || (capped = true)
        end
    end
    while !capped
        # invariant: the residual block vanishes (the fresh chain's `beta_k` breakdown)
        residual_gram = ComplexF64[
            tensor_inner(Rs[row], Rs[column]) for row = 1:basis_count, column = 1:basis_count
        ]
        sqrt(max(real(tr(residual_gram)), 0.0)) <= _KRY_BREAKDOWN && (closed = true; break)
        # time-weighted defect: the residuals combined with tau * phi_1(tau Hm) e_1
        ph = _phi1_e1(teff .* Hm[1:basis_count, 1:basis_count])
        defect = nothing
        for column = 1:basis_count
            term = (teff * ph[column]) * Rs[column]
            defect = defect === nothing ? term : defect + term
        end
        defect = TLArray(defect)
        nd = norm(defect)
        # Saad's defect can cancel while the residual does not: continue then along the heaviest
        # residual column instead
        if nd <= _KRY_BREAKDOWN
            defect = Rs[argmax([norm(residual) for residual in Rs])]
            nd = norm(defect)
        end
        weight =
            amp * _block_weight(
                Hm[1:basis_count, 1:basis_count],
                residual_gram,
                teff;
                state_seed = state_seed,
            )
        if weight <= alloc.krylov
            # kept, never applied: the active frontier a next round would continue from (as
            # `_krylov_enrich` keeps its last vector)
            push!(block_basis, TLArray((1.0 / nd) * defect))
            push!(ws, weight)
            closed = true
            break
        end
        basis_count >= nmax && (capped = true; break)
        bn = TLArray((1.0 / nd) * _orth_against(block_basis, defect))
        bn = TLArray((1.0 / norm(bn)) * bn)
        push!(block_basis, bn)
        push!(ws, weight)
        for column = 1:basis_count                      # old residuals lose their bn part
            ov = tensor_inner(bn, Rs[column])
            Hm[basis_count+1, column] += ov
            Rs[column] = TLArray(Rs[column] - ov * bn)
        end
        image!(basis_count + 1)
        nnew += 1
        basis_count += 1
        score!(bn, weight, basis_count == 2) || (capped = true)
    end
    if ranked
        frame, added, fits =
            _append_ranked(frame_start, candidates, side, alloc.krylov_rank, workcap)
        capped |= !fits
    end
    return (
        frame,
        (
            eta_c = eta_c,
            eta_d = eta_d,
            added = added,
            n_h1 = nref + nnew,
            d_after_c = eta_c <= alloc.krylov && added > 0,
            capped = capped,
            met = closed && !capped,
            basis = block_basis,
            weights = ws,
            n_recycled = nrec,
            n_refresh = nref,
            n_new = nnew,
        ),
    )
end

"""
    _recycled_or_fresh(rec, side, bond, entry, transport, seed, applyH, frame_start, tau, alloc;
                       kw...)

The bond's local Krylov stage: recycled from the entry `entry` (transported by `transport()`)
when there is one and the step is not imaginary time, `_krylov_enrich` otherwise. Logs the visit.
Real time recycles damped or not (`tau = (-i - gamma) h`); a gate on `real(tau) != 0` sent every
damped step to a fresh solve silently, so a damped "recycled" run was the fresh one.
"""
function _recycled_or_fresh(
    rec,
    side::Symbol,
    bond::Int,
    entry,
    transport,
    seed,
    applyH,
    frame_start,
    tau::ComplexF64,
    alloc::BasisAllocation;
    round::Int,
    kw...,
)
    imaginary_time = imag(tau) == 0 && real(tau) != 0
    if rec === nothing || entry === nothing || imaginary_time || get(kw, :fixed_depth, 0) > 0
        return _krylov_enrich(seed, applyH, frame_start, side, tau, alloc; kw...)
    end
    start_time = time_ns()
    recycled_vectors = transport()
    tt = (time_ns() - start_time) / 1e9
    if rec.mode === :frame
        # seed the frame with each transported vector whose weighted residual is large enough
        perp = side === :left ? perp_component : perp_component_right
        frame, nseed = frame_start, 0
        for (vector, weight) in zip(recycled_vectors, entry.weights)
            perp_part = TLArray(perp(frame, vector))
            weight * norm(perp_part) > alloc.krylov || continue
            frame, seeded, _ = _extend_frame(frame, perp_part, vector, side, weight, alloc.krylov)
            nseed += seeded
        end
        frame, st = _krylov_enrich(seed, applyH, frame, side, tau, alloc; kw...)
        push!(
            rec.log[side],
            (
                round = round,
                bond = bond,
                cached = length(entry.basis),
                recycled = nseed,
                n_refresh = 0,
                n_new = st.n_h1,
                added = st.added + nseed,
                t_transport = tt,
                met = st.met,
                capped = st.capped,
            ),
        )
        return frame, st
    end
    if rec.mode === :complement
        frame, st = _krylov_enrich(
            seed,
            applyH,
            frame_start,
            side,
            tau,
            alloc;
            kw...,
            recycled = recycled_vectors,
        )
        push!(
            rec.log[side],
            (
                round = round,
                bond = bond,
                cached = length(entry.basis),
                recycled = st.n_recycled,
                n_refresh = 0,
                n_new = st.n_h1,
                added = st.added,
                t_transport = tt,
                met = st.met,
                capped = st.capped,
                rec_closed = st.rec_closed,
            ),
        )
        return frame, st
    end
    frame, st = _krylov_enrich_recycled(
        seed,
        applyH,
        frame_start,
        side,
        tau,
        alloc,
        recycled_vectors;
        workcap = get(kw, :workcap, 0),
        amp = get(kw, :amp, 1.0),
        state_seed = get(kw, :state_seed, true),
    )
    push!(
        rec.log[side],
        (
            round = round,
            bond = bond,
            cached = length(entry.basis),
            recycled = st.n_recycled,
            n_refresh = st.n_refresh,
            n_new = st.n_new,
            added = st.added,
            t_transport = tt,
            met = st.met,
            capped = st.capped,
            eta_c = st.eta_c,
            eta_d = st.eta_d,
        ),
    )
    return frame, st
end
