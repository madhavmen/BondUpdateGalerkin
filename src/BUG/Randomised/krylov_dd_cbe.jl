# Thin U(1)-resolved DD probing of a one-site Krylov direction. The fixed
# opposite frame makes the current Krylov operator unchanged during enrichment.
#
# This is the experimental A/B policy, not part of the default BUG step.
function _krylov_dd_sketch(
    direction,
    current_frame,
    opposite_frame,
    mpo::MPO,
    bond::Int,
    lenv::MPOLink,
    renv::MPOLink,
    side::Symbol,
    width::Int,
    rng::AbstractRNG,
    kc::KernelCounters,
)
    width >= 1 || throw(ArgumentError("Krylov DD sketch width must be positive"))
    opposite = side === :left ? :right : :left
    Om = _dd_probe(opposite_frame, opposite, width; rng = rng, kc = kc)
    Om === nothing && return (nothing, 0)
    cols = leg_dim(Om, side === :left ? 1 : 3)
    # The Krylov direction is a one-site amplitude; reconstruct the two-site
    # candidate implicitly as direction * opposite_frame, never as a dense matrix.
    y, elapsed, bytes = _measure(
        () -> side === :left ?
              _stream_left(direction, opposite_frame, mpo, bond, lenv, renv, Om) :
              _stream_right(opposite_frame, direction, mpo, bond, lenv, renv, Om),
    )
    kc.n_stream += 1
    kc.n_stream_cols += cols
    kc.t_stream += elapsed
    kc.b_stream += bytes
    perp = side === :left ? perp_component : perp_component_right
    residual = TLArray(perp(current_frame, y))
    return (norm(residual) > _CBE_EPS ? residual : nothing, cols)
end
