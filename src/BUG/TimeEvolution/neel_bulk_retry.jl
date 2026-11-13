# Transactional rank-budget escalation for a bulk quench (e.g. a Neel state).
# A rejected candidate NEVER advances the physical state. Each retry uses the
# identical starting MPS and an identical random seed. All attempted work counts.
"""
    cbe_bug_bulk_retry_step!(psi, mpo, tau; modes=(:bulk,:maxdim), kwargs...)

Attempt the same time step with successively less restrictive working-frame
ceilings. Only a candidate whose *entire* BUG ledger meets tolerance is
committed. If all candidates fail, the input `psi` remains unchanged and
`accepted=false`; the caller must reduce dt or loosen the actual tolerance
rather than silently advancing a rejected step.

This is a conservative diagnostic-driven fallback, not a mathematical
guarantee that basis-test estimates bound the global-state error. Every trial
pays its measured runtime and operator calls; no misleading savings from
discarded trials. `maxdim` continues to bound the final compressed MPS.
"""
function cbe_bug_bulk_retry_step!(
    psi::SymMPS,
    mpo::MPO,
    tau::ComplexF64;
    modes::Tuple = (:bulk, :maxdim),
    bond_ceiling_factor::Real = 1.2,
    bulk_min_width::Int = 8,
    bulk_headroom::Float64 = 0.25,
    rng::AbstractRNG = MersenneTwister(0x5EED),
    kwargs...,
)
    isempty(modes) && throw(ArgumentError("modes cannot be empty"))
    all(m -> m in (:bond,:bulk,:maxdim), modes) ||
        throw(ArgumentError("modes must use :bond, :bulk or :maxdim"))
    initial = deepcopy(psi)
    seed = rand(rng, UInt64)
    history = NamedTuple[]
    for mode in modes
        trial = deepcopy(initial)
        started = time_ns()
        info = cbe_bug_step!(
            trial, mpo, tau;
            work_ceiling_rule=mode,
            bond_ceiling_factor=mode === :maxdim ? 0.0 : bond_ceiling_factor,
            bulk_min_width=bulk_min_width,
            bulk_headroom=bulk_headroom,
            rng=MersenneTwister(seed),
            kwargs...,
        )
        measured = (time_ns()-started)/1e9
        counters = info.record.counters
        ledger = info.ledger
        push!(history, (
            cap=mode, accepted=ledger.tolerance_met,
            seconds=measured, h1=counters.n_one_site,
            stream_columns=counters.n_stream_cols,
            stream_calls=counters.n_stream,
            root_h=counters.n_root,
            unmet=copy(ledger.unmet),
            D_final=maximum(bond_dims(trial)),
            D_work=info.record.D_stage,
        ))
        if ledger.tolerance_met
            psi.tensors = trial.tensors
            psi.center = trial.center
            return (
                accepted=true, info=info,
                history=history,
                seconds=sum(x.seconds for x in history),
                h1=sum(x.h1 for x in history),
                stream_columns=sum(x.stream_columns for x in history),
                root_h=sum(x.root_h for x in history),
            )
        end
    end
    return (
        accepted=false, info=nothing,
        history=history, seconds=sum(x.seconds for x in history),
        h1=sum(x.h1 for x in history),
        stream_columns=sum(x.stream_columns for x in history),
        root_h=sum(x.root_h for x in history),
    )
end
