"""
    Baselines

Independent comparison methods: TDVP (`TDVP/`) for dynamics and CBE-DMRG for ground states
(`DMRG/`). They use `BUG`'s `Core` and `Randomised` layers (environments, Krylov solvers, bond
expansion), never its BUG step, so that a comparison never runs the method it is compared with.
The QTT TDVP baselines also read the flow definitions of `BUG/TimeEvolution/flows.jl` (the vector
field, not the integrator) and the dense QTT contractions of `QTTS`.
"""
module Baselines

using LinearAlgebra, Random
using LurCGT, Telum
using ..BUG
using ..QTTS
# BUG internals the sweeps share: truncation rules, the state's stored size, the root link, the
# channel and environment kernels, the bond-expansion ledger hooks.
using ..BUG:
    _state_stored,
    _budget_svd,
    _trunc_weight,
    _sq_weight,
    _cut_divisor,
    _bug_root_link,
    _absorb_right!,
    _absorb_left!,
    _unmet!,
    _log_bond!,
    ADAPTIVE_GROWTH,
    TOL_DEFAULTS,
    _replace_state!
# the dense QTT contractions the QTT TDVP baselines build their environments from
using ..QTTS: _tc, _env_left, _env_right, _unit_env, _op_expect, _qtt_rank

include("TDVP/tdvp2.jl")
include("TDVP/tdvp1_cbe.jl")
include("TDVP/qtt_tdvp.jl")
include("DMRG/dmrg_cbe.jl")

# two-site TDVP and its compositions (TDVP/tdvp2.jl)
export TDVP2Info, tdvp2_step!, tdvp2_composed_step!
# one-site TDVP with CBE (TDVP/tdvp1_cbe.jl)
export TDVP1CBEInfo, tdvp1_cbe_step!, tdvp1_rsvd_cbe_step!, tdvp1_cbe_exact_step!
export TDVPCBEInfo, tdvp_cbe1s_step!
# QTT TDVP (TDVP/qtt_tdvp.jl)
export qtt_tdvp2_step!, qtt_split_tdvp_step!, tt_expv
# CBE-DMRG (DMRG/dmrg_cbe.jl)
export DMRGCBEInfo, DMRG_TOL_DEFAULTS, dmrg_etol, dmrg_cbe1s_sweep!, DMRGCBERun, dmrg_cbe1s!
export rsvd_cbe_dmrg1s!, exact_cbe_dmrg1s!, rsvd_cbe_dmrg1s_seeded!, GroundState

end
