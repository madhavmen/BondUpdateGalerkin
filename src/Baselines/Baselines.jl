"""
    Baselines

Independent comparison methods for dynamics: two-site TDVP (`TDVP/`). They use `BUG`'s `Core`
layer (environments, Krylov solvers, truncation), never its BUG step, so that a comparison never
runs the method it is compared with.
The QTT TDVP baselines also read the flow definitions of `BUG/TimeEvolution/flows.jl` (the vector
field, not the integrator) and the dense QTT contractions of `QTTS`.
"""
module Baselines

using LinearAlgebra, Random
using LurCGT, Telum
using ..BUG
using ..QTTS
# BUG internals the sweeps share: truncation rules, the state's stored size, the root link and
# the state replacement.
using ..BUG:
    _state_stored,
    _budget_svd,
    _trunc_weight,
    _sq_weight,
    _cut_divisor,
    _bug_root_link,
    _replace_state!
# the dense QTT contractions the QTT TDVP baselines build their environments from
using ..QTTS: _tc, _env_left, _env_right, _unit_env, _qtt_rank

include("TDVP/tdvp2.jl")
include("TDVP/qtt_tdvp.jl")

# two-site TDVP and its compositions (TDVP/tdvp2.jl)
export TDVP2Info, tdvp2_step!
# QTT TDVP (TDVP/qtt_tdvp.jl)
export qtt_tdvp2_step!, qtt_split_tdvp_step!, tt_expv

end
