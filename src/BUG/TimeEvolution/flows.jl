# Flows: the vector fields a nonlinear BUG step integrates.
#
# A flow is d_t psi = F(t, psi) = pref * (H[psi, t] psi - mu[psi] psi). H[psi, t] is rebuilt from
# the state it acts on at every evaluation. The real-linear Jacobian is
#   DF[d] = pref * ((J1 - mu) d + J2 conj(d)),
# with J2 = nothing for a complex-linear flow. Operators are dense operator cores
# `(w_l, out, in, w_r)`; a domain (QTTS) supplies the nonlinear flows and the root space they act
# in.

"""
    QTTFlow

A vector field `F(t, psi) = pref (H[psi, t] - mu[psi]) psi` on operator cores. Subtypes have a field
`pref` and methods of [`flow_op`](@ref) and [`flow_jac_ops`](@ref); [`flow_normalized`](@ref) says
whether `mu[psi] = <psi, H psi> / <psi, psi>` is subtracted (default: no).
"""
abstract type QTTFlow end

"""
    LinearFlow(operator_cores, pref = 1)

`F = pref H psi` for a fixed operator `H` (the field `H`, built from `operator_cores`): a
state-independent generator, e.g. the linear Gross-Pitaevskii or Heston operators.
"""
struct LinearFlow <: QTTFlow
    H::Vector{Array{ComplexF64,4}}
    pref::ComplexF64
end
LinearFlow(operator_cores::AbstractVector, pref::Number = 1.0) =
    LinearFlow([ComplexF64.(core) for core in operator_cores], ComplexF64(pref))

"Largest interior bond dimension of operator cores."
_op_bond_max(operator_cores::AbstractVector) =
    maximum((size(core, 4) for core in operator_cores[1:(end-1)]); init = 1)

"""
    flow_op(fl, state_cores, field_time) -> (H, info)

The operator `H[psi, t]` (cores) of flow `fl` at the state cores `state_cores` and the time
`field_time`, and an info tuple with the density rank `chi_rho`, its error bound `rho_err`, its
build seconds `t_rho`, whether the build was `retried`, and the operator's bond dimension `w`.
"""
function flow_op end
flow_op(fl::LinearFlow, state_cores, field_time) =
    (fl.H, (chi_rho = 0, rho_err = 0.0, t_rho = 0.0, retried = false, w = _op_bond_max(fl.H)))

"""
    flow_jac_ops(fl, state_cores, field_time; rho = nothing) -> (J1, J2)

The operators of the real-linear Jacobian at `state_cores` (file header); `J2 = nothing` when it
vanishes. `rho` is a density of the same state the flow already built (the field's info), reused
instead of rebuilt; flows without a density ignore it.
"""
function flow_jac_ops end
flow_jac_ops(fl::LinearFlow, state_cores, field_time; rho = nothing) = (fl.H, nothing)

"Whether the flow subtracts `mu[psi]` (the normalised gradient flow)."
flow_normalized(::QTTFlow) = false

"A frozen flow: the linear flow of `H[psi, t]` at `state_cores` (the lagged-density ablation)."
function frozen_flow(fl::QTTFlow, state_cores, field_time)
    operator_cores, _ = flow_op(fl, state_cores, field_time)
    return LinearFlow(operator_cores, fl.pref)
end
