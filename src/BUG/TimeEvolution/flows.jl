# Flows: the vector fields a nonlinear BUG step integrates.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_flows.jl (`QTTFlow`, `LinearFlow`,
# `flow_op`, `flow_jac_ops`, `frozen_flow`).
#
# A flow is d_t psi = F(t, psi) = pref * (H[psi, t] psi - mu[psi] psi). H[psi, t] is rebuilt from the
# state it acts on at every evaluation. The real-linear Jacobian is
#   DF[d] = pref * ((J1 - mu) d + J2 conj(d)),
# with J2 = nothing for a complex-linear flow. Operators are dense operator cores
# `(w_l, out, in, w_r)`; a domain (QTTS) supplies the nonlinear flows and the root space they act in.

"""
    QTTFlow

A vector field `F(t, psi) = pref (H[psi, t] - mu[psi]) psi` on operator cores. Subtypes have a field
`pref` and methods of [`flow_op`](@ref) and [`flow_jac_ops`](@ref); [`flow_normalized`](@ref) says
whether `mu[psi] = <psi, H psi> / <psi, psi>` is subtracted (default: no).
"""
abstract type QTTFlow end

"""
    LinearFlow(H, pref = 1)

`F = pref H psi` for a fixed operator `H` (operator cores): a state-independent generator, e.g. the
linear Gross-Pitaevskii or Heston operators.
"""
struct LinearFlow <: QTTFlow
    H::Vector{Array{ComplexF64,4}}
    pref::ComplexF64
end
LinearFlow(H::AbstractVector, pref::Number = 1.0) =
    LinearFlow([ComplexF64.(core) for core in H], ComplexF64(pref))

"Largest interior bond dimension of operator cores."
_op_bond_max(H::AbstractVector) = maximum((size(core, 4) for core in H[1:(end-1)]); init = 1)

"""
    flow_op(fl, A, t) -> (H, info)

The operator `H[A, t]` (cores) of flow `fl` at the state cores `A`, and an info tuple with the
density rank `chi_rho`, its error bound `rho_err`, its build seconds `t_rho`, whether the build was
`retried`, and the operator's bond dimension `w`.
"""
function flow_op end
flow_op(fl::LinearFlow, A, t) =
    (fl.H, (chi_rho = 0, rho_err = 0.0, t_rho = 0.0, retried = false, w = _op_bond_max(fl.H)))

"""
    flow_jac_ops(fl, A, t) -> (J1, J2)

The operators of the real-linear Jacobian at `A` (file header); `J2 = nothing` when it vanishes.
"""
function flow_jac_ops end
flow_jac_ops(fl::LinearFlow, A, t) = (fl.H, nothing)

"Whether the flow subtracts `mu[psi]` (the normalised gradient flow)."
flow_normalized(::QTTFlow) = false

"A frozen flow: the linear flow of `H[A, t]` (the lagged-density ablation)."
function frozen_flow(fl::QTTFlow, A, t)
    H, _ = flow_op(fl, A, t)
    return LinearFlow(H, fl.pref)
end
