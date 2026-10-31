# A moving paddle potential for the Gross-Pitaevskii vortex-shedding studies.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_models.jl (`GPPaddle`, `paddle_state`,
# `gp_paddle_op`). The operator is built from the grid and the paddle; `gp_paddle_op(m, t)` for a
# model is in `gross_pitaevskii.jl`.

"""
    GPPaddle(height, x_start, v_sweep, t_stop, ramp, width_x, width_y, y_c, smooth)

A rounded rectangular paddle, `V_pad = A(t) b_x(x - x_p(t)) b_y(y - y_c)` with the smoothed box
`b(u) = (tanh((u + w/2)/s) - tanh((u - w/2)/s)) / 2`, centre `x_p = x_start + v_sweep min(t, t_stop)`
and height ramped linearly to zero over `ramp` after `t_stop`. Separable, so its QTT is two 1D
TT-SVDs at any size. An engineering fixture: the shape of a published study must be read off its
source before a reproduction run.
"""
struct GPPaddle
    height::Float64
    x_start::Float64
    v_sweep::Float64
    t_stop::Float64
    ramp::Float64
    width_x::Float64
    width_y::Float64
    y_c::Float64
    smooth::Float64
end

"The smoothed box of width `w` and edge width `s` at offset `u`."
_box(u, w, s) = (tanh((u + w / 2) / s) - tanh((u - w / 2) / s)) / 2

"Height and centre of the paddle at time `t`."
function paddle_state(p::GPPaddle, t::Real)
    amplitude =
        t <= p.t_stop ? p.height :
        p.ramp > 0 ? p.height * max(0.0, 1 - (t - p.t_stop) / p.ramp) : 0.0
    return amplitude, p.x_start + p.v_sweep * min(t, p.t_stop)
end

"""
    gp_paddle_op(g, paddle, t) -> operator cores or nothing

`V_pad(t)` on grid `g` as a diagonal operator (`nothing` without a paddle or once it is off).
"""
function gp_paddle_op(g::QTTGrid, paddle::Union{Nothing,GPPaddle}, t::Real)
    paddle === nothing && return nothing
    amplitude, centre = paddle_state(paddle, t)
    amplitude == 0 && return nothing
    xs, ys = qtt_coords(g)
    box_x = tt_from_samples(_box.(xs .- centre, paddle.width_x, paddle.smooth))
    box_y = tt_from_samples(_box.(ys .- paddle.y_c, paddle.width_y, paddle.smooth))
    return op_scale(op_diag(qtt_embed(g; x = box_x, y = box_y)), amplitude)
end
