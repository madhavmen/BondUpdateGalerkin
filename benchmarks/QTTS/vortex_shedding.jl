# Damped nonlinear Gross-Pitaevskii vortex shedding behind a moving paddle: the research-grade QTT
# benchmark (implementation plan WP5). Not a validated example and not a prerequisite for usability.
#
# A homogeneous condensate (g = 1, gamma = 0.1, mu = 1) in a periodic box, relaxed in imaginary time
# around the paddle, then the paddle is swept at v_sweep and ramped down. The time reference is
# dense RK4 of the same finite-difference system (independent code, `common.jl`). Arms: BUG
# p = 2/3/4 at every step size, two-site TDVP with the lagged density, and the split-dissipation
# TDVP control. Per snapshot: N, energies, grand-canonical F, external power, vortex charges on a
# sampled grid (and of the reference), the bond profile, the error against the reference, wall time.
# The paddle is an engineering fixture (`GPPaddle`): a published shape must be read off its source
# before quoting a reproduction. Grouped QTT ordering (the default).
#
#   julia --project benchmarks/QTTS/vortex_shedding.jl
#
# ENV: VS_OUT (output directory, required), VS_N (bits per axis, 8 = 256^2), VS_BOX (64), VS_T (12),
#      VS_HS ("0.005,0.01,0.025,0.05,0.1"), VS_SNAP (1.0), VS_ORDERS ("2,3,4"), VS_MAXDIM (128),
#      VS_REFDT (1e-3), VS_CONSISTENT (0: also run TDVP2 with the stage-consistent density when 1),
#      VS_RELAX (2.0, imaginary-time relaxation of the start with the paddle in place).
# Resource needs at the defaults are not yet measured (the dense reference alone is 65536 unknowns).
#
# Ported from BUG-Julia@b155550:benchmarks/qtt/vortex_shedding.jl (ENV prefix T3V_ -> VS_).

include(joinpath(@__DIR__, "common.jl"))

haskey(ENV, "VS_OUT") || error("set VS_OUT to the output directory")
const OUTPUT_DIR = ENV["VS_OUT"]
const BITS = env_int("VS_N", 8)
const BOX = env_float("VS_BOX", 64.0)
const FINAL_TIME = env_float("VS_T", 12.0)
const STEP_SIZES = env_list("VS_HS", "0.005,0.01,0.025,0.05,0.1")
const SNAPSHOT_EVERY = env_float("VS_SNAP", 1.0)
const ORDERS = round.(Int, env_list("VS_ORDERS", "2,3,4"))
const MAXDIM = env_int("VS_MAXDIM", 128)
mkpath(OUTPUT_DIR)

g = qtt_grid(BITS, BITS; x = (-BOX / 2, BOX / 2), y = (-BOX / 2, BOX / 2))
paddle = GPPaddle(10.0, -BOX / 4, 1.5, BOX / 3 / 1.5, 2.0, 1.0, BOX / 6, 0.0, 0.5)
model = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 1.0, omega = 0.0, paddle = paddle)
# start: the uniform condensate n = mu / g, relaxed in imaginary time around the resting paddle
psi_start = qtt_mps(qtt_separable(g, x -> 1.0 + 0im, y -> 1.0 + 0im))
resting_paddle = GPPaddle(paddle.height, paddle.x_start, 0.0, 1e9, 0.0, paddle.width_x, paddle.width_y, paddle.y_c, paddle.smooth)
relaxation_model = gp_model(g; gnl = 1.0, gamma = 0.0, mu = 1.0, omega = 0.0, paddle = resting_paddle)
start_norm = qtt_l2norm(psi_start, g)^2
relaxation_time = env_float("VS_RELAX", 2.0)
relaxation_time > 0 && gp_ground_state!(
    psi_start,
    relaxation_model,
    uniform_schedule(0.1, round(Int, relaxation_time / 0.1));
    order = 2,
    maxdim = MAXDIM,
)
qtt_normalize!(psi_start, g; N = start_norm)
start_cores = qtt_cores(psi_start)
snapshot_times = collect(SNAPSHOT_EVERY:SNAPSHOT_EVERY:FINAL_TIME)

# the reference trajectory at the snapshot times
reference_dt = env_float("VS_REFDT", 1e-3)
reference = Dict{Float64,Vector{ComplexF64}}()
reference_state = grid_vec(start_cores, g)
reference_time = 0.0
reference_seconds = @elapsed for snapshot in snapshot_times
    global reference_state = gp_dense_rk4(model, reference_state, reference_time, snapshot; dt = reference_dt)
    global reference_time = snapshot
    reference[snapshot] = copy(reference_state)
end
println("reference done in $(round(reference_seconds; digits = 1)) s")

rows = NamedTuple[]
"Record the observables of `psi` at snapshot `t` against the reference."
function observe!(rows, arm, h, t, psi, wall, extra = (;))
    A = qtt_cores(psi)
    observed = gp_observables(model, A, t)
    stride = max(1, 2^BITS ÷ 256)
    _, _, sampled = qtt_sample(A, g; stride = stride)
    vortices_plus, vortices_minus, _ = vortex_count(sampled)
    reference_values = reference[t]
    values = grid_vec(A, g)
    reference_plus, reference_minus, _ =
        vortex_count(reshape(reference_values, 2^BITS, 2^BITS)[1:stride:end, 1:stride:end])
    push!(
        rows,
        merge(
            (
                arm = arm,
                h = h,
                t = t,
                N = observed.N,
                E_kin = observed.E_kin,
                E_int = observed.E_int,
                E_pad = observed.E_pad,
                F_gc = observed.F_gc,
                P_ext = observed.P_ext,
                vortices_plus = vortices_plus,
                vortices_minus = vortices_minus,
                ref_vortices_plus = reference_plus,
                ref_vortices_minus = reference_minus,
                err_state = norm(values - reference_values) / norm(reference_values),
                err_density = norm(abs2.(values) - abs2.(reference_values)) / norm(abs2.(reference_values)),
                chi_max = observed.chi,
                chi_profile = join(bond_dims(psi), ";"),
                wall = wall,
            ),
            extra,
        ),
    )
    println(rows[end])
    flush(stdout)
end

arms = Any[]
for order in ORDERS, h in STEP_SIZES
    push!(
        arms,
        (
            Symbol("bug$order"),
            h,
            (psi, t, step) ->
                qtt_bug_step!(psi, GPFlow(model), t, step; order = order, maxdim = MAXDIM, tol_basis = 1e-6, tol_trunc = 1e-8),
        ),
    )
end
for h in STEP_SIZES
    push!(arms, (:tdvp2_lagged, h, (psi, t, step) -> qtt_tdvp2_step!(psi, GPFlow(model), t, step; maxdim = MAXDIM, tol_trunc = 1e-8)))
    push!(arms, (:split, h, (psi, t, step) -> qtt_split_tdvp_step!(psi, model, t, step; maxdim = MAXDIM, tol_trunc = 1e-8)))
    env_int("VS_CONSISTENT", 0) == 1 && push!(
        arms,
        (
            :tdvp2_consistent,
            h,
            (psi, t, step) ->
                qtt_tdvp2_step!(psi, GPFlow(model), t, step; density = :consistent, maxdim = MAXDIM, tol_trunc = 1e-8),
        ),
    )
end
for (arm, h, step!) in arms
    psi = qtt_mps(start_cores)
    t, wall, steps_taken = 0.0, 0.0, 0
    backward_amplification = 1.0
    failed = false
    for snapshot in snapshot_times
        while t < snapshot - 1e-12
            step = min(h, snapshot - t)
            report = nothing
            seconds = @elapsed try
                report = step!(psi, t, step)
            catch err
                @warn "arm $arm h=$h failed at t=$t" err
                failed = true
            end
            failed && break
            report !== nothing &&
                hasproperty(report, :A_max) &&
                (backward_amplification = max(backward_amplification, report.A_max))
            wall += seconds
            t += step
            steps_taken += 1
        end
        failed && break
        observe!(rows, arm, h, snapshot, psi, wall, (steps = steps_taken, A_max = backward_amplification))
    end
    write_csv(joinpath(OUTPUT_DIR, "shedding.csv"), rows)
end
println("VORTEX-SHEDDING-DONE")
