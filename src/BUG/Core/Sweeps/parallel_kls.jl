# ── fully parallel K/L/S integrator ──
# Not a schedule of the standard step: K, L and the root core S are projected equations at bond `c`,
# run from `Y_n` at once. The new directions `U_+`, `V_+` are the parts of K(h) and L(h) outside the
# root frames. The zero block of the core is O(h^2), measured every step; above `tol_basis` the step
# falls back to the standard root solve.

const KLS_SCHEDULES = (:sequential, :kl_then_s, :concurrent)

# The value of a stage that may still be running: fetches a `Task`, passes anything else through.
_join(value) = value isa Task ? fetch(value) : value

"""
    _kls_start(psi, mpo, left_frames, right_frames, root_link, lenv, renv, stage, eshift,
               schedule) -> NamedTuple

The K and L stages. `lenv`/`renv` are the root environments (links `root_link` and
`root_link+2`, frame basis), `stage(apply, vector, report)` the step's root integrator (`expv` at
`tol_root`, its `root_method`). `:sequential` and `:kl_then_s` return the solved stages;
`:concurrent` returns them as running tasks, which `_kls_finish` joins after the root solve.
Nothing here writes `psi`, `left_frames` or `right_frames`, and each stage owns its operator,
environments and counters.
"""
function _kls_start(
    psi::SymMPS,
    mpo,
    left_frames,
    right_frames,
    root_link::Int,
    lenv,
    renv,
    stage,
    eshift,
    schedule::Symbol,
)
    schedule in KLS_SCHEDULES ||
        throw(ArgumentError("the K/L/S schedule must be one of $(KLS_SCHEDULES), got $schedule"))
    site_count = length(psi)
    # The frames' carries up to the root's neighbours, in any gauge (full contractions).
    CL = root_link == 1 ? nothing : _overlap_left(psi, left_frames, root_link - 1)                     # (b_{c-1}, link_c)
    DR = root_link == site_count - 1 ? nothing : _overlap_right(psi, right_frames, root_link + 1)                   # (link_{c+2}, b_{c+1})
    Kc = CL === nothing ? psi[root_link] : contract(CL, (2,), psi[root_link], (1,))
    Lc =
        DR === nothing ? psi[root_link+1] : contract(psi[root_link+1], (3,), DR, (1,))
    AR = contract(Lc, (2, 3), right_frames[root_link]', (2, 3))  # (link_{c+1}, b_R)
    AL = contract(left_frames[root_link]', (1, 2), Kc, (1, 2))   # (b_L, link_{c+1})
    k_start = contract(Kc, (3,), AR, (1,))                   # (b_{c-1}, s_c, b_R)
    l_start = contract(AL, (2,), Lc, (1,))                   # (b_L, s_{c+1}, b_{c+1})
    kK, kL = KernelCounters(), KernelCounters()
    # K sees the root's right frame as its right environment, L the root's left frame
    HK = OneSiteOp(
        shift_h(
            one_site_h(
                mpo,
                root_link,
                lenv,
                push_right_channels(renv, mpo, right_frames[root_link], root_link + 1),
            ),
            eshift,
        ),
        root_link,
        kK,
    )
    HL = OneSiteOp(
        shift_h(
            one_site_h(
                mpo,
                root_link + 1,
                push_left_channels(lenv, mpo, left_frames[root_link], root_link),
                renv,
            ),
            eshift,
        ),
        root_link,
        kL,
    )
    rK, rL = Ref{Any}(nothing), Ref{Any}(nothing)
    # each stage solve returns `(solution, seconds, bytes)` from `_measure`
    solveK() = _measure(() -> stage(vector -> apply_one_site(HK, vector), k_start, rK))
    solveL() = _measure(() -> stage(vector -> apply_one_site(HL, vector), l_start, rL))
    k_stage, Lh = if schedule === :sequential
        (solveK(), solveL())
    elseif schedule === :kl_then_s
        k_task = Threads.@spawn solveK()
        l_stage = solveL()
        (fetch(k_task), l_stage)
    else
        (Threads.@spawn(solveK()), Threads.@spawn(solveL()))
    end
    return (; schedule, k_start, l_start, HK, HL, k_stage, l_stage = Lh, rK, rL, kK, kL)
end

"""
    _kls_finish(st, mpo, left_frames, right_frames, root_link, lenv, renv, root_start, Sh, tau,
                solve0; nrm, tol_missing) ->
    (left_root, right_root, root_result, Up, Vp, k_result, l_result, omitted_block,
     account, kc)

Join the K/L stages of `st`, build `U_+`/`V_+` and the four-block core from the root solution
`Sh = S(h)`, and measure the omitted block's leading term `M11 = tau U_+' H Theta_n V_+'`
(`Theta_n = W[c] S0 Z[c]` with `S0 = root_start`, streamed against the thin `V_+`, never formed
as `H Theta`). When `||M11|| / nrm > tol_missing` the core is instead `solve0(H0, root_start)` in
the enlarged frames, the standard augmented root step. The returned `W`/`Z` are the root frames to
use from here on (`W[c]`, `Z[c]` grown); the caller's vectors are not written.
"""
function _kls_finish(
    st,
    mpo,
    left_frames,
    right_frames,
    root_link::Int,
    lenv,
    renv,
    root_start,
    Sh,
    tau::ComplexF64,
    solve0;
    nrm::Real = 1.0,
    tol_missing::Real = Inf,
)
    # the root's own frames W[c], Z[c] (only read here, never written)
    left_root, right_root = left_frames[root_link], right_frames[root_link]
    (Kh, tK, _), (Lh, tL, _) = _join(st.k_stage), _join(st.l_stage)
    Up, _ = _preselect_left(left_root, Kh, 0.0)            # numerical rank only: no physical cut
    Vp, _ = _preselect_right(right_root, Lh, 0.0)
    Up === nothing || (Up = TLArray(setitag(Up, 3, Telum.inds(left_root)[3].itags)))
    Vp === nothing || (Vp = TLArray(setitag(Vp, 1, Telum.inds(right_root)[1].itags)))
    Wn = Up === nothing ? left_root : TLArray(oplus([left_root, Up], (3,)))
    Zn = Vp === nothing ? right_root : TLArray(oplus([right_root, Vp], (1,)))
    EL = contract(Wn', (1, 2), left_root, (1, 2))            # (b_L', b_L)
    ER = contract(right_root, (2, 3), Zn', (2, 3))           # (b_R, b_R')
    # a core in the old root frames, rewritten in the enlarged ones: EL core ER
    embed(core) =
        contract(contract(EL, (2,), core, (1,)), (2,), ER, (1,))
    kc = merge_counters!(merge_counters!(KernelCounters(), st.kK), st.kL)
    omitted_block, missing = nothing, 0.0
    if Up !== nothing && Vp !== nothing                    # one side alone leaves no omitted block
        HV, dt, db = _measure(
            () -> _stream_left(
                TLArray(left_root * root_start),
                right_root,
                mpo,
                root_link,
                lenv,
                renv,
                Vp,
            ),
        )
        kc.n_stream += 1
        kc.n_stream_cols += leg_dim(Vp, 1)
        kc.t_stream += dt
        kc.b_stream += db
        # (b_+L, b_+R)
        omitted_block = _scaled(tau, contract(Up', (1, 2), HV, (1, 2)))
        missing = norm(omitted_block) / max(nrm, eps())
    end
    fallback = missing > tol_missing
    root_result = if fallback
        solve0(zero_site_h(mpo, root_link, lenv, renv, Wn, Zn), embed(root_start))
    else
        # S(h) embedded, plus the new-direction blocks U_+' K(h) and L(h) V_+'
        acc = embed(Sh)
        Up === nothing || (
            acc =
                acc + 
                    contract(
                        contract(Wn', (1, 2), perp_component(left_root, Kh), (1, 2)),
                        (2,),
                        ER,
                        (1,),
                    ),
                
        )
        Vp === nothing || (
            acc =
                acc + 
                    contract(
                        EL,
                        (2,),
                        
                            contract(perp_component_right(right_root, Lh), (2, 3), Zn', (2, 3)),
                        ,
                        (1,),
                    ),
                
        )
        TLArray(acc)
    end
    account = (
        schedule = st.schedule,
        n_new_l = Up === nothing ? 0 : leg_dim(Up, 3),
        n_new_r = Vp === nothing ? 0 : leg_dim(Vp, 1),
        missing = missing,
        fallback = fallback,
        n_K = st.kK.n_one_site,
        n_L = st.kL.n_one_site,
        err_K = st.rK[].err,
        err_L = st.rL[].err,
        t_K = tK,
        t_L = tL,
    )
    return (
        left_root = Wn,
        right_root = Zn,
        root_result = root_result,
        Up = Up,
        Vp = Vp,
        k_result = Kh,
        l_result = Lh,
        omitted_block = omitted_block,
        account = account,
        kc = kc,
    )
end

"""
    _parallel_kls(psi, mpo, fr, root_start, tau, stage; schedule = :sequential, nrm, tol_missing)

The whole K/L/S kernel on frames `fr` (`KLFrames`): `_kls_start`, the root solve of `root_start`
and `_kls_finish`. `cbe_bug_step!` interleaves the three itself; this is the kernel the tests drive.
Returns `_kls_finish`'s NamedTuple.
"""
function _parallel_kls(
    psi::SymMPS,
    mpo,
    fr::KLFrames,
    root_start,
    tau::ComplexF64,
    stage;
    schedule::Symbol = :sequential,
    nrm::Real = 1.0,
    tol_missing::Real = Inf,
)
    root_link = fr.root_link
    st = _kls_start(
        psi,
        mpo,
        fr.left_frames,
        fr.right_frames,
        root_link,
        fr.lenv,
        fr.renv,
        stage,
        fr.eshift,
        schedule,
    )
    # one root solve of `core` under the zero-site operator `operator`, shifted like the frames
    root(operator, core) = stage(
        vector -> apply_zero_site(shift_h(operator, fr.eshift), vector),
        core,
        Ref{Any}(nothing),
    )
    Sh = root(
        zero_site_h(
            mpo,
            root_link,
            fr.lenv,
            fr.renv,
            fr.left_frames[root_link],
            fr.right_frames[root_link],
        ),
        root_start,
    )
    return _kls_finish(
        st,
        mpo,
        fr.left_frames,
        fr.right_frames,
        root_link,
        fr.lenv,
        fr.renv,
        root_start,
        Sh,
        tau,
        root;
        nrm = nrm,
        tol_missing = tol_missing,
    )
end
