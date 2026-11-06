using HDF5

function _assert_tlarray_structural_equal(a::TLArray, b::TLArray)
    @test symm(a) == symm(b)
    @test Telum.inds(a) == Telum.inds(b)
    @test Telum.spaces(a) == Telum.spaces(b)
end

function _assert_tlarray_roundtrip(a::TLArray, b::TLArray; tol=1e-12)
    _assert_tlarray_structural_equal(a, b)
    @test norm(a - b) / max(norm(a), 1.0) < tol
end

function _with_saved_tlarray(f, q; name="tensor")
    path = tempname() * ".h5"
    try
        save_tlarray(path, q; name)
        loaded = load_tlarray(path; name)
        f(loaded)
    finally
        isfile(path) && rm(path)
    end
end

function _with_zero_sector(q::TLArray)
    qlabels = copy(Telum.stored_qlabels(q))
    push!(qlabels, first(Telum.stored_qlabels(q)))

    wmatdata = copy(q.wmatdata)
    wmatinfo = copy(q.wmatinfo)
    push!(wmatinfo, Telum._empty_wmat_info(Val(length(first(q.wmatinfo)))))

    RMTs = similar(q.RMTs, length(q.RMTs) + 1)
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        RMTs[sector] = deepcopy(_test_sector_rmt(q, sector))
    end

    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, Telum.inds(q), Telum.spaces(q))
end

function _push_copied_wmat_info!(wmatdata, wmatinfo, q::TLArray, sector::Int)
    M = length(q.wmatinfo[sector])
    info = ntuple(Val(M)) do slot
        offset0, nrow, ncol = q.wmatinfo[sector][slot]
        offset0 == 0 && return (0, 0, 0)
        wmat = Telum.sector_wmat_slot(q, sector, slot)
        offset = length(wmatdata) + 1
        append!(wmatdata, vec(wmat))
        return (offset, nrow, ncol)
    end
    push!(wmatinfo, info)
    return wmatdata, wmatinfo
end

function _with_defined_zero_sector(q::TLArray)
    qlabels = copy(Telum.stored_qlabels(q))
    push!(qlabels, first(Telum.stored_qlabels(q)))

    wmatdata = copy(q.wmatdata)
    wmatinfo = copy(q.wmatinfo)
    source_sector = first(i for i in Telum.sector_slots(q) if q.isdefined[i])
    _push_copied_wmat_info!(wmatdata, wmatinfo, q, source_sector)

    RMTs = similar(q.RMTs, length(q.RMTs) + 1)
    for sector in Telum.sector_slots(q)
        q.isdefined[sector] || continue
        RMTs[sector] = deepcopy(_test_sector_rmt(q, sector))
    end
    zero_rmt = deepcopy(_test_sector_rmt(q, source_sector))
    fill!(zero_rmt, zero(eltype(zero_rmt)))
    RMTs[end] = zero_rmt

    return TLArray(symm(q), qlabels, wmatdata, wmatinfo, RMTs, Telum.inds(q), Telum.spaces(q))
end

@testset "HDF5 TLArray round trip" begin
    q_nosym = TLArray((),
                      [((), ())],
                      Float64[],
                      [Telum._empty_wmat_info(Val(0))],
                      [reshape([2.0], 1, 1)],
                      (TLIndex("in", '+'), TLIndex("out", '-')),
                      ([(() , 1)], [(() , 1)]))
    @test symm(q_nosym) == ()
    @test Telum.stored_qlabels(q_nosym) == [((), ())]
    _with_saved_tlarray(q_nosym) do loaded
        _assert_tlarray_roundtrip(q_nosym, loaded)
        @test symm(loaded) == ()
        @test Telum.stored_qlabels(loaded) == [((), ())]
    end

    q0 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing), ("left", "right", "op"))
    q = q0.F

    q_u1 = getLocalSpace(FermionOptions(U1), ("in", "out", "op")).F
    _with_saved_tlarray(q_u1) do loaded
        _assert_tlarray_roundtrip(q_u1, loaded)
    end

    _with_saved_tlarray(q) do loaded
        _assert_tlarray_roundtrip(q, loaded)
    end

    v1_path = tempname() * ".h5"
    try
        save_tlarray(v1_path, q)
        HDF5.h5open(v1_path, "r+") do h5
            g = h5["tensor"]
            attrs = HDF5.attrs(g)
            attrs["schema_version"] = "1.0"
            delete!(attrs, "rmt_eltype")
            delete!(attrs, "conj")
            delete!(attrs, "scale")
            delete!(attrs, "perm")
        end
        loaded_v1 = load_tlarray(v1_path)
        _assert_tlarray_roundtrip(q, loaded_v1)
        @test !Telum.stored_conj(loaded_v1)
        @test Telum.stored_scale(loaded_v1) == one(eltype(loaded_v1))
        @test Telum.stored_perm(loaded_v1) == ntuple(identity, Val(ndims(loaded_v1)))
    finally
        isfile(v1_path) && rm(v1_path)
    end

    q_zero = _with_zero_sector(q)
    @test q_zero.iszero[end]
    @test !q_zero.isdefined[end]
    _with_saved_tlarray(q_zero) do loaded
        _assert_tlarray_roundtrip(q_zero, loaded)
        @test loaded.iszero[end]
        @test !loaded.isdefined[end]
    end

    q_defined_zero = _with_defined_zero_sector(q)
    @test q_defined_zero.iszero[end]
    @test q_defined_zero.isdefined[end]
    @test occursin(r"\t\s*zero", sprint(show, MIME"text/plain"(), q_defined_zero))

    q_absent_zero = _with_zero_sector(q)
    absent_sector = length(q_absent_zero.RMTs)
    @test !occursin("  $absent_sector.\t", sprint(show, MIME"text/plain"(), q_absent_zero))

    _with_saved_tlarray(q_defined_zero) do loaded
        _assert_tlarray_roundtrip(q_defined_zero, loaded)
        @test loaded.iszero[end]
        @test loaded.isdefined[end]
        @test iszero(sum(abs2, _test_sector_rmt(loaded, length(loaded.RMTs))))
    end

    first_active = first(sector for sector in Telum.sector_slots(q) if !q.iszero[sector])
    complex_RMT_type = Array{ComplexF64, ndims(_test_sector_rmt(q, first_active))}
    complex_RMTs = Vector{complex_RMT_type}(undef, length(q.RMTs))
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        complex_RMTs[sector] = ComplexF64.(_test_sector_rmt(q, sector))
    end
    q_complex = TLArray(symm(q), copy(Telum.stored_qlabels(q)), copy(q.wmatdata), copy(q.wmatinfo),
                        complex_RMTs, Telum.inds(q), Telum.spaces(q))
    _with_saved_tlarray(q_complex) do loaded
        _assert_tlarray_roundtrip(q_complex, loaded)
    end

    float32_RMT_type = Array{Float32, ndims(_test_sector_rmt(q, first_active))}
    float32_RMTs = Vector{float32_RMT_type}(undef, length(q.RMTs))
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        float32_RMTs[sector] = Float32.(_test_sector_rmt(q, sector))
    end
    q_float32 = TLArray(symm(q), copy(Telum.stored_qlabels(q)), copy(q.wmatdata), copy(q.wmatinfo),
                        float32_RMTs, Telum.inds(q), Telum.spaces(q))
    q_complex_scaled = (1.0 + 2.0im) * q_float32
    _with_saved_tlarray(q_complex_scaled) do loaded
        _assert_tlarray_roundtrip(q_complex_scaled, loaded; tol=1e-6)
        @test eltype(loaded) === ComplexF64
        @test eltype(typeof(loaded).parameters[end]) === Float32
        @test Telum.stored_scale(loaded) == 1.0 + 2.0im
        @test Telum.stored_perm(loaded) == Telum.stored_perm(q_complex_scaled)
        @test Telum.stored_conj(loaded) == Telum.stored_conj(q_complex_scaled)
    end

    q_view = permutedims(q, (2, 1, 3))
    q_view_eager = copy(q_view)
    _with_saved_tlarray(q_view) do loaded
        _assert_tlarray_roundtrip(q_view_eager, loaded)
        @test Telum.stored_inds(loaded) == Telum.stored_inds(q_view)
        @test Telum.stored_spaces(loaded) == Telum.stored_spaces(q_view)
        @test Telum.stored_perm(loaded) == Telum.stored_perm(q_view)
        @test Telum.stored_scale(loaded) == Telum.stored_scale(q_view)
        @test Telum.stored_conj(loaded) == Telum.stored_conj(q_view)
    end

    q_diag = get1jtensor(q0.I, 1)
    _with_saved_tlarray(q_diag) do loaded
        _assert_tlarray_roundtrip(q_diag, loaded)
    end

    lazy = Telum._lazy_contract(TLArray(q0.I, ("left", "bond")),
                                (2,),
                                TLArray(q0.I, ("bond", "right")),
                                (1,))
    @test_throws MethodError save_tlarray(tempname() * ".h5", lazy)
end
