@testset "empty_tlarray" begin
    # ── rank-1 to rank-5 construction ────────────────────────────────────────
    @testset "rank-$QD construction" for QD in 1:5
        symm = (SU{2},)
        inds = ntuple(i -> TLIndex("l$i", i == 1 ? '+' : '-'), QD)
        q = empty_tlarray(symm, inds)

        @test isempty(Telum.stored_qlabels(q))
        @test isempty(q.RMTs)
        @test Telum.symm(q) == symm
        @test Telum.inds(q)          == inds
        @test all(isempty(Telum.spaces(q)[l]) for l in 1:QD)
    end

    # ── multiple symmetries ───────────────────────────────────────────────────
    @testset "multi-symmetry construction" begin
        symm = (U1, SU{2})
        inds = (TLIndex("a", '+'), TLIndex("b", '-'), TLIndex("c", '-'))
        q = empty_tlarray(symm, inds)

        @test Telum.symm(q) == symm
        @test Telum.inds(q) == inds
        @test length(Telum.spaces(q)) == 3
        @test all(isempty(Telum.spaces(q)[l]) for l in 1:3)
    end

    # ── element type keyword ──────────────────────────────────────────────────
    @testset "element type" begin
        symm = (SU{2},)
        inds = (TLIndex("a", '+'), TLIndex("b", '-'))
        qf64 = empty_tlarray(symm, inds; T=Float64)
        qc64 = empty_tlarray(symm, inds; T=ComplexF64)

        @test eltype(qf64.RMTs) <: Array{Float64, 3}
        @test eltype(qc64.RMTs) <: Array{ComplexF64, 3}
    end

    @testset "zero preserves metadata on TLArray" begin
        option = FermionSOptions(3, :U1, :SU2, :SU3)
        q0 = getLocalSpace(option)
        q = TLArray(q0.F, ("site1", "site2", "op"))
        qz = zero(q)

        # zero(q) should clear sector payloads without changing the tensor's leg
        # metadata, so it can still participate in compatible tensor operations.
        @test qz isa TLArray
        @test isempty(Telum.stored_qlabels(qz))
        @test isempty(qz.RMTs)
        @test symm(qz) == symm(q)
        @test Telum.inds(qz) == Telum.inds(q)
        @test Telum.spaces(qz) == Telum.spaces(q)
    end

    # ── show does not error ───────────────────────────────────────────────────
    @testset "show on empty TLArray" begin
        symm = (SU{2},)
        inds = (TLIndex("a", '+'), TLIndex("b", '-'))
        q = empty_tlarray(symm, inds)
        buf = IOBuffer()
        @test (show(buf, MIME"text/plain"(), q); true)
        @test occursin("empty", String(take!(buf)))
    end

    @testset "printmeta on TLArray" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        q = TLArray(q0.F, ("site1", "site2", "op"))

        meta = sprint(printmeta, q)

        @test occursin("3D TLArray", meta)
        @test occursin("site1", meta)
        @test !occursin('\n', meta)

        left = TLArray(q0.I, ("left", "bond"))
        right = TLArray(q0.I, ("bond", "right"))
        lazy = Telum._lazy_contract(left, (2,), right, (1,))
        @test !hasmethod(printmeta, Tuple{IO, typeof(lazy)})
        @test_throws MethodError printmeta(IOBuffer(), lazy)
    end

    @testset "permuted TLArray view display uses logical RMT dimensions" begin
        q = TLArray(getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing)).F,
                    ("left", "right", "op"))
        qp = permutedims(q, (3, 1, 2))
        shown = sprint(show, MIME"text/plain"(), qp)

        for sector in Telum.sector_slots(qp)
            Telum.is_sector_defined(qp, sector) || continue
            dims = Telum.sector_rmt_dim(qp, sector)
            expected = join(dims[1:ndims(qp)], "x")
            line = only(filter(l -> startswith(strip(l), "$(sector)."), split(shown, '\n')))
            @test occursin(expected, line)
        end
    end
end

@testset "TLArray array storage copy option" begin
    data = reshape(collect(1.0:6.0), 2, 3)
    copied = TLArray(data; itag=("copy_left", "copy_right"), copy=true)
    shared_default = TLArray(data; itag=("default_left", "default_right"))
    shared = TLArray(data; itag=("share_left", "share_right"), copy=false)

    @test copied.RMTs[1] !== data
    @test shared_default.RMTs[1] === data
    @test shared.RMTs[1] === data

    data[1, 1] = 99.0
    @test shared_default.RMTs[1][1, 1] == 99.0
    @test shared.RMTs[1][1, 1] == 99.0
    @test copied.RMTs[1][1, 1] == 1.0

    shared.RMTs[1][2, 2] = -7.0
    @test data[2, 2] == -7.0

    view = @view data[:, :]
    @test_throws ArgumentError TLArray(view; copy=false)
    @test TLArray(view).RMTs[1] !== data
end

@testset "permutation and conjugation defer non-W-matrix data" begin
    q = TLArray(getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing)).F,
                ("left", "right", "op"))
    old_inds = getfield(q, :inds)
    old_spaces = getfield(q, :spaces)
    old_qlabels = getfield(q, :qlabels)
    old_rmts = q.RMTs

    qp = permutedims(q, (3, 1, 2))
    @test getfield(qp, :inds) === old_inds
    @test getfield(qp, :spaces) === old_spaces
    @test getfield(qp, :qlabels) === old_qlabels
    @test qp.RMTs === old_rmts
    @test qp.wmatdata !== q.wmatdata
    @test qp.wmatinfo !== q.wmatinfo
    @test Telum.inds(qp) == (Telum.inds(q)[3], Telum.inds(q)[1], Telum.inds(q)[2])
    @test Telum.spaces(qp) == (Telum.spaces(q)[3], Telum.spaces(q)[1], Telum.spaces(q)[2])
    @test all(Telum.logical_qlabels(qp)[s] == (Telum.stored_qlabels(q)[s][3],
                                                Telum.stored_qlabels(q)[s][1],
                                                Telum.stored_qlabels(q)[s][2])
              for s in Telum.sector_slots(q))

    qc = conj(q)
    @test getfield(qc, :inds) === old_inds
    @test getfield(qc, :spaces) === old_spaces
    @test getfield(qc, :qlabels) === old_qlabels
    @test qc.RMTs === old_rmts
    @test qc.wmatdata !== q.wmatdata
    @test qc.wmatinfo !== q.wmatinfo
    @test Telum.inds(qc) == map(Telum.change_dir, Telum.inds(q))
    @test Telum.spaces(qc) == Telum.spaces(q)
    @test Telum.logical_qlabels(qc) == Telum.logical_qlabels(q)
end

@testset "AbstractTLArray properties expose stored fields" begin
    q = TLArray(getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing)).F,
                ("left", "right", "op"))
    qp = permutedims(q, (3, 1, 2))

    @test qp.inds === getfield(qp, :inds)
    @test qp.spaces === getfield(qp, :spaces)
    @test qp.qlabels === getfield(qp, :qlabels)
    @test qp.inds === Telum.stored_inds(q)
    @test qp.inds != Telum.inds(qp)
    @test qp.spaces === Telum.stored_spaces(q)
    @test qp.spaces != Telum.spaces(qp)
    @test qp.qlabels === Telum.stored_qlabels(q)
    @test qp.qlabels != Telum.logical_qlabels(qp)
    @test propertynames(qp) == fieldnames(typeof(qp))

    left = TLArray(getLocalSpace(SpinOptions(nothing, 1)).I, ("left", "bond"))
    right = TLArray(getLocalSpace(SpinOptions(nothing, 1)).I, ("bond", "right"))
    lazy = Telum._lazy_contract(left, (2,), right, (1,))
    wrapped = addSingleton(lazy, 1; itag="aux", dir='+')
    @test wrapped isa Telum.AddSingletonTLArray
    @test wrapped.inds === getfield(wrapped, :inds)
    @test hasproperty(wrapped, :inds)
    @test !hasproperty(wrapped, :wmatdata)
    @test !hasproperty(wrapped, :wmatinfo)
    @test :wmatdata ∉ propertynames(wrapped)
    @test :wmatinfo ∉ propertynames(wrapped)
    @test_throws ErrorException wrapped.wmatdata
    @test_throws ErrorException wrapped.wmatinfo
end

@testset "complete sector constructors" begin
    symmetries = (U1, SU{2})
    spaces = (
        [(((0,), (0,)), 1), (((1,), (1,)), 2)],
        [(((0,), (0,)), 3), (((1,), (1,)), 4)],
    )
    q = allsectors_tlarray(symmetries, spaces, ('+', '-');
                            itag=("left", "right"), plev=(1, 0), lock=(2, 0))

    @test Telum.inds(q) == (TLIndex("left", '+', 1, 2), TLIndex("right", '-', 0, 0))
    @test Telum.spaces(q) == spaces
    @test Telum.sector_count(q) == 2
    @test all(Telum.is_sector_defined(q, sector) && Telum.is_sector_zero(q, sector)
              for sector in Telum.sector_slots(q))
    @test all(size(Telum.sector_wmat(q, sector, 2)) == (1, 1)
              for sector in Telum.sector_slots(q))
    @test all(size(Telum.sector_rmt(q, sector)[1]) == (sector == 1 ? 1 : 2,
                                                         sector == 1 ? 3 : 4, 1, 1)
              for sector in Telum.sector_slots(q))

    multi_om_spaces = ntuple(_ -> [(((1,),), 1)], 4)
    q_multi_om = allsectors_tlarray((SU{2},), multi_om_spaces, ('+', '+', '-', '-'))
    @test Telum.sector_count(q_multi_om) == 1
    @test Telum.sector_wmat(q_multi_om, 1, 1) == Matrix{Float64}(I, 2, 2)
    @test size(Telum.sector_rmt(q_multi_om, 1)[1]) == (1, 1, 1, 1, 2)

    qr = random_allsectors_tlarray(symmetries, spaces, ('+', '-'); T=ComplexF64)
    @test all(Telum.is_sector_defined(qr, sector) && !Telum.is_sector_zero(qr, sector)
              for sector in Telum.sector_slots(qr))
    @test eltype(qr) == ComplexF64
end

@testset "random_similar preserves TLArray structure" begin
    q = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing)).F
    q_random = random_similar(q)

    @test q_random isa TLArray
    @test q_random !== q
    @test Telum.inds(q_random) == Telum.inds(q)
    @test Telum.spaces(q_random) == Telum.spaces(q)
    @test Telum.stored_qlabels(q_random) == Telum.stored_qlabels(q)
    @test q_random.wmatdata == q.wmatdata
    @test q_random.wmatinfo == q.wmatinfo
    @test q_random.wmatdata !== q.wmatdata
    @test q_random.wmatinfo !== q.wmatinfo
    @test q_random.iszero == q.iszero

    for sector in Telum.sector_slots(q)
        if Telum.is_sector_zero(q, sector)
            @test !Telum.is_sector_defined(q_random, sector)
        else
            @test Telum.is_sector_defined(q_random, sector)
            @test size(q_random.RMTs[sector]) == size(q.RMTs[sector])
            @test q_random.RMTs[sector] !== q.RMTs[sector]
        end
    end

    left = TLArray(getLocalSpace(SpinOptions(nothing, 1)).I, ("left", "bond"))
    right = TLArray(getLocalSpace(SpinOptions(nothing, 1)).I, ("bond", "right"))
    lazy = Telum._lazy_contract(left, (2,), right, (1,))
    @test_throws MethodError random_similar(lazy)
    @test_throws Base.CanonicalIndexError copy(lazy)
end

@testset "zero_qlabels" begin
    q_empty = empty_tlarray((SU{2}, SU{3}), (TLIndex('+'), TLIndex('-')))
    @test zero_qlabels(q_empty) == ((0,), (0, 0))
    @test zero_qlabels(symm(q_empty)) == ((0,), (0, 0))

    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    @test zero_qlabels(q0.I) == ((0,), (0,))
end

@testset "qlabeltype" begin
    q_empty = empty_tlarray((U1, SU{3}), (TLIndex('+'), TLIndex('-')))
    expected = Tuple{Tuple{Int}, NTuple{2, Int}}
    expected_ps = ProductSymm{Tuple{U1, SU{3}}}

    @test qlabeltype(q_empty) == expected
    @test qlabeltype(symm(q_empty)) == expected
    @test productsymm(q_empty) == expected_ps
    @test @inferred(symm(q_empty)) == (U1, SU{3})
    @test eltype(Telum.spaces(q_empty)[1]) == Tuple{expected, Int}

    q_multi = empty_tlarray((U1, SU{2}, SU{3}), (TLIndex('+'),))
    @test qlabeltype(q_multi) == Tuple{Tuple{Int}, Tuple{Int}, NTuple{2, Int}}
    @test productsymm(q_multi) == ProductSymm{Tuple{U1, SU{2}, SU{3}}}

    q_local = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing)).I
    @test eltype(Telum.stored_qlabels(q_local)) == NTuple{2, qlabeltype(q_local)}
    info = Telum.leginfo(q_local, 1)
    @test qlabeltype(info) == qlabeltype(q_local)
    @test eltype(info.splist) == Tuple{qlabeltype(q_local), Int}
end

@testset "getsub sector slicing" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    candidate = nothing
    # Pick a real local-space tensor with both a full-sector selection and a
    # sector whose multiplicity is large enough to test reordering/slicing.
    for base in values(q0)
        for legcand in 1:length(Telum.spaces(base))
            qcand = oplus([base, 2.0 * base, 3.0 * base], legcand)
            reorder_idx = findfirst(i -> Telum.spaces(qcand)[legcand][i][2] >= 3, eachindex(Telum.spaces(qcand)[legcand]))
            isnothing(reorder_idx) && continue

            full_idx = findfirst(i -> i != reorder_idx, eachindex(Telum.spaces(qcand)[legcand]))
            isnothing(full_idx) && continue

            candidate = (
                q = qcand,
                leg = legcand,
                sector_reorder = Telum.spaces(qcand)[legcand][reorder_idx][1],
                dim_reorder = Telum.spaces(qcand)[legcand][reorder_idx][2],
                sector_full = Telum.spaces(qcand)[legcand][full_idx][1],
                dim_full = Telum.spaces(qcand)[legcand][full_idx][2],
            )
            break
        end
        !isnothing(candidate) && break
    end

    @test !isnothing(candidate)

    q = candidate.q
    leg = candidate.leg
    sector_reorder = candidate.sector_reorder
    dim_reorder = candidate.dim_reorder
    sector_full = candidate.sector_full
    dim_full = candidate.dim_full

    row_sector(qs::TLArray, sector_index::Int) = Telum.logical_qlabels(qs)[sector_index][leg]
    rows_for_sector(qs::TLArray, sector) =
        [i for i in _test_defined_sector_indices(qs) if row_sector(qs, i) == sector]

    # Keep one sector intact and reverse two multiplicity rows in another; this
    # checks that getsub updates spaces and slices only the selected RMT axis.
    q_pred_pairs = Telum.getsub(q, leg,
                                  sector -> sector == sector_full ? Colon() :
                                            sector == sector_reorder ? (dim_reorder, 1) :
                                            nothing)

    # Expected metadata for the selected leg: the full sector keeps its original
    # multiplicity, while the reordered sector is truncated to the two requested rows.
    expected_spaces_leg = [
        (sector, sector == sector_full ? dim_full : 2)
        for (sector, _) in Telum.spaces(q)[leg]
        if sector == sector_full || sector == sector_reorder
    ]

    # Only the selected leg's space list changes; all other legs remain compatible
    # with the source tensor.
    @test Telum.spaces(q_pred_pairs)[leg] == expected_spaces_leg
    for other_leg in 1:length(Telum.spaces(q))
        other_leg == leg && continue
        @test Telum.spaces(q_pred_pairs)[other_leg] == Telum.spaces(q)[other_leg]
    end
    @test Set(row_sector(q_pred_pairs, i) for i in _test_defined_sector_indices(q_pred_pairs)) ==
          Set([sector_full, sector_reorder])

    # Colon() keeps every RMT entry for the full sector unchanged.
    full_rows = rows_for_sector(q_pred_pairs, sector_full)
    orig_full_rows = rows_for_sector(q, sector_full)
    @test length(full_rows) == length(orig_full_rows)
    for (full_idx, orig_full_idx) in zip(full_rows, orig_full_rows)
        @test _test_sector_rmt(q_pred_pairs, full_idx) == _test_sector_rmt(q, orig_full_idx)
    end

    # Tuple selectors are applied in the requested order to the selected RMT axis.
    reorder_rows = rows_for_sector(q_pred_pairs, sector_reorder)
    orig_reorder_rows = rows_for_sector(q, sector_reorder)
    @test length(reorder_rows) == length(orig_reorder_rows)
    reorder_inds = [dim_reorder, 1]
    for (reorder_idx, orig_reorder_idx) in zip(reorder_rows, orig_reorder_rows)
        orig_rmt = _test_sector_rmt(q, orig_reorder_idx)
        reorder_selector = ntuple(d -> d == leg ? reorder_inds : Colon(), ndims(orig_rmt))
        @test _test_sector_rmt(q_pred_pairs, reorder_idx) == orig_rmt[reorder_selector...]
    end

    # Range selectors are normalized to the same explicit RMT-axis selection.
    q_range = Telum.getsub(q, leg, sector -> sector == sector_reorder ? (1:2) : nothing)
    @test Telum.spaces(q_range)[leg] == [(sector_reorder, 2)]
    range_rows = rows_for_sector(q_range, sector_reorder)
    @test length(range_rows) == length(orig_reorder_rows)
    for (range_idx, orig_reorder_idx) in zip(range_rows, orig_reorder_rows)
        orig_rmt = _test_sector_rmt(q, orig_reorder_idx)
        range_selector = ntuple(d -> d == leg ? [1, 2] : Colon(), ndims(orig_rmt))
        @test _test_sector_rmt(q_range, range_idx) == orig_rmt[range_selector...]
    end

    # Negative indices follow Julia indexing from the end of the multiplicity axis.
    q_negative = Telum.getsub(q, leg, sector -> sector == sector_reorder ? -1 : nothing)
    negative_rows = rows_for_sector(q_negative, sector_reorder)
    @test Telum.spaces(q_negative)[leg] == [(sector_reorder, 1)]
    @test length(negative_rows) == length(orig_reorder_rows)
    for (negative_idx, orig_reorder_idx) in zip(negative_rows, orig_reorder_rows)
        orig_rmt = _test_sector_rmt(q, orig_reorder_idx)
        negative_selector = ntuple(d -> d == leg ? [dim_reorder] : Colon(), ndims(orig_rmt))
        @test _test_sector_rmt(q_negative, negative_idx) == orig_rmt[negative_selector...]
    end

    # Mixed negative and positive selectors should preserve the explicit order.
    q_mixed = Telum.getsub(q, leg, sector -> sector == sector_reorder ? [-1, 1] : nothing)
    mixed_rows = rows_for_sector(q_mixed, sector_reorder)
    @test Telum.spaces(q_mixed)[leg] == [(sector_reorder, 2)]
    @test length(mixed_rows) == length(orig_reorder_rows)
    for (mixed_idx, orig_reorder_idx) in zip(mixed_rows, orig_reorder_rows)
        orig_rmt = _test_sector_rmt(q, orig_reorder_idx)
        mixed_selector = ntuple(d -> d == leg ? [dim_reorder, 1] : Colon(), ndims(orig_rmt))
        @test _test_sector_rmt(q_mixed, mixed_idx) == orig_rmt[mixed_selector...]
    end

    # Returning nothing for every sector drops all payloads and clears only the
    # selected leg's space list.
    q_empty = Telum.getsub(q, leg, _ -> nothing)
    @test isempty(_test_defined_sector_indices(q_empty))
    @test isempty(Telum.spaces(q_empty)[leg])
    for other_leg in 1:length(Telum.spaces(q))
        other_leg == leg && continue
        @test Telum.spaces(q_empty)[other_leg] == Telum.spaces(q)[other_leg]
    end

    # Invalid selectors fail early: bad legs, duplicates, empty selections, zero,
    # out-of-bounds values, unsupported return types, and slicing with preserved spaces.
    @test_throws ArgumentError Telum.getsub(q, 0, _ -> Colon())
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? [1, 1] : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? Int[] : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? 0 : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? (dim_reorder + 1) : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? "bad" : nothing)
    @test_throws ArgumentError Telum.getsub(q, leg, sector -> sector == sector_reorder ? 1 : nothing; preserve_space=true)
end

@testset "getsub sector predicate" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    candidate = nothing
    # Use an oplus tensor so at least one leg has multiple sectors for the
    # predicate to filter while the untouched legs keep their original spaces.
    for base in values(q0)
        for legcand in 1:length(Telum.spaces(base))
            qcand = oplus([base, 2.0 * base, 3.0 * base], legcand)
            length(Telum.spaces(qcand)[legcand]) >= 2 || continue
            candidate = (q = qcand, leg = legcand, target_sector = Telum.spaces(qcand)[legcand][1][1])
            break
        end
        !isnothing(candidate) && break
    end

    @test !isnothing(candidate)

    q = candidate.q
    leg = candidate.leg
    target_sector = candidate.target_sector

    row_sector(qs::TLArray, sector_index::Int) = Telum.logical_qlabels(qs)[sector_index][leg]
    expected_rows = [i for i in _test_defined_sector_indices(q) if row_sector(q, i) == target_sector]
    expected_leg_spaces = [entry for entry in Telum.spaces(q)[leg] if entry[1] == target_sector]

    q_component = Telum.getsub(q, leg, sector -> sector[1] == target_sector[1] ? Colon() : nothing)
    expected_component_rows = [i for i in _test_defined_sector_indices(q) if row_sector(q, i)[1] == target_sector[1]]
    expected_component_spaces = [entry for entry in Telum.spaces(q)[leg] if entry[1][1] == target_sector[1]]
    _test_tlarrays_same_sector_payloads(q_component, TLArray(q, expected_component_rows))
    @test Telum.spaces(q_component)[leg] == expected_component_spaces

    # preserve_space keeps the original leg metadata even when only a subset of
    # sector payloads remains defined.
    q_preserved = Telum.getsub(q, leg, sector -> sector == target_sector ? Colon() : nothing; preserve_space=true)
    _test_tlarrays_same_sector_payloads(q_preserved, TLArray(q, expected_rows))
    @test Telum.spaces(q_preserved) == Telum.spaces(q)

    q_none_preserved = Telum.getsub(q, leg, _ -> nothing; preserve_space=true)
    @test isempty(_test_defined_sector_indices(q_none_preserved))
    @test Telum.spaces(q_none_preserved) == Telum.spaces(q)

    @test_throws ArgumentError Telum.getsub(q, leg, _ -> false)
    @test_throws ArgumentError Telum.getsub(q, leg, _ -> true)
end

@testset "getsub multi-leg sector predicate" begin
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    base = getLocalSpace(option, ("sel,left", "sel,right", "op")).I
    q = oplus([base, 2.0 * base, 3.0 * base], (1, 2))
    legs = (1, 2)

    @test length(Telum.spaces(q)[1]) >= 2
    @test length(Telum.spaces(q)[2]) >= 2
    @test !isempty(_test_defined_sector_indices(q))

    row_sector_at(sector_index, leg) = Telum.logical_qlabels(q)[sector_index][leg]
    first_sector = first(_test_defined_sector_indices(q))
    allowed = Set{Any}([row_sector_at(first_sector, 1), row_sector_at(first_sector, 2)])
    pred = sector -> sector in allowed ? Colon() : nothing

    # Filtering both tagged legs must keep only sectors whose labels pass on
    # every selected leg, while non-selected legs retain their original spaces.
    expected_rows = [
        i for i in _test_defined_sector_indices(q)
        if all(row_sector_at(i, leg) in allowed for leg in legs)
    ]
    expected_spaces_1 = [entry for entry in Telum.spaces(q)[1] if entry[1] in allowed]
    expected_spaces_2 = [entry for entry in Telum.spaces(q)[2] if entry[1] in allowed]

    q_multi = Telum.getsub(q, legs, pred)
    _test_tlarrays_same_sector_payloads(q_multi, TLArray(q, expected_rows))
    @test Telum.spaces(q_multi)[1] == expected_spaces_1
    @test Telum.spaces(q_multi)[2] == expected_spaces_2
    for other_leg in 1:length(Telum.spaces(q))
        other_leg in legs && continue
        @test Telum.spaces(q_multi)[other_leg] == Telum.spaces(q)[other_leg]
    end

    pred_slice = sector -> sector in allowed ? 1 : nothing
    q_multi_sliced = Telum.getsub(q, legs, pred_slice)
    @test Telum.spaces(q_multi_sliced)[1] == [(entry[1], 1) for entry in Telum.spaces(q)[1] if entry[1] in allowed]
    @test Telum.spaces(q_multi_sliced)[2] == [(entry[1], 1) for entry in Telum.spaces(q)[2] if entry[1] in allowed]
    for other_leg in 1:length(Telum.spaces(q))
        other_leg in legs && continue
        @test Telum.spaces(q_multi_sliced)[other_leg] == Telum.spaces(q)[other_leg]
    end
    @test length(_test_defined_sector_indices(q_multi_sliced)) == length(expected_rows)
    for (sliced_idx, orig_idx) in zip(_test_defined_sector_indices(q_multi_sliced), expected_rows)
        orig_rmt = _test_sector_rmt(q, orig_idx)
        slice_selector = ntuple(d -> d in legs ? [1] : Colon(), ndims(orig_rmt))
        @test _test_sector_rmt(q_multi_sliced, sliced_idx) == orig_rmt[slice_selector...]
    end

    q_multi_preserved = Telum.getsub(q, legs, pred; preserve_space=true)
    _test_tlarrays_same_sector_payloads(q_multi_preserved, TLArray(q, expected_rows))
    @test Telum.spaces(q_multi_preserved) == Telum.spaces(q)

    q_multi_kw = Telum.getsub(q, pred; itag="sel")
    _test_tlarrays_same_sector_payloads(q_multi_kw, q_multi)
    @test Telum.spaces(q_multi_kw) == Telum.spaces(q_multi)

    @test_throws ArgumentError Telum.getsub(q, Int[], pred)
    @test_throws ArgumentError Telum.getsub(q, (1, 1), pred)
    @test_throws ArgumentError Telum.getsub(q, (0, 1), pred)
    @test_throws ArgumentError Telum.getsub(q, legs, pred_slice; preserve_space=true)
    @test_throws ArgumentError Telum.getsub(q, pred; itag="missing")
end

@testset "getvac" begin
    @testset "single trivial sector with default tags" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        vac = getvac(q0.I)

        # The vacuum tensor is a rank-2 identity on the trivial sector, independent
        # of the nontrivial sectors present in the input local space.
        @test symm(vac) == symm(q0.I)
        @test length(_test_defined_sector_indices(vac)) == 1
        @test Telum.inds(vac) == (TLIndex("", '+'), TLIndex("", '-'))
        @test length(Telum.spaces(vac)[1]) == 1
        @test length(Telum.spaces(vac)[2]) == 1

        trivial = zero_qlabels(vac)
        @test Telum.spaces(vac)[1][1] == (trivial, 1)
        @test Telum.spaces(vac)[2][1] == (trivial, 1)
        @test Array(to_sparse_array(vac)) == ones(1, 1)
    end

    @testset "optional tags are applied" begin
        option = FermionSOptions(1, :U1, :SU2, nothing)
        q0 = getLocalSpace(option)
        vac = getvac(q0.F, ("vin", "vout"))

        @test Telum.inds(vac)[1] == TLIndex("vin", '+')
        @test Telum.inds(vac)[2] == TLIndex("vout", '-')
    end
end

@testset "addSingleton" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("ain", "aout", "op"))
    q = q0.F
    q_rank2 = TLArray(q0.I, ("lin", "lout"))

    q_default = addSingleton(q, 2)
    @test Telum.inds(q_default)[1] == Telum.inds(q)[1]
    @test Telum.inds(q_default)[2] == TLIndex("", '+')
    @test Telum.inds(q_default)[3] == Telum.inds(q)[2]
    @test Telum.inds(q_default)[4] == Telum.inds(q)[3]

    trivial = zero_qlabels(q)
    # Added singleton legs always carry the trivial sector with multiplicity one.
    @test Telum.spaces(q_default)[2] == [(trivial, 1)]

    q_append_default = addSingleton(q)
    @test Telum.inds(q_append_default)[1:3] == Telum.inds(q)
    @test Telum.inds(q_append_default)[4] == TLIndex("", '+')
    @test Telum.spaces(q_append_default)[4] == [(trivial, 1)]

    q_append_two = addSingleton(q; nlegs=2,
                                itag=("tail_left", "tail_right"),
                                plev=(1, 2),
                                lock=(0, 1),
                                dir=('+', '-'))
    @test Telum.inds(q_append_two)[1:3] == Telum.inds(q)
    @test Telum.inds(q_append_two)[4] == TLIndex("tail_left", '+', 1, 0)
    @test Telum.inds(q_append_two)[5] == TLIndex("tail_right", '-', 2, 1)
    @test Telum.spaces(q_append_two)[4] == [(trivial, 1)]
    @test Telum.spaces(q_append_two)[5] == [(trivial, 1)]
    @test Array(to_sparse_array(q_append_two)) ==
          Array(to_sparse_array(addSingleton(q, (4, 5);
                                             itag=("tail_left", "tail_right"),
                                             plev=(1, 2),
                                             lock=(0, 1),
                                             dir=('+', '-'))))

    @test_throws ArgumentError addSingleton(q; nlegs=0)

    q_added = addSingleton(q, (1, 4);
                           itag=("left_aux", "right_aux"),
                           plev=(2, 3),
                           lock=(0, 1),
                           dir=('-', '+'))

    @test Telum.inds(q_added)[1] == TLIndex("left_aux", '-', 2, 0)
    @test Telum.inds(q_added)[2] == Telum.inds(q)[1]
    @test Telum.inds(q_added)[3] == Telum.inds(q)[2]
    @test Telum.inds(q_added)[4] == TLIndex("right_aux", '+', 3, 1)
    @test Telum.inds(q_added)[5] == Telum.inds(q)[3]
    @test Telum.spaces(q_added)[1] == [(trivial, 1)]
    @test Telum.spaces(q_added)[4] == [(trivial, 1)]

    arr_ref = _dense_addSingleton_ref(Array(to_sparse_array(q)), (1, 4))
    arr_added = Array(to_sparse_array(q_added))
    # Dense comparison verifies that inserting metadata-only singleton legs did
    # not reorder existing physical axes.
    @test size(arr_added) == size(arr_ref)
    @test norm(arr_added - arr_ref) < 1e-10

    q_rank2_added = addSingleton(q_rank2, 2)
    arr_rank2_ref = _dense_addSingleton_ref(Array(to_sparse_array(q_rank2)), 2)
    arr_rank2_added = Array(to_sparse_array(q_rank2_added))
    @test size(arr_rank2_added) == size(arr_rank2_ref)
    @test norm(arr_rank2_added - arr_rank2_ref) < 1e-10
end

@testset "deleteSingleton" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("ain", "aout", "op"))
    q = q0.F
    q_rank2 = TLArray(q0.I, ("lin", "lout"))

    q_one = addSingleton(q, 2; itag="aux", plev=3, dir='-')
    q_two = addSingleton(q, (1, 4);
                         itag=("left_aux", "right_aux"),
                         plev=(2, 5),
                         dir=('-', '+'))
    q_rank2_added = addSingleton(q_rank2, 2; itag="mid_aux", dir='+')

    # Deleting singleton legs should restore both metadata and sparse values for
    # explicit leg selection, keyword selection, and default singletons.
    q_deleted_leg = deleteSingleton(q_one, 2)
    @test Telum.inds(q_deleted_leg) == Telum.inds(q)
    @test Telum.spaces(q_deleted_leg) == Telum.spaces(q)
    @test Array(to_sparse_array(q_deleted_leg)) == Array(to_sparse_array(q))

    q_deleted_legs = deleteSingleton(q_two, (1, 4))
    @test Telum.inds(q_deleted_legs) == Telum.inds(q)
    @test Telum.spaces(q_deleted_legs) == Telum.spaces(q)
    @test Array(to_sparse_array(q_deleted_legs)) == Array(to_sparse_array(q))

    q_deleted_kw_tag = deleteSingleton(q_two; itag="left_aux")
    @test length(Telum.inds(q_deleted_kw_tag)) == 4
    @test Telum.inds(q_deleted_kw_tag)[1] == Telum.inds(q)[1]
    @test Telum.inds(q_deleted_kw_tag)[2] == Telum.inds(q)[2]
    @test Telum.inds(q_deleted_kw_tag)[3] == TLIndex("right_aux", '+', 5, 0)
    @test Telum.inds(q_deleted_kw_tag)[4] == Telum.inds(q)[3]
    @test Array(to_sparse_array(q_deleted_kw_tag)) == Array(to_sparse_array(addSingleton(q, 3; itag="right_aux", plev=5, dir='+')))

    q_deleted_kw_plev = deleteSingleton(q_two; plev=5)
    @test length(Telum.inds(q_deleted_kw_plev)) == 4
    @test Telum.inds(q_deleted_kw_plev)[1] == TLIndex("left_aux", '-', 2, 0)
    @test Telum.inds(q_deleted_kw_plev)[2] == Telum.inds(q)[1]
    @test Telum.inds(q_deleted_kw_plev)[3] == Telum.inds(q)[2]
    @test Telum.inds(q_deleted_kw_plev)[4] == Telum.inds(q)[3]

    q_rank2_roundtrip = deleteSingleton(q_rank2_added)
    @test Telum.inds(q_rank2_roundtrip) == Telum.inds(q_rank2)
    @test Telum.spaces(q_rank2_roundtrip) == Telum.spaces(q_rank2)
    @test Array(to_sparse_array(q_rank2_roundtrip)) == Array(to_sparse_array(q_rank2))

    @test_throws ArgumentError deleteSingleton(q, 1)
    @test_throws ArgumentError deleteSingleton(q_two, (1, 2))
    @test_throws ArgumentError deleteSingleton(q_two, Int[])
    @test_throws ArgumentError deleteSingleton(q_two, (1, 1))
    @test_throws ArgumentError deleteSingleton(q_two, 0)

    @test_logs (:warn, r"no singleton legs found") deleteSingleton(q)
    @test_logs (:warn, r"no singleton legs matched") deleteSingleton(q_two; itag="ain")
end

@testset "lazy add/deleteSingleton" begin
    q0 = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q0.I, ("left", "bond"))
    right = TLArray(q0.Sz, ("bond", "right"))
    eager = contract(left, (2,), right, (1,))

    lazy_add_source = Telum._lazy_contract(left, (2,), right, (1,))
    add_lazy = addSingleton(lazy_add_source, 1; itag="aux", dir='+')
    @test add_lazy isa Telum.AddSingletonTLArray
    @test !Telum.is_sector_defined(lazy_add_source, 1)
    @test Telum.sector_rmt_dim(add_lazy, 1) == (1, Telum.sector_rmt_dim(lazy_add_source, 1)...)
    @test_throws ArgumentError Telum.sector_rmt(add_lazy, 1)
    Telum.compute_sectors(add_lazy, [1])
    @test Telum.is_sector_defined(lazy_add_source, 1)
    rmt_add, add_scale = Telum.sector_rmt(add_lazy, 1)
    @test add_scale == 1
    @test rmt_add == reshape(Telum.sector_rmt(lazy_add_source, 1)[1], size(rmt_add))
    sum_add = add_lazy + add_lazy
    @test sum_add isa TLArray
    @test Telum.sector_rmt(sum_add, 1)[1] == 2 .* rmt_add

    add_locked = lock(add_lazy, 1)
    @test add_locked isa Telum.AddSingletonTLArray
    @test add_locked.arr === add_lazy.arr
    @test Telum.inds(add_locked)[1].lock == Telum.inds(add_lazy)[1].lock + 1
    add_primed = prime(add_lazy, 1; inc=2)
    @test add_primed isa Telum.AddSingletonTLArray
    @test add_primed.arr === add_lazy.arr
    @test Telum.inds(add_primed)[1].plev == Telum.inds(add_lazy)[1].plev + 2
    add_tagged = additag(add_lazy, 1, "marked")
    @test add_tagged isa Telum.AddSingletonTLArray
    @test add_tagged.arr === add_lazy.arr
    @test occursin("marked", String(Telum.inds(add_tagged)[1].itag))

    aux = TLArray((), [((),)], Float64[], [Telum._empty_wmat_info(Val(0))],
                  [reshape([1.0], 1)], (TLIndex("aux", '-'),), ([((), 1)],))
    contracted_add = contract(addSingleton(contract(left, (2,), right, (1,)),
                                          1; itag="aux", dir='+'),
                              (1,), aux, (1,))
    @test Telum.sector_rmt(contracted_add, 1)[1] == Telum.sector_rmt(eager, 1)[1]

    lazy_delete_source = Telum._lazy_contract(left, (2,), right, (1,))
    add_for_delete = addSingleton(lazy_delete_source, 1; itag="aux", dir='+')
    delete_lazy = deleteSingleton(add_for_delete, 1)
    @test delete_lazy isa Telum.DeleteSingletonTLArray
    @test !Telum.is_sector_defined(lazy_delete_source, 1)
    @test Telum.sector_rmt_dim(delete_lazy, 1) == Telum.sector_rmt_dim(lazy_delete_source, 1)
    Telum.compute_sectors(delete_lazy, [1])
    @test Telum.is_sector_defined(lazy_delete_source, 1)
    @test Telum.sector_rmt(delete_lazy, 1)[1] == Telum.sector_rmt(eager, 1)[1]
    sum_delete = delete_lazy + delete_lazy
    @test sum_delete isa TLArray
    @test Telum.sector_rmt(sum_delete, 1)[1] == 2 .* Telum.sector_rmt(eager, 1)[1]

    delete_lockp = lockp(delete_lazy, 1)
    @test delete_lockp isa Telum.DeleteSingletonTLArray
    @test delete_lockp.arr === delete_lazy.arr
    @test Telum.inds(delete_lockp)[1].lock == -1
    delete_unlocked = unlock(delete_lockp, 1)
    @test delete_unlocked isa Telum.DeleteSingletonTLArray
    @test delete_unlocked.arr === delete_lazy.arr
    @test Telum.inds(delete_unlocked)[1].lock == 0
    delete_setprime = setprime(delete_lazy, 1, 3)
    @test delete_setprime isa Telum.DeleteSingletonTLArray
    @test delete_setprime.arr === delete_lazy.arr
    @test Telum.inds(delete_setprime)[1].plev == 3
    delete_noprime = noprime(delete_setprime, 1)
    @test delete_noprime isa Telum.DeleteSingletonTLArray
    @test Telum.inds(delete_noprime)[1].plev == 0
    delete_retagged = replaceitag(additag(delete_lazy, 1, "old"), 1, "old"=>"new")
    @test delete_retagged isa Telum.DeleteSingletonTLArray
    @test occursin("new", String(Telum.inds(delete_retagged)[1].itag))
    delete_cleared = removeitag(delete_retagged, 1, "new")
    @test delete_cleared isa Telum.DeleteSingletonTLArray
    @test !occursin("new", String(Telum.inds(delete_cleared)[1].itag))
    delete_settag = setitag(delete_lazy, 1, "only")
    @test delete_settag isa Telum.DeleteSingletonTLArray
    @test String(Telum.inds(delete_settag)[1].itag) == "only"

    lazy_view_source = Telum._lazy_contract(left, (2,), right, (1,))
    view = 2 * permutedims(addSingleton(lazy_view_source, 3; itag="aux", dir='+'), (2, 1, 3))
    delete_view = deleteSingleton(view, 3)
    @test delete_view isa Telum.DeleteSingletonTLArray
    @test delete_view.scale == 2
    view_ref = 2 * permutedims(eager, (2, 1))
    Telum.compute_sectors(delete_view, [1])
    delete_view_rmt, delete_view_scale = Telum.sector_rmt(delete_view, 1)
    view_ref_rmt, view_ref_scale = Telum.sector_rmt(view_ref, 1)
    @test delete_view_scale .* delete_view_rmt ≈ view_ref_scale .* view_ref_rmt
end

@testset "@lazy composes structural operations" begin
    q0 = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q0.I, ("left", "bond"))
    right = TLArray(q0.Sz, ("bond", "right"))
    eager = deleteSingleton(
        getsub(addSingleton(left * right, 1; itag="aux", dir='+'), 1, _ -> Colon()), 1)

    result = @lazy begin
        q = left * right
        q = addSingleton(q, 1; itag="aux", dir='+')
        q = getsub(q, 1, _ -> Colon())
        deleteSingleton(q, 1)
    end
    @test result isa TLArray
    @test _test_sector_rmt(result, 1) ≈ _test_sector_rmt(eager, 1)

    result_reordered = @lazy begin
        q = left * right
        q = getsub(q, 1, _ -> Colon())
        q = addSingleton(q, 1; itag="aux", dir='+')
        deleteSingleton(q, 1)
    end
    @test result_reordered isa TLArray
    @test _test_sector_rmt(result_reordered, 1) ≈ _test_sector_rmt(eager, 1)
end

@testset "deleteSingleton densifies singleton diagonal axis" begin
    symm = (U1,)
    trivial = ((0,),)
    qlabels = [(trivial, trivial)]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0))]
    spaces = ([(trivial, 1)], [(trivial, 1)])
    qdiag = TLArray(symm, qlabels, wmatdata, wmatinfo,
                    [DiagRMT([2.0], Val(3), (1, 2))],
                    (TLIndex("x", '+'), TLIndex("y", '-')), spaces)

    deleted = deleteSingleton(qdiag, 1)
    @test deleted.RMTs[1] isa Array{Float64, 2}
    @test deleted.RMTs[1] == reshape([2.0], 1, 1)
end

@testset "TLArray tensor product" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    q1 = TLArray(q0.I, ("l1_in", "l1_out"))
    q2 = TLArray(q0.F, ("l2_in", "l2_out", "l2_op"))

    q12 = Telum.:⊗(q1, q2)
    q12_kron = kron(q1, q2)

    @test symm(q12) == symm(q1) == symm(q2)
    @test Telum.inds(q12) == (Telum.inds(q1)..., Telum.inds(q2)...)
    @test Telum.spaces(q12) == (Telum.spaces(q1)..., Telum.spaces(q2)...)
    @test Telum.inds(q12_kron) == Telum.inds(q12)
    @test Telum.spaces(q12_kron) == Telum.spaces(q12)

    arr_ref = _dense_tensor_product_ref(Array(to_sparse_array(q1)), Array(to_sparse_array(q2)))
    arr_q12 = Array(to_sparse_array(q12))
    arr_q12_kron = Array(to_sparse_array(q12_kron))
    # Compare against a dense tensor-product reference, and check kron is the
    # same public operation as the unicode tensor product.
    @test size(arr_q12) == size(arr_ref)
    @test norm(arr_q12 - arr_ref) < 1e-10
    @test norm(arr_q12_kron - arr_q12) < 1e-10
end

# Helper: build a rank-4 test TLArray from getIdentity with 3 input legs.
#
#   leg 1 ('+', "l1"), leg 2 ('+', "l2"), leg 3 ('+', "l3"),
#   leg 4 ('-', "fused")
# ─────────────────────────────────────────────────────────────────────────────
function _make_test_tlarray_rank4()
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0  = getLocalSpace(option)
    qi1 = TLArray(q0.I, ("b1a", "b1b"))
    qi2 = TLArray(q0.I, ("b2a", "b2b"))
    qi3 = TLArray(q0.I, ("b3a", "b3b"))
    a4  = getIdentity((qi1, 2), (qi2, 2), (qi3, 2); itag="fused")
    return TLArray(a4, ("l1", "l2", "l3", "fused"))
end


@testset "scalar add/subtract on rank-2 TLArray" begin
    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option)
    q = TLArray(q0.I, ("left", "right"))

    # Scalar arithmetic is defined as adding/subtracting a scaled identity on the
    # same rank-2 space, not as touching every stored sector directly.
    idq = getIdentity(q, 2; itag=Telum.inds(q)[2].itag)
    idq = TLArray(idq, Telum.inds(q))

    @test norm((q + 2.5) - (q + 2.5 * idq)) < 1e-10
    @test norm((q - 2.5) - (q - 2.5 * idq)) < 1e-10
    @test norm((2.5 + q) - (q + 2.5 * idq)) < 1e-10
    @test norm((2.5 - q) - (2.5 * idq - q)) < 1e-10

    q_complex_left = (1.0 + 2.0im) * q
    q_complex_right = q * (1.0 + 2.0im)
    arr_q = Array(to_sparse_array(q))
    @test Array(to_sparse_array(q_complex_left)) ≈ (1.0 + 2.0im) .* arr_q
    @test Array(to_sparse_array(q_complex_right)) ≈ (1.0 + 2.0im) .* arr_q
    shown_complex = sprint(show, MIME"text/plain"(), q_complex_left)
    @test occursin("im", shown_complex)

    q_bad_rank = _make_test_tlarray_rank4()
    @test_throws AssertionError q_bad_rank + 1.0

    q_bad_dirs = getIdentity(q, 1, 2; itag="fused")
    @test_throws AssertionError q_bad_dirs + 1.0
end


# ─────────────────────────────────────────────────────────────────────────────
# Tests for arbitrary-rank (rank ≥ 4) TLArray
# ─────────────────────────────────────────────────────────────────────────────

@testset "arbitrary rank TLArray" begin
    q4 = _make_test_tlarray_rank4()

    # ── basic structure ───────────────────────────────────────────────────────
    @testset "rank-4 structure" begin
        @test length(Telum.inds(q4))  == 4
        @test length(Telum.spaces(q4)) == 4
        @test Telum.inds(q4)[1].dir   == '+'
        @test Telum.inds(q4)[2].dir   == '+'
        @test Telum.inds(q4)[3].dir   == '+'
        @test Telum.inds(q4)[4].dir   == '-'
        @test Telum.inds(q4)[1].itags == "l1"
        @test Telum.inds(q4)[4].itags == "fused"
        @test !isempty(_test_defined_sector_indices(q4))
        @test all(!isempty(Telum.spaces(q4)[l]) for l in 1:4)
    end

    @testset "svd keyword leg selection" begin
        ref = svd(q4, (1, 2, 3), "kwL", "kwR")
        kw = svd(q4, "kwL", "kwR"; dir='+')
        U_ref, S_ref, Vd_ref = ref.U, ref.S, ref.Vd
        U_kw, S_kw, Vd_kw = kw.U, kw.S, kw.Vd

        # Keyword selection by direction should choose the same left legs as the
        # explicit tuple and therefore produce identical factor metadata/values.
        @test Telum.inds(U_kw) == Telum.inds(U_ref)
        @test Telum.inds(S_kw) == Telum.inds(S_ref)
        @test Telum.inds(Vd_kw) == Telum.inds(Vd_ref)

        @test Telum.spaces(U_kw) == Telum.spaces(U_ref)
        @test Telum.spaces(S_kw) == Telum.spaces(S_ref)
        @test Telum.spaces(Vd_kw) == Telum.spaces(Vd_ref)

        @test norm(U_kw - U_ref) < 1e-12
        @test norm(S_kw - S_ref) < 1e-12
        @test norm(Vd_kw - Vd_ref) < 1e-12

        @test_throws ArgumentError svd(q4; itag="missing")
        @test_throws ArgumentError svd(q4; lock=0)
    end

    # ── scalar multiplication ─────────────────────────────────────────────────
    @testset "scalar multiplication on rank-4" begin
        q_scaled = 2.5 * q4
        arr_orig   = Array(to_sparse_array(q4))
        arr_scaled = Array(to_sparse_array(q_scaled))
        @test norm(arr_scaled - 2.5 .* arr_orig) < 1e-10
    end

    @testset "embedded view state preserves storage" begin
        q_perm = permutedims(q4, (4, 1, 2, 3))
        @test q_perm isa TLArray
        @test q_perm.perm == (4, 1, 2, 3)

        q_complex = (1.0 + 2.0im) * q_perm
        @test eltype(q_complex) === ComplexF64
        @test eltype(typeof(q_complex).parameters[end]) === eltype(typeof(q4).parameters[end])
        @test q_complex.scale == 1.0 + 2.0im
        @test q_complex.perm == q_perm.perm

        copied = copy(q_complex)
        @test copied isa TLArray
        @test copied.perm == q_complex.perm
        @test copied.scale == q_complex.scale
        @test copied.conj == q_complex.conj
        @test copied.RMTs !== q_complex.RMTs

        reference = (1.0 + 2.0im) .* permutedims(Array(to_sparse_array(q4, ComplexF64)), (4, 1, 2, 3))
        @test norm(Array(to_sparse_array(q_complex, ComplexF64)) - reference) < 1e-10

        sub = getsub(q_perm, 1, _ -> Colon(); preserve_space=true)
        @test sub.perm == q_perm.perm
        @test sub.scale == q_perm.scale
        @test sub.conj == q_perm.conj
        @test norm(Array(to_sparse_array(sub)) - Array(to_sparse_array(q_perm))) < 1e-10

        added = addSingleton(q_perm, 2; dir='+')
        @test added.perm == (4, 5, 1, 2, 3)
        deleted = deleteSingleton(added, 2)
        @test deleted.perm == q_perm.perm
        @test norm(Array(to_sparse_array(deleted)) - Array(to_sparse_array(q_perm))) < 1e-10
    end

    # ── addition: q4 + q4 ≈ 2*q4 ─────────────────────────────────────────────
    @testset "addition on rank-4: q + q ≈ 2*q" begin
        q_sum    = q4 + q4
        q_double = 2.0 * q4
        arr_sum    = Array(to_sparse_array(q_sum))
        arr_double = Array(to_sparse_array(q_double))
        @test norm(arr_sum - arr_double) < 1e-10
    end

    # ── conj: conj(conj(q)) ≈ q ───────────────────────────────────────────────
    @testset "double-conj roundtrip on rank-4" begin
        qc  = conj(q4)
        # all leg directions must flip
        @test all(Telum.inds(qc)[i].dir != Telum.inds(q4)[i].dir for i in 1:4)
        qcc = conj(qc)
        # directions restored
        @test all(Telum.inds(qcc)[i].dir == Telum.inds(q4)[i].dir for i in 1:4)
        # tensor values preserved
        arr_orig = Array(to_sparse_array(q4))
        arr_cc   = Array(to_sparse_array(qcc))
        @test norm(arr_orig - arr_cc) < 1e-10
    end

    @testset "show on rank-4 TLArray" begin
        buf = IOBuffer()
        @test (show(buf, MIME"text/plain"(), q4); true)
        out = String(take!(buf))
        @test occursin("4D TLArray", out)
    end
end
@testset "unsymmetric element indexing preserves tensor legs" begin
    data = reshape(collect(1.0:24.0), 2, 3, 4)
    q = TLArray(data, (TLIndex("a", '+'), TLIndex("b", '-'), TLIndex("c", '+')))

    @test q[1, 2, 3] == data[1, 2, 3]
    sliced = q[2:2, 2, 3]
    @test sliced isa TLArray
    @test size(sliced) == (1,)
    @test inds(sliced) == (inds(q)[1],)

    sliced2 = q[:, 3, :]
    @test sliced2 isa TLArray
    @test Array(sliced2) == data[:, 3, :]
    @test inds(sliced2) == (inds(q)[1], inds(q)[3])
end

@testset "unsymmetric matrix linear algebra" begin
    data = [2.0 1.0; 3.0 4.0]
    q = TLArray(data, (TLIndex("row", '+'), TLIndex("col", '-')))

    @test diag(q) == diag(data)
    @test tr(q) == tr(data)
    @test det(q) ≈ det(data)
    @test Array(inv(q)) ≈ inv(data)
    @test inds(inv(q)) == (Telum.change_dir(inds(q)[2]), Telum.change_dir(inds(q)[1]))
    @test _dense_matrix(q * inv(q)) ≈ Matrix{Float64}(I, 2, 2)
end
