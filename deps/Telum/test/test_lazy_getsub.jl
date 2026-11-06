@testset "lazy getsub preserves known-zero sector slots" begin
    symm = (U1,)
    qlabels = [(((0,),), ((0,),)), (((1,),), ((1,),))]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0)), Telum._empty_wmat_info(Val(0))]
    RMTs = Vector{Array{Float64, 3}}(undef, 2)
    RMTs[1] = reshape([1.0], 1, 1, 1)
    spaces = ([(((0,),), 1), (((1,),), 1)],
              [(((0,),), 1), (((1,),), 1)])
    q = TLArray(symm, qlabels, wmatdata, wmatinfo, RMTs,
                (TLIndex("x", '+'), TLIndex("y", '-')), spaces)

    sub = Telum._getsub_lazy(q, 1, _ -> Colon())

    @test sub isa Telum.SubTLArray
    @test Telum.stored_qlabels(sub) == Telum.stored_qlabels(q)
    @test sub.source_sectors == [1, 2]
    @test Telum.source_sector(sub, 2) == 2
    @test !Telum.is_sector_zero(sub, 1)
    @test Telum.is_sector_zero(sub, 2)
    @test !Telum.is_sector_defined(sub, 2)
    Telum.compute_sectors(sub, [2])
    @test !Telum.is_sector_defined(sub, 2)
end

@testset "lazy getsub shares or densifies DiagRMT sectors by selector shape" begin
    symm = (U1,)
    qlabels = [(((0,),), ((0,),)), (((1,),), ((1,),))]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0)), Telum._empty_wmat_info(Val(0))]
    spaces = ([(((0,),), 2), (((1,),), 2)],
              [(((0,),), 2), (((1,),), 2)])
    q = TLArray(symm, qlabels, wmatdata, wmatinfo,
                [DiagRMT([1.0, 2.0], Val(3), (1, 2)),
                 DiagRMT([3.0, 4.0], Val(3), (1, 2))],
                (TLIndex("x", '+'), TLIndex("y", '-')), spaces)

    whole = Telum._getsub_lazy(q, 1, _ -> Colon())
    @test eltype(whole.RMTs) <: DiagRMT
    Telum.compute_sectors(whole, [1])
    @test whole.RMTs[1] === q.RMTs[1]

    mixed = Telum._getsub_lazy(q, 1,
                               sector -> sector == ((0,),) ? Colon() :
                                         sector == ((1,),) ? 1 : nothing)
    @test eltype(mixed.RMTs) <: Array{Float64, 3}
    Telum.compute_sectors(mixed, [1, 2])
    @test mixed.RMTs[1] == Array(q.RMTs[1])
    @test mixed.RMTs[1] !== q.RMTs[1]
    selector = ([1], Colon(), Colon())
    @test mixed.RMTs[2] == q.RMTs[2][selector...]
end

@testset "lazy getsub keyword dispatch does not materialize source contraction" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.Sz, ("bond", "right"))
    lazy = Telum._lazy_contract(left, (2,), right, (1,))

    sub = Telum.getsub(lazy, _ -> Colon(); itag="left")

    @test sub isa Telum.SubTLArray
    @test !Telum.is_sector_defined(lazy, 1)
    Telum.compute_sectors(sub, [1])
    @test Telum.is_sector_defined(lazy, 1)
end

@testset "lazy getsub maps embedded-state visible legs to source legs" begin
    symm = (U1,)
    qlabels = [(((0,),), ((10,),)), (((1,),), ((11,),))]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0)), Telum._empty_wmat_info(Val(0))]
    spaces = ([(((0,),), 1), (((1,),), 1)],
              [(((10,),), 1), (((11,),), 1)])
    q = TLArray(symm, qlabels, wmatdata, wmatinfo,
                [reshape([2.0], 1, 1, 1), reshape([3.0], 1, 1, 1)],
                (TLIndex("x", '+'), TLIndex("y", '-')), spaces)
    view_q = permutedims(q, (2, 1))

    sub_view = Telum.getsub(view_q, 1, sector -> sector == ((10,),) ? Colon() : nothing)

    @test sub_view isa TLArray
    @test sub_view.perm == view_q.perm
    @test Telum.sector_count(sub_view) == 1
    @test Telum.sector_qlabel(sub_view, 1, 1) == ((10,),)
    @test Array(to_sparse_array(sub_view)) == Array(to_sparse_array(view_q))[1:1, :]
end

@testset "lazy getsub preserve_space rejects slicing before materialization" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.Sz, ("bond", "right"))
    lazy = Telum._lazy_contract(left, (2,), right, (1,))

    @test_throws ArgumentError Telum.getsub(lazy, 1, _ -> 1; preserve_space=true)
    @test !Telum.is_sector_defined(lazy, 1)
end

@testset "chained lazy getsub materializes through sector maps" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.Sz, ("bond", "right"))
    lazy = Telum._lazy_contract(left, (2,), right, (1,))

    sub1 = Telum.getsub(lazy, 1, _ -> Colon())
    sub2 = Telum.getsub(sub1, 2, _ -> Colon())

    @test sub2 isa Telum.SubTLArray
    @test Telum.source_sector(sub2, 1) == 1
    @test !Telum.is_sector_defined(lazy, 1)
    Telum.compute_sectors(sub2, [1])
    @test Telum.is_sector_defined(lazy, 1)
    @test sub2.RMTs[1] === sub1.RMTs[1]
end

@testset "TLArray conversion aliases materialized lazy getsub storage" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.Sz, ("bond", "right"))
    lazy = Telum._lazy_contract(left, (2,), right, (1,))
    sub = Telum.getsub(lazy, 1, _ -> Colon())

    converted = TLArray(sub)

    @test converted isa TLArray
    @test Telum.is_sector_defined(sub, 1)
    @test Telum.stored_qlabels(converted) === Telum.stored_qlabels(sub)
    @test first(Telum.stored_wmat_storage(converted)) === first(Telum.stored_wmat_storage(sub))
    @test last(Telum.stored_wmat_storage(converted)) === last(Telum.stored_wmat_storage(sub))
    @test converted.RMTs === sub.RMTs
    @test converted.isdefined === sub.isdefined
    @test converted.iszero === sub.iszero
    @test converted.RMTs[1] === sub.RMTs[1]

    copied = copy(converted)
    @test copied.RMTs !== converted.RMTs
    @test copied.RMTs[1] !== converted.RMTs[1]
end
