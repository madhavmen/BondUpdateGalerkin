function _factorization_svd_inputs(option::LocalSpaceOptions)
    q = getLocalSpace(option)
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    a = getIdentity((qi1, 2), (qi2, 2))
    qf = TLArray(q.F, ("lur2", "lur2", "op"))
    ct = qf * a

    inputs = Tuple{String, TLArray, Vector{Int}}[
        ("identity", q.I, [1]),
        ("fermion operator", q.F, [1, 2]),
        ("contracted operator [2, 4]", ct, [2, 4]),
        ("contracted operator [1, 4]", ct, [1, 4]),
        ("contracted operator [1, 2]", ct, [1, 2]),
    ]
    hasproperty(q, :S) && push!(inputs, ("spin operator", q.S, [1, 2]))
    return inputs
end

function _test_svd_truncation_matches_full(q::TLArray, left_legs)
    full = svd(q, left_legs; cutoff=0.0, get_lists=true)
    full_singular_values = sort(first.(full.kept_list); rev=true)
    @test !isempty(full_singular_values)

    for nkeep in eachindex(full_singular_values)
        @testset "Nkeep=$nkeep" begin
            truncated = svd(q, left_legs; cutoff=0.0, Nkeep=nkeep, get_lists=true)
            kept_singular_values = sort(first.(truncated.kept_list); rev=true)

            @test kept_singular_values ≈ full_singular_values[1:nkeep]
            @test length(truncated.kept_list) == nkeep
            @test length(truncated.kept_list) + length(truncated.trunc_list) == length(full.kept_list)
            @test Telum.spaces(truncated.U)[end] == Telum.spaces(truncated.S)[1]
            @test Telum.spaces(truncated.Vd)[1] == Telum.spaces(truncated.S)[2]
        end
    end
end

function _dense_svd_values(q::TLArray, left_legs)
    left = collect(Int, left_legs)
    right = [leg for leg in 1:ndims(q) if leg ∉ left]
    arr = Array(to_sparse_array(q))
    arr_perm = permutedims(arr, (left..., right...))
    left_dim = prod(size(arr, leg) for leg in left; init=1)
    right_dim = prod(size(arr, leg) for leg in right; init=1)
    return sort(LinearAlgebra.svdvals(reshape(arr_perm, left_dim, right_dim)); rev=true)
end

function _test_svd_singular_values_match_dense(q::TLArray, left_legs)
    result = svd(q, left_legs; cutoff=0.0, get_lists=true)
    telum_vals = Float64[]
    for (singular_value, degeneracy, _, _) in result.kept_list
        for _ in 1:degeneracy
            push!(telum_vals, singular_value)
        end
    end
    filter!(>(1e-10), telum_vals)
    sort!(telum_vals; rev=true)

    dense_vals = filter(>(1e-10), _dense_svd_values(q, left_legs))

    @test length(telum_vals) == length(dense_vals)
    @test telum_vals ≈ dense_vals atol=1e-10 rtol=1e-10
end

function _test_identity_svd_values_are_one(option::LocalSpaceOptions)
    q = getLocalSpace(option)
    result = svd(q.I, (1,); cutoff=0.0, get_lists=true)

    @test !isempty(result.kept_list)
    @test all(entry -> isapprox(first(entry), 1.0; atol=1e-10, rtol=1e-10), result.kept_list)
end

@testset "svd reconstruction" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(2, :U1, :SU2, :SU2),
        FermionSOptions(3, :U1, :SU2, :SU3),
        )
        for (label, q, left_legs) in _factorization_svd_inputs(option)
            @testset "$label" begin
                diff = test_svdQS(q, left_legs; verbose=false)
                @test diff < 1e-9
            end
        end
    end
end

@testset "svd reconstruction for unsymmetric input" begin
    q = getLocalSpace(SpinOptions(nothing, 1))
    for (label, tensor, left_legs) in (
        ("identity", q.I, (1,)),
        ("spin-z", q.Sz, (1,)),
        ("spin-lowering", q.Sm, (1,)),
    )
        @testset "$label" begin
            diff = test_svdQS(tensor, left_legs; cutoff=0.0, verbose=false)
            @test diff < 1e-10
            _test_svd_singular_values_match_dense(tensor, left_legs)

            no_lists = svd(tensor, left_legs; cutoff=0.0)
            @test no_lists.kept_list === nothing
            @test no_lists.trunc_list === nothing

            metadata = svd(tensor, left_legs; cutoff=0.0, get_lists=true)
            @test metadata.kept_list !== nothing
            @test metadata.trunc_list !== nothing
            @test length(metadata.kept_list) + length(metadata.trunc_list) ==
                  min(size(tensor, 1), size(tensor, 2))
        end
    end

    q4 = TLArray(reshape(collect(1.0:36.0), 2, 3, 2, 3);
                 itag=("u1", "u2", "u3", "u4"))
    @testset "rank-four noncontiguous bipartition" begin
        diff = test_svdQS(q4, (1, 3); cutoff=0.0, verbose=false)
        @test diff < 1e-10
        _test_svd_singular_values_match_dense(q4, (1, 3))

        truncated = svd(q4, (1, 3); cutoff=0.0, Nkeep=2, get_lists=true)
        @test length(truncated.kept_list) == 2
        @test length(truncated.trunc_list) == 2
        @test size(truncated.S, 1) == 2
    end
end

@testset "identity svd singular values are one" begin
    _test_identity_svd_values_are_one(FermionSOptions(1, :U1, :SU2, nothing))
    _test_identity_svd_values_are_one(FermionSOptions(3, :U1, :SU2, :SU3))
end

test_svd_diag_rmt()

@testset "svd singular values match dense SVD" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(3, :U1, :SU2, :SU3),
    )
        for (label, q, left_legs) in _factorization_svd_inputs(option)
            @testset "$label" begin
                _test_svd_singular_values_match_dense(q, left_legs)
            end
        end
    end
end

@testset "svd truncation keeps leading singular values" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(3, :U1, :SU2, :SU3),
    )
        for (label, q, left_legs) in _factorization_svd_inputs(option)
            @testset "$label" begin
                _test_svd_truncation_matches_full(q, left_legs)
            end
        end
    end
end

@testset "factorizations accept lazy contraction input" begin
    q = getLocalSpace(SpinOptions(:SU2, 1))
    left = TLArray(q.I, ("left", "bond"))
    right = TLArray(q.S, ("bond", "bond", "right"))
    lazy = Telum._lazy_contract(left, (2,), right, (1,))

    @test lazy isa Telum.TLArrayContraction
    @test svd(lazy, (1,)).U isa TLArray
    @test qr(lazy, (1,)).Q isa TLArray
end

@testset "svd accepts embedded-state input" begin
    q = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))

    scaled_view = 2.0 * q.F
    @test scaled_view isa TLArray
    @test scaled_view.scale == 2.0
    test_svdQS(scaled_view, (1, 2); verbose=false)

    permuted_view = permutedims(q.F, (3, 1, 2))
    @test permuted_view isa TLArray
    @test permuted_view.perm == (3, 1, 2)
    test_svdQS(permuted_view, (2, 3); verbose=false)
end

@testset "qr reconstruction" begin
    q0 = getLocalSpace(SpinOptions(nothing, 1))
    _test_qr_reconstructs(q0.Sz, (1,))
    _test_qr_reconstructs(q0.Sm, (1,))

    q4 = TLArray(reshape(collect(1.0:36.0), 2, 3, 2, 3);
                 itag=("qr_u1", "qr_u2", "qr_u3", "qr_u4"))
    _test_qr_reconstructs(q4, (1, 3); tol=1e-10)

    q_u1 = getLocalSpace(SpinOptions(:U1, 1))
    _test_qr_reconstructs(q_u1.Sz, (1,))
    _test_qr_reconstructs(q_u1.Sm, (1, 2))

    q_su2 = getLocalSpace(SpinOptions(:SU2, 1))
    _test_qr_reconstructs(q_su2.S, (1, 2))

    q_fermion = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))
    _test_qr_reconstructs(q_fermion.F, (1, 2))
end

@testset "qr reconstruction higher-rank tensors" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(2, :U1, :SU2, :SU2),
        FermionSOptions(3, :U1, :SU2, :SU3),
        )
        for (label, q, left_legs) in _factorization_svd_inputs(option)
            ndims(q) > 2 || continue
            @testset "$label" begin
                _test_qr_reconstructs(q, left_legs; tol=1e-8)
            end
        end
    end
end

@testset "qr accepts DiagRMT sector storage" begin
    symm = (U1,)
    qlabels = [(((0,),), ((0,),))]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0))]
    spaces = ([(((0,),), 2)], [(((0,),), 2)])
    q = TLArray(symm, qlabels, wmatdata, wmatinfo,
                [DiagRMT([3.0, -1.5], Val(3), (1, 2))],
                (TLIndex("left", '+'), TLIndex("right", '-')), spaces)

    @test all(i -> q.iszero[i] || q.RMTs[i] isa DiagRMT, eachindex(q.RMTs))
    _test_qr_reconstructs(q, (1,); tol=1e-10)

    q_su2 = get1jtensor(getLocalSpace(SpinOptions(:SU2, 1)).I, 1)
    @test all(i -> q_su2.iszero[i] || q_su2.RMTs[i] isa DiagRMT, eachindex(q_su2.RMTs))
    _test_qr_reconstructs(q_su2, (1,); tol=1e-10)
end

@testset "qr accepts embedded-state input" begin
    q = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))

    scaled_view = 2.0 * q.F
    @test scaled_view isa TLArray
    @test scaled_view.scale == 2.0
    _test_qr_reconstructs(scaled_view, (1, 2))

    permuted_view = permutedims(q.F, (3, 1, 2))
    @test permuted_view isa TLArray
    @test permuted_view.perm == (3, 1, 2)
    _test_qr_reconstructs(permuted_view, (2, 3))
end

@testset "qr keyword selection and errors" begin
    q = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))
    q4 = addSingleton(q.F, 1; itag="left_aux", dir='+')
    explicit = qr(q4, (1, 2, 3), "kwqr")
    keyword = qr(q4, "kwqr"; dir='+')
    @test norm(explicit.Q * explicit.R - keyword.Q * keyword.R) < 1e-9
    @test_throws ArgumentError qr(q4)
    @test_throws ArgumentError qr(q4, Int[])
    @test_throws ArgumentError qr(q4, collect(1:ndims(q4)))
    @test_throws ArgumentError qr(q4, [1, 1])
    @test_throws ArgumentError qr(q4, [0])
end

@testset "randomized SVD leading singular pairs" begin
    for T in (Float64, ComplexF64), dims in ((32, 24), (24, 32))
        rng = MersenneTwister(47)
        m, n = dims
        d = min(m, n)
        left = Matrix(qr(randn(rng, T, m, d)).Q)
        right = Matrix(qr(randn(rng, T, n, d)).Q)
        values = [10.0, 7.0, 4.0, fill(0.01, d - 3)...]
        A = left * Diagonal(values) * right'
        original = copy(A)
        F = @inferred Telum._svd_matrix(A, 3; randomize=true,
                                         random_rng=MersenneTwister(48))
        repeated = Telum._svd_matrix(A, 3; randomize=true,
                                     random_rng=MersenneTwister(48))
        @test F.S ≈ values[1:3] rtol=1e-10
        @test F.U == repeated.U
        @test F.S == repeated.S
        @test F.Vt == repeated.Vt
        @test F.U' * F.U ≈ I
        @test F.Vt * F.Vt' ≈ I
        @test norm(A - F.U * Diagonal(F.S) * F.Vt) ≈ sqrt(d - 3) * 0.01 rtol=1e-7
        @test A == original
        @test issorted(F.S; rev=true)

        # With no power iterations, a larger sketch improves the approximation
        # while leaving the returned candidate count unchanged.
        bare = Telum._svd_matrix(A, 3; randomize=true, random_niter=0,
                                 random_oversampling=0, random_rng=MersenneTwister(48))
        oversampled = @inferred Telum._svd_matrix(A, 3; randomize=true, random_niter=0,
                            random_oversampling=4, random_rng=MersenneTwister(48))
        @test length(oversampled.S) == 3
        @test size(oversampled.U) == (m, 3)
        @test size(oversampled.Vt) == (3, n)
        @test norm(oversampled.S - values[1:3]) < norm(bare.S - values[1:3])
        capped = @inferred Telum._svd_matrix(A, 3; randomize=true, random_niter=0,
                            random_oversampling=typemax(Int), random_rng=MersenneTwister(48))
        @test capped.S ≈ values[1:3]
        @test length(capped.S) == 3

        lowrank = left[:, 1:2] * Diagonal(values[1:2]) * right[:, 1:2]'
        deficient = Telum._svd_matrix(lowrank, 5; randomize=true,
                                      random_rng=MersenneTwister(49))
        @test deficient.U * Diagonal(deficient.S) * deficient.Vt ≈ lowrank
        full = Telum._svd_matrix(A, d + 3; randomize=true)
        @test full.S ≈ svdvals(A)
        for Z in (zeros(T, m, n), zeros(T, m, 0))
            empty = @inferred Telum._svd_matrix(Z, 3; randomize=true)
            @test isempty(empty.S)
            @test size(empty.U) == (size(Z, 1), 0)
            @test size(empty.Vt) == (0, size(Z, 2))
        end
    end
end

@testset "randomized SVD for unsymmetric tensors" begin
    for T in (Float64, ComplexF64)
        rng = MersenneTwister(51)
        A = randn(rng, T, 24, 3) * randn(rng, T, 3, 20)
        q = TLArray(A; itag=("left", "right"))
        before = copy(Array(to_sparse_array(q)))
        result = svd(q, (1,), "L", "R"; randomize=true, Nkeep=3,
                     random_rng=MersenneTwister(52), get_lists=true)
        @test first.(result.kept_list) ≈ svdvals(A)[1:3]
        @test isempty(result.trunc_list) # Only three candidate values were computed.
        @test Array(to_sparse_array(result.U * result.S * result.Vd)) ≈ A
        @test Array(to_sparse_array(q)) == before
        @test Telum.spaces(result.U)[end] == Telum.spaces(result.S)[1]
        @test Telum.spaces(result.Vd)[1] == Telum.spaces(result.S)[2]
        @test Telum.inds(result.U)[end] == TLIndex("L", '-')
        @test Telum.inds(result.S) == (TLIndex("L", '+'), TLIndex("R", '+'))
        @test Telum.inds(result.Vd)[1] == TLIndex("R", '-')
        selected = svd(q, "L", "R"; itag="left", randomize=true, Nkeep=3,
                       random_rng=MersenneTwister(52), get_lists=true)
        alias = svd_cgtsvd(q, (1,), "L", "R"; randomize=true, Nkeep=3,
                           random_rng=MersenneTwister(52), get_lists=true)
        @test selected.kept_list == result.kept_list == alias.kept_list
        cutoff = svd(q, (1,); randomize=true, Nkeep=3, cutoff=0.9,
                     random_rng=MersenneTwister(52), get_lists=true)
        @test length(cutoff.kept_list) < 3
        @test length(cutoff.kept_list) + length(cutoff.trunc_list) == 3
        no_lists = svd(q, (1,); randomize=true, Nkeep=3)
        @test no_lists.kept_list === nothing
        @test no_lists.trunc_list === nothing
    end
    # Exercise keyword forwarding through both overloads and the compatibility alias.
    A = randn(MersenneTwister(53), 24, 20)
    q = TLArray(A; itag=("left", "right"))
    options = (randomize=true, Nkeep=3, random_niter=0,
               random_oversampling=typemax(Int), get_lists=true)
    for result in (svd(q, (1,); options...),
                   svd(q; itag="left", options...),
                   svd_cgtsvd(q, (1,); options...))
        @test first.(result.kept_list) ≈ svdvals(A)[1:3]
        @test isempty(result.trunc_list)
    end
    qzero = TLArray(zeros(8, 6); itag=("left", "right"))
    zero_result = svd(qzero, (1,); randomize=true, Nkeep=2, get_lists=true)
    @test isempty(zero_result.kept_list)
    @test isempty(zero_result.trunc_list)
    @test size(zero_result.S) == (0, 0)
end

@testset "randomized SVD class allocation and global truncation" begin
    for T in (Float64, ComplexF64)
        qlabels = [(((charge,),), ((charge,),)) for charge in 0:2]
        splist = [(((charge,),), d) for (charge, d) in zip(0:2, (12, 12, 4))]
        # Three symmetry classes include one known-zero class, which contributes no candidates.
        blocks = [reshape(Matrix{T}(Diagonal(v)), length(v), length(v), 1) for v in
                  ([100.0, 90.0, 20.0, fill(0.001, 9)...],
                   [10.0, 8.0, fill(0.001, 10)...], zeros(4))]
        q = TLArray((U1,), qlabels, Float64[],
                    [Telum._empty_wmat_info(Val(0)) for _ in 1:3], blocks,
                    (TLIndex("left", '+'), TLIndex("right", '-')), (splist, splist))
        before = copy(Array(to_sparse_array(q)))
        proportional = svd(q, (1,); randomize=true, Nkeep=3, random_niter=4,
                           random_rng=MersenneTwister(54), get_lists=true)
        per_class = svd(q, (1,); randomize=true, Nkeep=3, random_mode=:per_class,
                        random_niter=0, random_oversampling=typemax(Int),
                        random_rng=MersenneTwister(54), get_lists=true)
        @test first.(proportional.kept_list) ≈ [100.0, 90.0, 10.0] rtol=1e-8
        @test first.(per_class.kept_list) ≈ [100.0, 90.0, 20.0] rtol=1e-8
        @test length(proportional.kept_list) + length(proportional.trunc_list) == 4
        @test length(per_class.kept_list) + length(per_class.trunc_list) == 6
        @test issorted(first.(per_class.kept_list); rev=true)
        @test all(entry -> entry[3] == ((0,),), per_class.kept_list)
        exact = svd(q, (1,); Nkeep=3, get_lists=true)
        @test first.(per_class.kept_list) ≈ first.(exact.kept_list)
        @test Array(to_sparse_array(per_class.U * per_class.S * per_class.Vd)) ≈
              Array(to_sparse_array(exact.U * exact.S * exact.Vd))
        @test Array(to_sparse_array(q)) == before
    end
end

@testset "randomized SVD preserves non-Abelian factors and embedded state" begin
    base = getLocalSpace(SpinOptions(:SU2, 1)).I
    values = [9.0, 6.0, 3.0, zeros(9)...]
    block = reshape(base.RMTs[1][1] .* Matrix(Diagonal(values)), 12, 12, 1)
    splist = [(first(base.spaces[1])[1], 12)]
    q = TLArray((SU{2},), copy(base.qlabels), copy(base.wmatdata), copy(base.wmatinfo),
                [block], (TLIndex("left", '+'), TLIndex("right", '-')), (splist, splist))
    for source in (q, (2 + im) * conj(permutedims(q, (2, 1))))
        before = copy(Array(to_sparse_array(source)))
        result = svd(source, (1,); randomize=true, Nkeep=3, random_mode=:per_class,
                     random_rng=MersenneTwister(56), get_lists=true)
        exact = svd(source, (1,); Nkeep=3, get_lists=true)
        @test first.(result.kept_list) ≈ first.(exact.kept_list)
        @test Array(to_sparse_array(result.U * result.S * result.Vd)) ≈ before
        @test Array(to_sparse_array(source)) == before
        for factor in (result.U, result.S, result.Vd), sector in Telum.sector_slots(factor)
            Telum.is_sector_zero(factor, sector) && continue
            W = Telum.sector_wmat(factor, sector, 1)
            @test W' * W ≈ I
        end
    end
end

@testset "randomized SVD option validation and exact defaults" begin
    @test (@inferred Telum._svd_random_candidate_count(12, 28, 3, :proportional, 0.1)) == 2
    @test Telum._svd_random_candidate_count(2, 2, 3, :per_class, 0.1) == 2
    @test Telum._svd_random_candidate_count(0, 0, 3, :proportional, 0.1) == 0
    q = TLArray(Matrix(Diagonal([3.0, 2.0, 1.0])); itag=("left", "right"))
    @test_throws ArgumentError svd(q, (1,); randomize=true)
    for k in (0, -1)
        @test_throws ArgumentError svd(q, (1,); randomize=true, Nkeep=k)
    end
    for options in ((random_mode=:invalid,), (random_tol=0.0,), (random_tol=-1.0,),
                    (random_tol=Inf,), (random_tol=NaN,), (random_niter=-1,),
                    (random_niter=0.5,), (random_oversampling=-1,),
                    (random_oversampling=0.5,), (random_oversampling=nothing,), (random_rng=nothing,))
        @test_throws ArgumentError svd(q, (1,); randomize=true, Nkeep=1, options...)
    end
    exact = svd(q, (1,); get_lists=true)
    ignored = svd(q; itag="left", get_lists=true, random_mode=:invalid,
                  random_tol=-1, random_niter=-1, random_oversampling=-1, random_rng=nothing)
    @test exact.kept_list == ignored.kept_list
    @test Array(to_sparse_array(exact.U)) == Array(to_sparse_array(ignored.U))
    @test first.(svd(q, (1,); randomize=true, Nkeep=20, get_lists=true).kept_list) == [3, 2, 1]
end
