function _one_sector_matrix_tlarray(matrix::AbstractMatrix{T};
                                    qlabel=((0,),),
                                    tags=("phys", "phys")) where {T}
    n = size(matrix, 1)
    @assert size(matrix, 2) == n
    symmetries = (U1,)
    qlabels = [(qlabel, qlabel)]
    wmatdata = Float64[]
    wmatinfo = [Telum._empty_wmat_info(Val(0))]
    rmt = reshape(Matrix{T}(matrix), n, n, 1)
    inds = (TLIndex(tags[1], '+'), TLIndex(tags[2], '-'))
    spaces = ([(qlabel, n)], [(qlabel, n)])
    return TLArray(symmetries, qlabels, wmatdata, wmatinfo, [rmt], inds, spaces)
end

function _dense_matrix(q::TLArray)
    arr = Array(to_sparse_array(q))
    return reshape(arr, size(arr, 1), size(arr, 2))
end

function _eigen_reconstruction(result)
    # Lock the original matrix leg so multiplication contracts only the eigen bond.
    V = lock(result.V, 1)
    return isnothing(result.V_inv) ? V * result.D * result.V' : V * result.D * result.V_inv
end

function _assert_eigen_reconstructs(q::TLArray; hermitian::Bool=false, tol=1e-11)
    result = eigen(q; hermitian)
    rec = _eigen_reconstruction(result)
    @test norm(q - rec) / max(norm(q), 1.0) < tol
    return result
end

function _expanded_eigenvalues(result)
    vals = mapreduce(entry -> fill(entry[1], entry[2]), append!, result.eig_list;
                     init=eltype(first(result.eig_list)[1])[])
    return sort(vals; by=abs)
end

function _dense_eigenvalues(q::TLArray; hermitian::Bool=false)
    vals = hermitian ? LinearAlgebra.eigvals(Hermitian(_dense_matrix(q))) :
                       LinearAlgebra.eigvals(_dense_matrix(q))
    return sort(vals; by=abs)
end

@testset "identity eigenvalues are one" begin
    for option in (
        FermionSOptions(1, :U1, :SU2, nothing),
        FermionSOptions(3, :U1, :SU2, :SU3),
    )
        q = getLocalSpace(option)
        result = eigen(q.I; hermitian=true)
        vals = _expanded_eigenvalues(result)
        @test !isempty(vals)
        @test all(isapprox.(vals, one(eltype(vals)); atol=1e-10, rtol=1e-10))
        @test isnothing(result.V_inv)
    end

    q10 = _one_sector_matrix_tlarray(Matrix{Float64}(I, 10, 10))
    result10 = eigen(q10; hermitian=true)
    @test _expanded_eigenvalues(result10) ≈ ones(10)
end

@testset "eigen path assigns inverse only for non-Hermitian input" begin
    hermitian_matrix = Symmetric(randn(10, 10))
    hermitian_q = _one_sector_matrix_tlarray(Matrix(hermitian_matrix))
    hermitian_result = _assert_eigen_reconstructs(hermitian_q; hermitian=true)
    @test isnothing(hermitian_result.V_inv)

    general_matrix = randn(10, 10)
    general_matrix[1, 2] += 3.0
    general_matrix[2, 1] -= 2.0
    general_q = _one_sector_matrix_tlarray(general_matrix)
    general_result = _assert_eigen_reconstructs(general_q)
    @test !isnothing(general_result.V_inv)
end

@testset "eigen diagonalizes unsymmetric tensors" begin
    A = [1.0 2.0 0.0; -3.0 4.0 1.0; 0.5 0.0 2.0]
    q = TLArray(A, (TLIndex("phys", '+'), TLIndex("phys", '-')))
    result = eigen(q)
    @test !isnothing(result.V_inv)
    @test sort([entry[1] for entry in result.eig_list]; by=abs) ≈
          sort(eigvals(A); by=abs)
    @test _dense_matrix(_eigen_reconstruction(result)) ≈ A

    H = [2.0 1.0; 1.0 3.0]
    qH = TLArray(H, (TLIndex("phys", '+'), TLIndex("phys", '-')))
    resultH = eigen(qH)
    @test isnothing(resultH.V_inv)
    @test _dense_matrix(_eigen_reconstruction(resultH)) ≈ H

    result_zero = eigen(zero(qH))
    @test isnothing(result_zero.V_inv)
    @test all(iszero, _expanded_eigenvalues(result_zero))
end

@testset "unsymmetric matrix operations honor deferred views" begin
    data = ComplexF64[1 + 2im 3 - im; -2 + 0.5im 4 - 3im]
    base = TLArray(data, (TLIndex("row", '+'), TLIndex("col", '-')))
    q = 2.5 * conj(permutedims(base, (2, 1)))
    dense = 2.5 .* conj.(transpose(data))

    @test Array(q) ≈ dense
    @test diag(q) ≈ LinearAlgebra.diag(dense)
    @test tr(q) ≈ LinearAlgebra.tr(dense)
    @test det(q) ≈ LinearAlgebra.det(dense)

    qinv = inv(q)
    @test _dense_matrix(qinv) ≈ LinearAlgebra.inv(dense)
    @test inds(qinv) == (Telum.change_dir(inds(q)[2]), Telum.change_dir(inds(q)[1]))
end

@testset "eigen accepts lazy contraction input" begin
    left = _one_sector_matrix_tlarray(Matrix(Symmetric(randn(5, 5)));
                                     tags=("phys", "bond"))
    right = _one_sector_matrix_tlarray(Matrix{Float64}(I, 5, 5);
                                      tags=("bond", "phys"))
    lazy = Telum._lazy_contract(left, (2,), right, (1,))

    @test lazy isa Telum.TLArrayContraction
    @test eigen(lazy; hermitian=true).V isa TLArray
end

@testset "eigenvalues match dense eigendecomposition" begin
    real_hermitian = Matrix(Symmetric(randn(10, 10)))
    real_hermitian_q = _one_sector_matrix_tlarray(real_hermitian)
    real_hermitian_result = _assert_eigen_reconstructs(real_hermitian_q; hermitian=true)
    @test _expanded_eigenvalues(real_hermitian_result) ≈
          _dense_eigenvalues(real_hermitian_q; hermitian=true) atol=1e-10 rtol=1e-10

    complex_general = randn(ComplexF64, 10, 10)
    complex_general[1, 2] += 2.0 - 0.5im
    complex_general[2, 1] -= 1.0 + 1.5im
    complex_general_q = _one_sector_matrix_tlarray(complex_general)
    complex_general_result = _assert_eigen_reconstructs(complex_general_q; tol=1e-8)
    @test !isnothing(complex_general_result.V_inv)
    @test _expanded_eigenvalues(complex_general_result) ≈
          _dense_eigenvalues(complex_general_q) atol=1e-8 rtol=1e-8
end

test_eigen_multi_sector()
test_eigen_tlarrayview()

@testset "discard_eigen signed count and threshold selection" begin
    values = result -> first.(result.eig_list)
    spectrum = [-3.0, -1.0, 0.0, 2.0, 2.0, 5.0]
    q = _one_sector_matrix_tlarray(Matrix(Diagonal(spectrum)))
    result = eigen(q; hermitian=true)
    before = copy(result.eig_list)
    for (options, expected) in (
        ((Nkeep=2, tol=0.0), [-3.0, -1.0]),
        ((Nkeep=2, keep_low=false, tol=0.0), [2.0, 5.0]),
        ((threshold=2.0,), [-3.0, -1.0, 0.0, 2.0, 2.0]),
        ((threshold=2.0, keep_low=false), [2.0, 2.0, 5.0]),
        ((threshold=prevfloat(2.0),), [-3.0, -1.0, 0.0]),
        ((threshold=nextfloat(2.0), keep_low=false), [5.0]),
        ((threshold=0.0,), [-3.0, -1.0, 0.0]),
        ((threshold=0.0, keep_low=false), [0.0, 2.0, 2.0, 5.0]),
        ((threshold=-100.0,), Float64[]),
        ((threshold=100.0,), spectrum),
        ((Nkeep=0,), Float64[]),
        ((Nkeep=typemax(Int),), spectrum),
    )
        kept, discarded = discard_eigen(result, "K", "D"; options...)
        @test values(kept) == expected
        @test sort(vcat(values(kept), values(discarded))) == spectrum
        @test issorted(values(discarded))
        @test kept.V_inv === discarded.V_inv === nothing
        @test Telum.inds(kept.D)[1] == TLIndex("K", '+')
        @test Telum.inds(discarded.D)[1] == TLIndex("D", '+')
        @test _dense_matrix(_eigen_reconstruction(kept)) +
              _dense_matrix(_eigen_reconstruction(discarded)) ≈ _dense_matrix(q)
    end
    @test result.eig_list == before
    @test _dense_matrix(_eigen_reconstruction(result)) ≈ _dense_matrix(q)
    kept, _ = discard_eigen(result; threshold=0.0, tol=100.0)
    @test values(kept) == [-3.0, -1.0, 0.0] # Explicit thresholds never expand.
    empty, _ = discard_eigen(result; Nkeep=0)
    for options in ((Nkeep=3,), (threshold=0.0,))
        kept, discarded = discard_eigen(empty; options...)
        @test isempty(kept.eig_list) && isempty(discarded.eig_list)
    end
    zeros_result = eigen(_one_sector_matrix_tlarray(zeros(3, 3)); hermitian=true)
    kept, discarded = discard_eigen(zeros_result; Nkeep=2, tol=0)
    @test values(kept) == [0.0, 0.0]
    @test values(discarded) == [0.0]
end

@testset "discard_eigen gap window and validation" begin
    values = result -> first.(result.eig_list)
    result = eigen(_one_sector_matrix_tlarray(Matrix(Diagonal(
                    [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]))); hermitian=true)
    for keep_low in (true, false)
        kept, _ = discard_eigen(result; Nkeep=2, tol=0.5, keep_low=keep_low)
        @test values(kept) == (keep_low ? [0.0, 1.0, 2.0] : [10.0, 11.0, 12.0])
        kept, discarded = discard_eigen(result; Nkeep=4, tol=0.5, keep_low=keep_low)
        @test length(kept.eig_list) == 6 && isempty(discarded.eig_list)
        for tol in (floatmax(Float64), typemax(Int))
            kept, _ = discard_eigen(result; Nkeep=2, tol=tol, keep_low=keep_low)
            @test length(kept.eig_list) == 6
        end
    end
    uniform = eigen(_one_sector_matrix_tlarray(Matrix(Diagonal(collect(0.0:5.0)))); hermitian=true)
    for keep_low in (true, false)
        kept, _ = discard_eigen(uniform; Nkeep=2, tol=0.5, keep_low=keep_low)
        @test length(kept.eig_list) == 2 # Equal gaps choose the smaller count.
    end
    for options in ((;), (Nkeep=2, threshold=1.0), (Nkeep=-1,),
                    (threshold=Inf,), (threshold=NaN,), (Nkeep=2, tol=-1.0),
                    (Nkeep=2, tol=Inf), (threshold=0.0, tol=NaN))
        @test_throws ArgumentError discard_eigen(result; options...)
    end
    @test_throws MethodError discard_eigen(result, 2)
end

@testset "discard_eigen sector and inverse alignment" begin
    values = result -> first.(result.eig_list)
    qlabels = [(((charge,),), ((charge,),)) for charge in 0:1]
    splist = [(((charge,),), 3) for charge in 0:1]
    blocks = [reshape(Matrix(Diagonal(v)), 3, 3, 1) for v in
              ([2.0, -3.0, 0.0], [2.0, 5.0, -1.0])]
    q = TLArray((U1,), qlabels, Float64[],
                [Telum._empty_wmat_info(Val(0)) for _ in 1:2], blocks,
                (TLIndex("phys", '+'), TLIndex("phys", '-')), (splist, splist))
    result = eigen(q; hermitian=true)
    kept, discarded = discard_eigen(result; Nkeep=2, keep_low=false, tol=0)
    @test values(kept) == [2.0, 5.0]
    @test kept.eig_list[1][3] == ((0,),) # Tied values retain their source order.
    for part in (kept, discarded), entry in part.eig_list
        slot = only(filter(i -> Telum._eig_sector_qlabels(part.D, i) == entry[3],
                           collect(Telum.sector_slots(part.D))))
        rmt, scale = Telum.sector_rmt(part.D, slot)
        @test scale * rmt[entry[4], entry[4], 1] ≈ entry[1]
    end
    @test _dense_matrix(_eigen_reconstruction(kept)) +
          _dense_matrix(_eigen_reconstruction(discarded)) ≈ _dense_matrix(q)
    # Real, non-Hermitian input exercises V_inv slicing and signed ordering.
    matrix = [-4.0 2.0 0.0; 0.0 0.0 1.0; 0.0 0.0 3.0]
    general = eigen(TLArray(matrix, (TLIndex("phys", '+'), TLIndex("phys", '-'))))
    for options in ((Nkeep=1, tol=0.0), (threshold=0.0, keep_low=false))
        kept, discarded = discard_eigen(general; options...)
        @test kept.V_inv !== nothing && discarded.V_inv !== nothing
        @test values(kept) == (haskey(options, :Nkeep) ? [-4.0] : [0.0, 3.0])
        @test _dense_matrix(_eigen_reconstruction(kept)) +
              _dense_matrix(_eigen_reconstruction(discarded)) ≈ matrix
    end
    # Count whole non-Abelian multiplets, not individual degenerate states.
    spin = eigen(getLocalSpace(SpinOptions(:SU2, 1)).I; hermitian=true)
    kept, _ = discard_eigen(spin; Nkeep=1, tol=0)
    @test length(kept.eig_list) == 1
    @test kept.eig_list[1][2] == 2
    for factor in (kept.V, kept.D), slot in Telum.sector_slots(factor)
        Telum.is_sector_zero(factor, slot) && continue
        W = Telum.sector_wmat(factor, slot, 1)
        @test W' * W ≈ I
    end
end
