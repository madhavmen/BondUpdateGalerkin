@testset "re-exported LurCGT symmetries" begin
    @test :Z in names(Telum)
    @test :U1 in names(Telum)
    @test :SU in names(Telum)
    @test :SO in names(Telum)
    @test :Sp in names(Telum)
    @test :printmeta in names(Telum)
    @test :getSymmetryInfo in names(Telum)

    @test Telum.Z === LurCGT.Z
    @test Telum.U1 === LurCGT.U1
    @test Telum.SU === LurCGT.SU
    @test Telum.SO === LurCGT.SO
    @test Telum.Sp === LurCGT.Sp
end

@testset "SpinOptions local operators across symmetries and spins" begin
    for symmetry in (:SU2, :U1, nothing), s2 in 1:5
        q = @test_nowarn getLocalSpace(SpinOptions(symmetry, s2))
        expected_ops = symmetry == :SU2 ? (:S, :I) : (:Sp, :Sz, :Sm, :I)

        for op in expected_ops
            @test hasproperty(q, op)
            tensor = getproperty(q, op)
            @test !isempty(tensor.RMTs)
            @test any(tensor.isdefined)
        end
    end
end

@testset "FermionS full-channel IROP names omit channel suffixes" begin
    q1 = getLocalSpace(FermionSOptions(1, :U1, :SU2, nothing))
    @test hasproperty(q1, :F)
    @test !hasproperty(q1, :F1)

    q3 = getLocalSpace(FermionSOptions(3, :U1, :SU2, :SU3))
    @test hasproperty(q3, :F)
    @test !hasproperty(q3, :F123)
end

@testset "three-channel SO(3) local space" begin
    opts = FermionSOptions(3, :U1, :SU2, :SO3)
    symmetries, weights, lowering, mwirops = getSymmetryInfo(opts)
    @test symmetries == (U1, SU{2}, SO{3})

    basic = Telum.FermionS_basicops(3)
    n1 = basic.NN[1, 1] + basic.NN[2, 1]
    n3 = basic.NN[1, 3] + basic.NN[2, 3]
    z = 2 * (n1 - n3)
    minus = sqrt(2) * sum(
        basic.FF[spin, to]' * basic.FF[spin, from]
        for spin in 1:2 for (to, from) in ((2, 1), (3, 2)))
    @test weights[3] == [(Int(value),) for value in diag(z)]
    @test lowering[3][1] ≈ minus
    @test LurCGT.comm(z, minus) ≈ -2 * minus
    @test LurCGT.comm(minus', minus) ≈ z
    @test mwirops[:F] == basic.FF[2, 3]
    @test mwirops[:S] ≈ -sum(basic.SS[i]' for i in 1:3) / sqrt(2)

    q = getLocalSpace(opts)
    @test symm(q.I) == (U1, SU{2}, SO{3})
    @test hasproperty(q, :F)
    @test hasproperty(q, :S)
    @test Array(to_sparse_array(q.I)) ≈ Matrix{Float64}(I, 64, 64)
    @test any(x -> !iszero(x), Array(to_sparse_array(q.F)))
    @test any(x -> !iszero(x), Array(to_sparse_array(q.S)))

    alias_q = getLocalSpace(FermionSOptions(3, :U1, :SU2, :SO))
    @test symm(alias_q.I) == symm(q.I)
    so_only = getLocalSpace(FermionSOptions(3, nothing, nothing, :SO3))
    @test symm(so_only.I) == (SO{3},)
    @test_throws ArgumentError getSymmetryInfo(FermionSOptions(3, :U1, :SU2, :SO4))
    @test_throws ArgumentError getSymmetryInfo(FermionSOptions(2, :U1, :SU2, :SO3))
end

@testset "two-orbital L3 in spinful local-space convention" begin
    basic = Telum.FermionS_basicops(2)
    L3 = Telum.FermionS_L3(basic.FF)
    expected = im * sum(
        basic.FF[spin, 1]' * basic.FF[spin, 2] -
        basic.FF[spin, 2]' * basic.FF[spin, 1]
        for spin in 1:2
    )
    @test size(L3) == (16, 16)
    @test L3 == expected
    @test L3' == L3
end

@testset "Fermion local space IROPs" begin
    q1 = getLocalSpace(FermionOptions(U1))
    @test hasproperty(q1, :F)
    @test hasproperty(q1, :Z)
    @test hasproperty(q1, :I)
    @test !hasproperty(q1, :F1)
    @test size(to_sparse_array(q1.I)) == (2, 2)
    @test Array(to_sparse_array(q1.I)) == Array(_test_to_sparse_array(q1.I))

    q3 = getLocalSpace(FermionOptions(3, :U1, :SU3))
    @test hasproperty(q3, :F)
    @test !hasproperty(q3, :F123)
    @test size(to_sparse_array(q3.I)) == (8, 8)
end

@testset "no-symmetry local spaces" begin
    spin = getLocalSpace(SpinOptions(nothing, 1))
    @test symm(spin.I) == ()
    @test typeof(spin.I).parameters[6] === ProductSymm{Tuple{}}
    @test Telum.stored_qlabels(spin.I) == [((), ())]
    @test Telum.spaces(spin.I) == ([(() , 2)], [(() , 2)])
    @test _test_sector_rmt(spin.I, 1) ≈ Matrix{Float64}(I, 2, 2)
    @test _test_sector_rmt(spin.Sz, 1) ≈ [0.5 0.0; 0.0 -0.5]
    @test _test_sector_rmt(spin.Sp, 1) ≈ (-1 / sqrt(2)) * [0.0 1.0; 0.0 0.0]
    @test _test_sector_rmt(spin.Sm, 1) ≈ (1 / sqrt(2)) * [0.0 0.0; 1.0 0.0]

    fermion = getLocalSpace(FermionOptions(1, nothing, nothing))
    @test symm(fermion.I) == ()
    @test typeof(fermion.I).parameters[6] === ProductSymm{Tuple{}}
    @test Telum.stored_qlabels(fermion.I) == [((), ())]
    @test Telum.spaces(fermion.I) == ([(() , 2)], [(() , 2)])
    @test _test_sector_rmt(fermion.I, 1) ≈ Matrix{Float64}(I, 2, 2)
    @test size(_test_sector_rmt(fermion.F, 1)) == (2, 2)
    @test size(_test_sector_rmt(fermion.Z, 1)) == (2, 2)

    spinful = getLocalSpace(FermionSOptions(1, nothing, nothing, nothing))
    @test symm(spinful.I) == ()
    @test typeof(spinful.I).parameters[6] === ProductSymm{Tuple{}}
    @test Telum.stored_qlabels(spinful.I) == [((), ())]
    @test Telum.spaces(spinful.I) == ([(() , 4)], [(() , 4)])
    @test _test_sector_rmt(spinful.I, 1) ≈ Matrix{Float64}(I, 4, 4)
    @test size(_test_sector_rmt(spinful.Fu, 1)) == (4, 4)
    @test size(_test_sector_rmt(spinful.Fd, 1)) == (4, 4)
    @test size(_test_sector_rmt(spinful.Z, 1)) == (4, 4)
end

struct NonCommutingSymmetryOptions <: LocalSpaceOptions end

function Telum.getSymmetryInfo(::NonCommutingSymmetryOptions)
    symm = (U1, SU{2})
    weights = ([(1,), (-1,)], [(1,), (-1,)])
    lowering_ops = (Matrix{Int}[], [sparse([0 0; 1 0])])

    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    mwirops[:I] = spdiagm(0 => ones(Float64, 2))
    return symm, weights, lowering_ops, mwirops
end

@testset "getLocalSpace validates cross-symmetry commutation" begin
    err = try
        getLocalSpace(NonCommutingSymmetryOptions())
        nothing
    catch caught
        caught
    end

    @test err isa ArgumentError
    @test occursin("must commute", sprint(showerror, err))
    @test occursin("weight[1]", sprint(showerror, err))
    @test occursin("lowering[1]", sprint(showerror, err))
end

"""
    _test_matrix_localspace(symmetries, weights, lowering, operators)

Build matrix-defined local operators and return their physical basis in the
same sorted sector order used by dense TLArray expansion. Inputs are not mutated.
"""
function _test_matrix_localspace(symmetries, weights, lowering, operators)
    ws, ls, os, U = Telum._normalize_localspace_data(symmetries, weights, lowering, operators)
    V, ranges = decompose_space(symmetries, ws, ls)
    order = Int[]
    for q in sort(collect(keys(ranges))), r in ranges[q]
        append!(order, r[1]:r[2])
    end
    B = (U === nothing ? Matrix(V) : U * V)[:, order]
    tensors = Telum._localspace_from_data(symmetries, ws, ls, os)
    return tensors, B, ws, ls, os, U
end

@testset "matrix quantum numbers preserve operator reconstruction" begin
    sy, ws, ls, os = getSymmetryInfo(SpinOptions(:SU2, 1))
    reference = getLocalSpace(SpinOptions(:SU2, 1))
    block = reshape(ComplexF64[1, im], 1, 1, 1, 2)
    W, reduced, _ = Telum._localspace_wmat(block, 4)
    @test eltype(W) <: Real
    @test W' * W ≈ I
    @test reconstruct(W, reduced, 4) ≈ block
    Z = Diagonal(first.(ws[1]))
    diagonal = Telum._normalize_localspace_data(sy, ([Z],), ls, os)
    @test diagonal[1] == ws
    @test diagonal[4] === nothing
    @test Telum._normalize_localspace_data(sy, ws, ls, os)[1] === ws
    explicit = Telum._localspace_from_data(sy, diagonal[1:3]...)
    @test Array(to_sparse_array(explicit.S)) ≈ Array(to_sparse_array(reference.S))
    for R in ([cos(0.3) -sin(0.3); sin(0.3) cos(0.3)],
              ComplexF64[1 im; im 1] / sqrt(2))
        zs = ([R * Z * R'],)
        lower = ([R * ls[1][1] * R'],)
        # An imaginary seed phase must survive CGT/RMT factorization.
        phase = eltype(R) <: Real ? 1.0 : im
        operators = Dict(name => R * op * R' * (name == :S ? phase : 1) for (name,op) in os)
        saved = deepcopy((zs, lower, operators))
        q, B, weights, lowering, ops, U = _test_matrix_localspace(sy, zs, lower, operators)
        @test (zs, lower, operators) == saved
        @test U' * U ≈ I
        @test B' * B ≈ I
        @test Array(to_sparse_array(q.I)) ≈ B' * operators[:I] * B
        @test Array(to_sparse_array(q.S))[:,:,1] ≈ B' * operators[:S] * B
        @test eltype(q.S) == (eltype(R) <: Real ? Float64 : ComplexF64)
        for sector in Telum.sector_slots(q.S)
            W = Telum.sector_wmat(q.S, sector, 1)
            @test W' * W ≈ I
        end
    end
end

@testset "commuting Cartan matrices refine degenerate eigenspaces" begin
    # The first generator retains two two-dimensional degenerate eigenspaces.
    z1 = Diagonal([1, 1, -1, -1])
    z2 = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]
    operators = Dict(:I => Matrix{Float64}(I,4,4))
    empty_lops = (Matrix{Float64}[], Matrix{Float64}[])
    q, B, ws, ls, os, U = _test_matrix_localspace((U1,U1), ([(x,) for x in diag(z1)], [z2]), empty_lops, operators)
    @test U' * z1 * U ≈ Diagonal(first.(ws[1]))
    @test U' * z2 * U ≈ Diagonal(first.(ws[2]))
    @test length(unique(zip(ws...))) == 4
    @test Array(to_sparse_array(q.I)) ≈ I
    # Two Cartan components within a valid defining SU(3) representation.
    sy, weights, lowering, ops = getSymmetryInfo(FermionOptions(3, nothing, :SU3))
    d = length(weights[1])
    R = Matrix{Float64}(I,d,d)
    R[2:3,2:3] = [cos(0.2) -sin(0.2); sin(0.2) cos(0.2)]
    zs = ([R * Diagonal([w[j] for w in weights[1]]) * R' for j in 1:2],)
    q, B, ws, ls, os, U = _test_matrix_localspace(sy, zs, ([R*l*R' for l in lowering[1]],), Dict(n => R*o*R' for (n,o) in ops))
    @test Array(to_sparse_array(q.I)) ≈ I
    for j in 1:2
        @test U' * zs[1][j] * U ≈ Diagonal([w[j] for w in ws[1]])
    end
end

@testset "matrix quantum-number input validation" begin
    ops = Dict(:I => Matrix{Float64}(I,2,2))
    empty_lops = (Matrix{Float64}[],)
    for bad in ([0.0 1; 0 0], [0.5 0; 0 1], [NaN 0; 0 1], zeros(2,3), zeros(0,0))
        @test_throws ArgumentError Telum._normalize_localspace_data((U1,), ([bad],), empty_lops, ops)
    end
    @test_throws ArgumentError Telum._normalize_localspace_data((U1,), (Matrix{Float64}[],), empty_lops, ops)
    @test_throws ArgumentError Telum._normalize_localspace_data((U1,), ([zeros(2,2)],), ([zeros(2,2)],), ops)
    @test_throws ArgumentError Telum._normalize_localspace_data((U1,), ([zeros(2,2)],), empty_lops, Dict(:bad=>zeros(3,3)))
    @test_throws ArgumentError Telum._normalize_localspace_data((U1,U1), ([Float64[1 0;0 -1]], [Float64[0 1;1 0]]), (empty_lops...,empty_lops...), ops)
    @test_throws ArgumentError Telum._normalize_localspace_data((U1,SU{2}), ([Float64[1 0;0 -1]], [(1,),(-1,)]), (Matrix{Float64}[], [Float64[0 0;1 0]]), ops)
    for (delta, accepted) in ((2e-13,true), (2e-12,false))
        input = ([Matrix(Diagonal([1+delta,-1.0]))],)
        if accepted
            @test first(Telum._normalize_localspace_data((U1,), input, empty_lops, ops)) == ([(1,),(-1,)],)
        else
            @test_throws ArgumentError Telum._normalize_localspace_data((U1,), input, empty_lops, ops)
        end
    end
    seed = [1e-16 1.0; 0 0]
    @test LurCGT.getweight_irop(U1, [(1,),(-1,)], Matrix{Float64}[], seed) == (2,)
    @test_throws ArgumentError LurCGT.getweight_irop(U1, [(1,),(-1,)], Matrix{Float64}[], zeros(2,2))
end

@testset "SO(2) circular fermions retain the public operators" begin
    basic = Telum.FermionS_basicops(2)
    for channs in ([1,2], [2,1]), spin_symmetry in (:SU2, nothing)
        opts = FermionSOptions(2, :U1, spin_symmetry, [(:SO2,channs)])
        sy, w, l, o = getSymmetryInfo(opts)
        @test sy == (spin_symmetry === :SU2 ? (U1,SU{2},U1) : (U1,U1))
        internal, B, ws, ls, os, U = _test_matrix_localspace(sy, w, l, o)
        public = getLocalSpace(opts)
        @test !any(n -> startswith(String(n), "_so2_"), keys(public))
        @test all(q -> q isa TLArray, values(public))
        a,b = channs
        Z = Telum.FermionS_L3(basic.FF[:,channs])
        @test U' * Z * U ≈ Diagonal(first.(ws[end]))
        if spin_symmetry === :SU2
            @test Array(to_sparse_array(public.S))[:,:,1] ≈ B' * o[:S] * B
        end
        for spin in 1:2
            fplus = (basic.FF[spin,a] - im*basic.FF[spin,b])/sqrt(2)
            fminus = (basic.FF[spin,a] + im*basic.FF[spin,b])/sqrt(2)
            @test LurCGT.comm(Z,fplus) ≈ fplus
            @test LurCGT.comm(Z,fminus) ≈ -fminus
            name = spin_symmetry === :SU2 ? :F : (spin == 1 ? :Fu : :Fd)
            F = getproperty(public,name)
            @test eltype(F) == ComplexF64
            @test Telum.spaces(F)[1:2] == Telum.spaces(public.I)
            offsets,_ = Telum._sparse_leg_offsets(sy, Telum.spaces(F)[3])
            pluskey = only(filter(q -> q[end] == (1,), collect(keys(offsets))))
            minuskey = only(filter(q -> q[end] == (-1,), collect(keys(offsets))))
            # The existing canonical spinor basis orders (c_down, c_up).
            component = spin_symmetry === :SU2 && spin == 1 ? 2 : 1
            dense = Array(to_sparse_array(F))
            plus = dense[:,:,offsets[pluskey][component]]
            minus = dense[:,:,offsets[minuskey][component]]
            @test plus ≈ B' * fplus * B
            @test minus ≈ B' * fminus * B
            ca = (plus+minus)/sqrt(2)
            cb = im*(plus-minus)/sqrt(2)
            @test ca ≈ B' * basic.FF[spin,a] * B
            @test cb ≈ B' * basic.FF[spin,b] * B
            @test ca' ≈ B' * basic.FF[spin,a]' * B
            @test cb' ≈ B' * basic.FF[spin,b]' * B
            @test ca*ca' + ca'*ca ≈ I
            @test cb*cb' + cb'*cb ≈ I
            @test norm(ca*cb' + cb'*ca) < 1e-12
            if spin_symmetry === :SU2
                for sector in Telum.sector_slots(F)
                    W = Telum.sector_wmat(F,sector,2)
                    @test W' * W ≈ I
                end
            end
        end
    end
    grouped = getLocalSpace(FermionSOptions(3,:U1,:SU2,[(:SO2,[1,3])]), ("ket","bra","op"))
    @test hasproperty(grouped,:F13) && hasproperty(grouped,:F2)
    @test Telum.inds(grouped.F13) == (TLIndex("ket",'+'),TLIndex("bra",'-'),TLIndex("op",'-'))
    for (charge, spin) in ((:SU2,:SU2), (:U1,:U1))
        combined = getLocalSpace(FermionSOptions(2,charge,spin,:SO2))
        @test !any(n -> startswith(String(n), "_so2_"), keys(combined))
        @test hasproperty(combined, spin == :SU2 ? :F : :Fu)
    end
    only_channel = getLocalSpace(FermionSOptions(2,nothing,nothing,:SO2))
    @test symm(only_channel.I) == (U1,)
    @test hasproperty(only_channel,:Fu) && hasproperty(only_channel,:Fd)
    for channels in ([1], [1,2,3], [1,1], [1,3])
        @test_throws ArgumentError getLocalSpace(FermionSOptions(2,:U1,:SU2,[(:SO2,channels)]))
    end
end
