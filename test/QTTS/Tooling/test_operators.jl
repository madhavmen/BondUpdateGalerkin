# Dense against TT operators on small 1D and 2D grids, in grouped and interleaved layouts
# and the BUG MPO applying the same matrix as the dense cores.
#
# The references are written row by row on the grid index (axis 1 fastest) and share no code with
# the core algebra: a wrong placement by the ordering shows up as a wrong matrix.

using Test, LinearAlgebra
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QTTS

include(joinpath(@__DIR__, "..", "..", "common", "qtt_fixtures.jl"))

set_symmetry!(:none)
clear_fusion_cache!()

const LAYOUTS = (
    (:grouped, :lsb_first),
    (:grouped, :msb_first),
    (:interleaved, :lsb_first),
    (:interleaved, :msb_first),
)

"The periodic forward shift `(S v)[i] = v[i + 1 mod N]`, written row by row."
function ref_shift(N)
    S = zeros(N, N)
    for i = 1:N
        S[i, mod1(i + 1, N)] = 1.0
    end
    return S
end

"The same shift as an operator chain, one `|r><c|` per entry (no shift construction)."
shift_chain(n) = op_lincomb([(1.0, op_ket_bra(n, r, mod(r + 1, 2^n))) for r = 0:(2^n-1)])

"`Ax (x) Ay` on the grid index `ix + 1 + Nx iy` (`nothing` = identity)."
function ref2(g, Ax, Ay)
    Nx, Ny = 2^g.nx, 2^g.ny
    return kron(
        Ay === nothing ? Matrix(1.0I, Ny, Ny) : Ay,
        Ax === nothing ? Matrix(1.0I, Nx, Nx) : Ax,
    )
end

@testset "1D: shift and diagonal against dense ($o, $e)" for (o, e) in LAYOUTS
    for n in (3, 4)
        ord = QTTOrdering((n,); order = o, endian = e)
        N = 2^n
        S = qtt_embed(ord; x = shift_chain(n))
        @test _relerr(_grid_matrix(ord, op_dense(S)), ref_shift(N)) < 1e-12
        v = collect(1.0:N) .^ 2
        D = qtt_embed(ord; x = op_diag(tt_from_samples(v)))
        @test _relerr(_grid_matrix(ord, op_dense(D)), Diagonal(v)) < 1e-12
    end
end

@testset "2D: embedded operators against dense ($o, $e)" for (o, e) in LAYOUTS
    for (nx, ny) in ((2, 3), (3, 3))
        g = qtt_grid(nx, ny; x = (-2.0, 2.0), y = (0.0, 1.0), order = o, endian = e)
        xs, ys = qtt_coords(g)
        Sx, Sy = shift_chain(nx), shift_chain(ny)
        X = op_diag(tt_from_samples(collect(xs)))
        Y = op_diag(tt_from_samples(collect(ys)))
        cases = (
            (qtt_embed(g; x = Sx), ref2(g, ref_shift(2^nx), nothing)),
            (qtt_embed(g; y = Sy), ref2(g, nothing, ref_shift(2^ny))),
            (qtt_embed(g; x = X), ref2(g, Diagonal(xs), nothing)),
            (qtt_embed(g; x = Sx, y = Y), ref2(g, ref_shift(2^nx), Diagonal(ys))),
            (qtt_embed(g; x = X, y = Sy), ref2(g, Diagonal(xs), ref_shift(2^ny))),
        )
        for (W, ref) in cases
            @test _relerr(_grid_matrix(g, op_dense(W)), ref) < 1e-12
        end
        # boundary wrap of the periodic shift on grid values
        C = qtt_function(g, (x, y) -> x + 10y + 0.5x * y)
        V = qtt_values(C, g)
        @test _relerr(
            qtt_values(first(tt_compress(op_apply(cases[1][1], C))), g),
            circshift(V, (-1, 0)),
        ) < 1e-12
        @test _relerr(
            qtt_values(first(tt_compress(op_apply(cases[2][1], C))), g),
            circshift(V, (0, -1)),
        ) < 1e-12
        # multiplication by the coordinate field is the diagonal operator
        Xf = op_diag(qtt_separable(g, identity, nothing))
        @test _relerr(_grid_matrix(g, op_dense(Xf)), ref2(g, Diagonal(xs), nothing)) < 1e-12
    end
end

@testset "operator-core algebra" begin
    rng = Random.MersenneTwister(11)
    L = 5
    rand_op(w) = [randn(rng, ComplexF64, k == 1 ? 1 : w, 2, 2, k == L ? 1 : w) for k = 1:L]
    A, B = rand_op(3), rand_op(2)
    Ad, Bd = op_dense(A), op_dense(B)
    C = _random_tt(_tt_ranks(L, 4); rng = rng)
    @test _relerr(vec(tt_dense(op_apply(A, C))), Ad * vec(tt_dense(C))) < 1e-12
    @test _relerr(op_dense(op_compose(A, B)), Ad * Bd) < 1e-12
    @test _relerr(op_dense(op_adjoint(A)), Ad') < 1e-14
    @test _relerr(op_dense(op_transpose(A)), transpose(Ad)) < 1e-14
    @test _relerr(op_dense(op_add(A, B)), Ad + Bd) < 1e-12
    @test _relerr(op_dense(op_lincomb([(2.0, A), (1im, B)])), 2Ad + 1im * Bd) < 1e-12
    @test _relerr(op_dense(op_compress(op_add(A, A))), 2Ad) < 1e-12
    @test maximum(tt_ranks(op_compress(op_add(A, A)))) <= 3
    @test _relerr(op_dense(op_scale(A, -0.5)), -0.5Ad) < 1e-14
    @test op_dense(op_identity(L)) == Matrix(1.0I, 2^L, 2^L)
    K = op_dense(op_ket_bra(L, 6, 19))
    @test K[7, 20] == 1 && sum(abs, K) == 1                  # site 1 = bit 0 = fastest index
end

@testset "BUG MPO applies the dense matrix ($o, $e)" for (o, e) in LAYOUTS
    g = qtt_grid(2, 3; order = o, endian = e)
    L = qtt_length(g)
    xs, ys = qtt_coords(g)
    W = op_lincomb([
        (1.0, qtt_embed(g; x = shift_chain(2), y = op_diag(tt_from_samples(collect(ys))))),
        (0.3im, qtt_embed(g; x = op_diag(tt_from_samples(collect(xs))), y = shift_chain(3))),
    ])
    mpo = qtt_to_mpo(W)
    @test length(mpo) == L
    @test mpo_virtual_dims(mpo) == [1; tt_ranks(W); 1]
    @test _relerr(op_dense(qtt_mpo_cores(mpo)), op_dense(W)) < 1e-14
    for T in (Float64, ComplexF64)
        C = _random_tt(_tt_ranks(L, 4); rng = Random.MersenneTwister(L), T = T)
        out = mpo_product(mpo, qtt_mps(C))
        @test _relerr(vec(tt_dense(qtt_cores(out))), op_dense(W) * vec(tt_dense(C))) < 1e-12
        @test inner_product(qtt_mps(C), out) ≈ tt_dot(C, op_apply(W, C)) rtol = 1e-12
    end
end
