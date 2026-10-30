# Quantics bit orderings and uniform 2D grids.
#
# Moved from BUG-Julia@b155550:src/RSVDCBEBondUpdate/qtt/qtt_grid.jl. The site map is split out of
# `QTTGrid` into `QTTOrdering` (implementation plan WP2), which also covers one axis or more than two.
#
# A field on a grid of `2^n_1 x ... x 2^n_N` points is a tensor train of `sum n_a` sites, one site per
# bit; bit value `b` of a site is its local index `b + 1`. The default ordering is GROUPED (every bit
# of axis 1 before every bit of axis 2) with each coordinate LEAST significant bit first,
#   i_x = sum_k 2^k b_{x,k},   site k + 1 holds b_{x,k},   site nx + k + 1 holds b_{y,k}.
# `:interleaved` (x_1 y_1 x_2 y_2 ...) and `:msb_first` exist as diagnostics: grouped and interleaved
# are NOT equivalent at a fixed attainable rank. For Heston the second coordinate is the variance.

"""
    QTTOrdering(bits; axes, order = :grouped, endian = :lsb_first)

Where every bit of every axis sits in the tensor train. `bits[a]` is the number of bits of axis `a`
(`2^bits[a]` points), `axes[a]` its name (`:x`, `:y`, ... by default). `order` is `:grouped` (all
bits of axis 1, then axis 2, ...) or `:interleaved` (bit `j` of every axis before bit `j + 1`; an
axis with fewer bits drops out once it runs out). `endian` is `:lsb_first` or `:msb_first` within
each axis.

`sites[a][k + 1]` is the site holding bit `k` (`k = 0` the least significant) of axis `a`'s index.
"""
struct QTTOrdering{N}
    axes::NTuple{N,Symbol}
    bits::NTuple{N,Int}
    order::Symbol
    endian::Symbol
    sites::NTuple{N,Vector{Int}}
end

const _DEFAULT_AXES = (:x, :y, :z)

function QTTOrdering(
    bits::NTuple{N,Integer};
    axes::NTuple{N,Symbol} = N <= 3 ? _DEFAULT_AXES[1:N] : ntuple(a -> Symbol(:x, a), N),
    order::Symbol = :grouped,
    endian::Symbol = :lsb_first,
) where {N}
    N >= 1 || throw(ArgumentError("QTTOrdering needs at least one axis"))
    all(>=(1), bits) || throw(ArgumentError("every axis needs at least one bit, got $bits"))
    allunique(axes) || throw(ArgumentError("axis names must be distinct, got $axes"))
    order in (:grouped, :interleaved) ||
        throw(ArgumentError("order must be :grouped or :interleaved, got :$order"))
    endian in (:lsb_first, :msb_first) ||
        throw(ArgumentError("endian must be :lsb_first or :msb_first, got :$endian"))
    nb = Int.(bits)
    return QTTOrdering{N}(axes, nb, order, endian, _qtt_site_map(nb, order, endian))
end

"Site of every bit: slot `j` of an axis holds bit `j - 1` (`:lsb_first`) or `n - j` (`:msb_first`)."
function _qtt_site_map(bits::NTuple{N,Int}, order::Symbol, endian::Symbol) where {N}
    bit(j, n) = endian === :lsb_first ? j - 1 : n - j
    sites = ntuple(a -> zeros(Int, bits[a]), N)
    s = 0
    if order === :grouped
        for a = 1:N, j = 1:bits[a]
            sites[a][bit(j, bits[a])+1] = (s += 1)
        end
    else
        for j = 1:maximum(bits), a = 1:N
            j <= bits[a] && (sites[a][bit(j, bits[a])+1] = (s += 1))
        end
    end
    return sites
end

"Number of quantics sites, `sum(bits)`."
qtt_length(o::QTTOrdering) = sum(o.bits)

"The sites of axis `axis` (a name or a position), bit `k` at entry `k + 1`."
qtt_sites(o::QTTOrdering, axis::Integer) = o.sites[axis]
function qtt_sites(o::QTTOrdering, axis::Symbol)
    a = findfirst(==(axis), o.axes)
    a === nothing && throw(ArgumentError("no axis :$axis in $(o.axes)"))
    return o.sites[a]
end

"""
    qtt_bits(o, idx...) -> Vector{Int}

The bit value (0 or 1) on every site for the grid point with 0-based axis indices `idx`.
"""
function qtt_bits(o::QTTOrdering{N}, idx::Vararg{Integer,N}) where {N}
    all(0 <= idx[a] < 2^o.bits[a] for a = 1:N) ||
        throw(BoundsError(ntuple(a -> 2^o.bits[a], N), idx))
    b = zeros(Int, qtt_length(o))
    for a = 1:N, k = 0:(o.bits[a]-1)
        b[o.sites[a][k+1]] = (idx[a] >> k) & 1
    end
    return b
end

"""
    qtt_point(o, bits) -> NTuple{N,Int}

The 0-based axis indices of a site configuration, the inverse of [`qtt_bits`](@ref).
"""
function qtt_point(o::QTTOrdering{N}, bits::AbstractVector{<:Integer}) where {N}
    length(bits) == qtt_length(o) || throw(DimensionMismatch("need $(qtt_length(o)) bits"))
    return ntuple(a -> sum(bits[o.sites[a][k+1]] << k for k = 0:(o.bits[a]-1)), N)
end

"""
    qtt_site_permutation(o) -> Vector{Int}

`p[c]` is the 1-based grid index `1 + i_1 + 2^n_1 (i_2 + 2^n_2 (...))` (axis 1 fastest) of site
configuration `c`, where `c - 1 = sum_s 2^(s-1) b_s` (site 1 fastest, the column-major order of a
dense `(2, ..., 2)` core contraction). For dense comparisons on small grids only.
"""
function qtt_site_permutation(o::QTTOrdering{N}) where {N}
    L = qtt_length(o)
    L <= 24 || throw(ArgumentError("qtt_site_permutation is a small-grid check (L = $L)"))
    strides = ntuple(a -> prod((2^o.bits[b] for b = 1:(a-1)); init = 1), N)
    p = zeros(Int, 2^L)
    bits = zeros(Int, L)
    for c = 0:(2^L-1)
        for s = 1:L
            bits[s] = (c >> (s - 1)) & 1
        end
        p[c+1] = 1 + sum(qtt_point(o, bits) .* strides)
    end
    return p
end

"""
    QTTGrid

A uniform 2D grid `x_i = x0 + i dx`, `y_j = y0 + j dy`, `i < 2^nx`, `j < 2^ny`, and its bit
[`QTTOrdering`](@ref) (axes `:x`, `:y`). `periodic_*` records which boundary the grid spacing was
chosen for (`dx = (xmax - xmin) / 2^nx` periodic, `/(2^nx - 1)` with both end points on the grid).
"""
struct QTTGrid
    nx::Int
    ny::Int
    x0::Float64
    dx::Float64
    y0::Float64
    dy::Float64
    periodic_x::Bool
    periodic_y::Bool
    ordering::QTTOrdering{2}
end

"""
    qtt_grid(nx, ny; x = (-1, 1), y = (-1, 1), periodic_x = true, periodic_y = true,
             order = :grouped, endian = :lsb_first) -> QTTGrid

The grid on `x[1] <= x < x[2]` (periodic) or `x[1] <= x <= x[2]` (end points on the grid), likewise
in `y`. `order` and `endian` are those of [`QTTOrdering`](@ref).
"""
function qtt_grid(
    nx::Int,
    ny::Int;
    x = (-1.0, 1.0),
    y = (-1.0, 1.0),
    periodic_x::Bool = true,
    periodic_y::Bool = true,
    order::Symbol = :grouped,
    endian::Symbol = :lsb_first,
)
    nx >= 1 && ny >= 1 || throw(ArgumentError("qtt_grid needs at least one bit per coordinate"))
    x[2] > x[1] && y[2] > y[1] || throw(ArgumentError("empty domain $x x $y"))
    o = QTTOrdering((nx, ny); axes = (:x, :y), order = order, endian = endian)
    dx = (x[2] - x[1]) / (periodic_x ? 2^nx : 2^nx - 1)
    dy = (y[2] - y[1]) / (periodic_y ? 2^ny : 2^ny - 1)
    return QTTGrid(nx, ny, Float64(x[1]), dx, Float64(y[1]), dy, periodic_x, periodic_y, o)
end

qtt_length(g::QTTGrid) = qtt_length(g.ordering)
qtt_sites(g::QTTGrid, axis) = qtt_sites(g.ordering, axis)
qtt_bits(g::QTTGrid, ix::Integer, iy::Integer) = qtt_bits(g.ordering, ix, iy)
qtt_point(g::QTTGrid, bits::AbstractVector{<:Integer}) = qtt_point(g.ordering, bits)
qtt_site_permutation(g::QTTGrid) = qtt_site_permutation(g.ordering)

"Grid coordinates `(x, y)` as two vectors (length `2^nx`, `2^ny`)."
qtt_coords(g::QTTGrid) = (g.x0 .+ g.dx .* (0:(2^g.nx-1)), g.y0 .+ g.dy .* (0:(2^g.ny-1)))

"The quadrature weight `dx dy` of the discrete L2 inner product."
qtt_cell(g::QTTGrid) = g.dx * g.dy
