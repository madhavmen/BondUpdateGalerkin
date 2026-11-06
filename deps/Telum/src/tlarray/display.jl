# ─── Pretty-printing for TLArray ──────────────────────────────────────────────
#
# Format:
#
#   TLArray{T}  [Symm1, Symm2, ...]
#     leg 1:  dir  'tag'  (plev=k)
#     leg 2:  dir  'tag'
#     ...
#     1.  <sector inline>
#     2.  <sector inline>
#     ...
#
# When the number of sectors exceeds TLARRAY_DISPLAY_HEAD + TLARRAY_DISPLAY_TAIL,
# only the first TLARRAY_DISPLAY_HEAD and last TLARRAY_DISPLAY_TAIL sectors are shown.
# Set these globals to control the truncation behaviour.
# ─────────────────────────────────────────────────────────────────────────────

const TLARRAY_DISPLAY_HEAD = Ref(5)   # number of first sectors to show
const TLARRAY_DISPLAY_TAIL = Ref(5)   # number of last sectors to show

"""
    show(io::IO, q::TLArray)

Print a compact text representation of a concrete tensor.

The display includes rank, symmetry names, visible index metadata, sector RMT
physical dimensions, outer-multiplicity dimensions, qlabels, and scalar sector
values when available. Sector output is truncated by `TLARRAY_DISPLAY_HEAD` and
`TLARRAY_DISPLAY_TAIL`.
"""
Base.show(io::IO, qs::TLArray) = show(io, MIME"text/plain"(), qs)

"""
    show(io::IO, q::SubTLArray)

Print a lazy subarray through a materialized concrete `TLArray` view.

The concrete view aliases the subarray's materialized storage and preserves its
embedded state. This method is intentionally display-oriented and may compute
lazy sectors.
"""
function Base.show(io::IO, qs::SubTLArray)
    concrete = TLArray(qs)
    show(io, concrete)
    return concrete
end

"""
    _format_qindex(idx::TLIndex) -> String

Format one `TLIndex` for TLArray text display.

The string includes normalized tags, direction, prime level when nonzero, and
lock level when nonzero. The dual flag is handled by the caller with color so
that the raw textual index remains compact.
"""
_qindex_plev_string(plev::Int) =
    plev == 0 ? "" : "p$(plev)"

_qindex_lock_string(lock::Int) = lock == 0 ? "" : "🔒$(lock)"

function _format_qindex(idx::TLIndex)
    return "\"$(idx.itags)$(idx.dir)\"$(_qindex_plev_string(idx.plev))$(_qindex_lock_string(idx.lock))"
end

"""
    _print_tlarray_header(io::IO, q::TLArray)

Print the one-line non-payload TLArray header.

The header contains visible rank, number of symmetries, symmetry text keys, and
formatted visible indices. Dual indices are colored when the terminal supports
ANSI escapes.
"""
function _print_tlarray_header(io::IO, qs::TLArray{T, QD, N}) where {T, QD, N}
    if N == 0
        leg_strs = map(inds(qs)) do idx
            raw = _format_qindex(idx)
            idx.dual ? "\e[32m$(raw)\e[0m" : raw
        end
        size_str = isempty(size(qs)) ? "()" : join(size(qs), "x")
        print(io, size_str, " TLArray, [", join(leg_strs, ", "), "]")
        return
    end

    symm_names = join((totxt(s) for s in symm(qs)), ", ")
    print(io, "$(QD)D TLArray, $N symmetries [$symm_names]")
    leg_strs = map(inds(qs)) do idx
        raw = _format_qindex(idx)
        idx.dual ? "\e[32m$(raw)\e[0m" : raw
    end
    print(io, "  [", join(leg_strs, ", "), "]")
end

function _print_tlarray_header(io::IO, qs::TLArray{T, 0, N}) where {T, N}
    symm_names = join((totxt(s) for s in symm(qs)), ", ")
    print(io, "0D TLArray{$T}, $N symmetries [$symm_names]")
end

"""
    printmeta([io::IO], q::TLArray)

Print only the non-RMT metadata line for `q`, matching the non-numerical prefix
of the standard `TLArray` text display. This lightweight helper is defined only
for concrete `TLArray` values; it reads no RMT payloads and does not materialize
lazy evaluation objects.
"""
function printmeta(io::IO, q::TLArray)
    _print_tlarray_header(io, q)
    println()
end
printmeta(q::TLArray) = printmeta(stdout, q)

# Special pretty-printing for 0-dimensional TLArray (scalar result of full contraction).
"""
    show(io::IO, ::MIME"text/plain", q::TLArray{T,0,N,N})

Pretty-print a scalar TLArray produced by full contraction. A no-symmetry
scalar uses the compact no-symmetry summary and does not append a value.

For tensors with symmetries, the scalar method prints the same header as
higher-rank tensors and then appends the single scalar value instead of a
sector table.
"""
function Base.show(io::IO, ::MIME"text/plain", qs::TLArray{T, 0, N, N}) where {T, N}
    _print_tlarray_header(io, qs)
    N == 0 && return nothing
    print(io, ": ", _fmt_scalar_str(qs[]))
end

"""
    show(io::IO, ::MIME"text/plain", q::TLArray)

Pretty-print a TLArray. Symmetry-aware tensors include sector metadata and
small scalar payload summaries. A tensor without symmetry is shown only as its
shape and logical indices.

`q` is assumed concrete. The method displays active sectors and defined zero
sectors (marked `zero`), while omitting slots that have no RMT payload. It
computes per-symmetry label widths for alignment, marks omitted middle sectors
when the displayed-sector count exceeds the configured head/tail limits, and
avoids printing full dense RMT payloads.
"""
function Base.show(io::IO, ::MIME"text/plain", qs::TLArray{T, QD, N, RD}) where {T, QD, N, RD}
    # --- Header: symmetries and leg dirs/tags on one line ---
    # Format:  TLArray{...}  [Sym1, Sym2]  ["tag1"+, "tag2"-', ...]
    _print_tlarray_header(io, qs)
    N == 0 && return nothing
    println(io)

    # --- Sectors: one per line with global label width for cross-sector alignment ---
    # A defined zero sector has payload and q-label metadata worth displaying;
    # an undefined zero slot represents a sector that does not exist.
    displayable_indices = [i for i in sector_slots(qs)
                           if !qs.iszero[i] || qs.isdefined[i]]
    nr = length(displayable_indices)
    if nr == 0
        print(io, "  (empty)")
        return
    end

    # Determine which sector indices to display.
    head = TLARRAY_DISPLAY_HEAD[]
    tail = TLARRAY_DISPLAY_TAIL[]
    truncate = nr > head + tail
    display_indices = if truncate
        vcat(displayable_indices[1:head], displayable_indices[nr-tail+1:nr])
    else
        displayable_indices
    end

    # Compute per-symmetry widths globally across displayed sectors only.
    widths = map(1:N) do n
        maximum(display_indices) do i
            _sector_label_widths(qs, i)[n]
        end
    end
    # Pre-compute scalar width for alignment (only sectors with scalar RMT).
    scalar_width = 0
    for i in display_indices
        qs.iszero[i] && begin
            scalar_width = max(scalar_width, length("zero"))
            continue
        end
        rmt, _ = sector_rmt(qs, i)
        length(rmt) == 1 || continue
        scalar_width = max(scalar_width, length(_fmt_scalar_str(_display_scalar_rmt(qs, i))))
    end

    for (k, i) in enumerate(display_indices)
        # Print ellipsis between head and tail blocks.
        if truncate && k == head + 1
            println(io)
            println(io, "  ⋮  ($(nr - head - tail) sectors omitted)")
        end
        rmt, _ = sector_rmt(qs, i)
        om_dim = prod(size(rmt)[QD+1:end]; init=1)
        phys_str = join(sector_rmt_dim(qs, i)[1:QD], "x")
        om_str   = om_dim > 1 ? " @$om_dim" : ""
        print(io, "  $i.\t", phys_str, om_str, "\t")
        _print_sector_cgt_dims(io, qs, i)
        _print_sector_qlabels(io, qs, i, widths)
        qs.iszero[i] ? print(io, "\t", lpad("zero", scalar_width)) : length(rmt) == 1 &&
            print(io, "\t", lpad(_fmt_scalar_str(_display_scalar_rmt(qs, i)), scalar_width))
        QD == 2 && print(io, "\t√", _sector_cgt_size_2d(qs, i))
        k < length(display_indices) && println(io)
    end
end
