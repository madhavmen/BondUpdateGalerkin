# The basis-selection strategy the Core sweeps call.
#
# Core never names a sketch. The half-sweeps and the frontier sweeps receive a `BasisExpansion` and
# call only the four functions below on it; `BUG/Randomised` defines the strategies and adds their
# methods. CBE is therefore a strategy, not a fork of the sweeps: another basis selection adds its
# own subtype and methods without touching `Core/Sweeps`.

"""
    BasisExpansion

A basis-selection (enrichment) strategy for the bond frames of a BUG sweep. A subtype implements
[`expand_frame`](@ref), [`sketch_closures`](@ref), [`folded_sketch_closures`](@ref) and
[`sampled_leak`](@ref). `BUG/Randomised` provides `DDQRExpansion`, the default of
`cbe_bug_step!`.
"""
abstract type BasisExpansion end

"""
    expand_frame(strategy, frame, mpo, bond, lch, rch, side; kwargs...) -> CBEExpansion

`side`'s frame of the bond frame `frame` at bond `bond` (environments `lch`, `rch`) grown by the
strategy. The keywords are the step's expansion settings (allocation, ledger, RNG, work ceiling,
counters); the result carries the grown frames and the number of directions admitted.
"""
function expand_frame end

"""
    sketch_closures(strategy, frame, build_action) -> (sketch_left, sketch_right)

The two sketch closures of bond frame `frame` on the two-site action that `build_action()` forms.
"""
function sketch_closures end

"""
    folded_sketch_closures(strategy, frame, mpo, bond, lch, rch) -> (sketch_left, sketch_right)

As [`sketch_closures`](@ref), with the probe folded into the environment contractions so the
two-site action is never formed.
"""
function folded_sketch_closures end

"""
    sampled_leak(strategy, frame, sketch_left, sketch_right, side, probe_count, rng) -> Float64

An estimate, from `probe_count` probe columns, of how much of the two-site action `side`'s frame
misses.
"""
function sampled_leak end
