"""
    Baselines

Independent comparison methods: TDVP (`TDVP/`), DMRG for ground states (`DMRG/`) and small exact
references (`exact.jl`). They may use `BUG`'s `Core` and `Randomised` layers, never its
`TimeEvolution` or `ITE` layers, so that a comparison never runs the method it is compared with.
"""
module Baselines

using ..BUG

end
