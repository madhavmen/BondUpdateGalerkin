"""
    QTTS

Quantics tensor trains (QTT) for PDEs, evolved by the BUG engine.

Include directories:

- `Tooling/`: grids and bit orderings, QTT construction, MPO algebra, derivative MPOs and the
  adapter to the `BUG` state interface.
- `Elementwise/`: pointwise operations on QTT functions (Hadamard products, `|u|^2`).
- `Models/Heston/`, `Models/GrossPitaevskii/`: equations, parameters, initial data, references
  and observables.
"""
module QTTS

using ..BUG

end
