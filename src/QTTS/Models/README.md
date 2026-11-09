# `src/QTTS/Models/`

The two PDEs the package solves on QTT. Each model is an operator (or a nonlinear flow) built from
`../Tooling/`, plus an independent reference.

| folder | contents |
|---|---|
| [`Heston/`](Heston/README.md) | The Heston option-pricing generator: a linear flow, stepped by `qtt_bug_step!` with the rational root. |
| [`GrossPitaevskii/`](GrossPitaevskii/README.md) | The damped Gross-Pitaevskii equation on a periodic 2D grid: a nonlinear flow with a moving paddle, its ground states and observables. |
