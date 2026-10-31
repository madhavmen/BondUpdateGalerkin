# Build the documentation: julia --project=docs docs/make.jl (after Pkg.instantiate in docs/).
# The site is internal, so it is built and checked in CI but not deployed.

using Documenter
using BondUpdateGalerkin
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.QTTS
using BondUpdateGalerkin.Baselines

makedocs(;
    sitename = "BondUpdateGalerkin",
    modules = [BondUpdateGalerkin, BUG, QMB, QTTS, Baselines],
    format = Documenter.HTML(; prettyurls = false),
    remotes = nothing,
    doctest = true,
    # Many internal names are exported without a docstring yet, and docstrings cite names that are
    # not on any page; both are reported but do not fail the build. Doctest failures still fail it.
    warnonly = [:missing_docs, :cross_references, :docs_block],
    pages = [
        "Home" => "index.md",
        "Method" => "theory.md",
        "Examples" => "examples.md",
        "API" => [
            "BUG" => "api/bug.md",
            "QMB" => "api/qmb.md",
            "QTTS" => "api/qtts.md",
            "Baselines" => "api/baselines.md",
        ],
        "Developer architecture" => "developer_architecture.md",
    ],
)
