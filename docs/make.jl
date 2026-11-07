# Build the documentation: julia --project=docs docs/make.jl (after Pkg.instantiate in docs/).
# The site is internal, so it is built and checked locally (pre-commit hook `julia-docs`) but not
# deployed.

using Documenter
using BondUpdateGalerkin
using BondUpdateGalerkin.BUG, BondUpdateGalerkin.QMB, BondUpdateGalerkin.QTTS
using BondUpdateGalerkin.Baselines

makedocs(;
    sitename = "BondUpdateGalerkin",
    modules = [BondUpdateGalerkin, BUG, QMB, QTTS, Baselines],
    # The API pages list every exported docstring of a submodule on one page (api/bug.md is over
    # Documenter's default 200 KiB limit); the site is internal, so page size is not a concern.
    format = Documenter.HTML(; prettyurls = false, size_threshold = nothing),
    remotes = nothing,
    doctest = true,
    # Many internal names are exported without a docstring yet, and docstrings cite names that are
    # not on any page; both are reported but do not fail the build. Doctest failures still fail it.
    warnonly = [:missing_docs, :cross_references, :docs_block],
    pages = [
        "Home" => "index.md",
        "Method" => "theory.md",
        "Matrix product states" => "mps.md",
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
