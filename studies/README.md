# Studies

Research code that is not part of the package: exploratory geometries, campaign drivers and their
analysis. Nothing here is loaded by `using BondUpdateGalerkin`.

The study code of BUG-Julia `b155550` is archived, not copied: the models `kagome.jl`,
`breathing_kagome.jl`, `square_cylinder.jl`, `triangular_cylinder.jl` and `haldane_shastry.jl`
(with their two tests), the campaign benchmarks marked `study`, the Slurm wrappers
`scripts/run_julia.sbatch` and `scripts/run_python.sbatch`, and the notebooks. They run as they
are at the BUG-Julia tag `pre-bondupdategalerkin` with its pinned `Manifest.toml`. A study that
resumes is ported here against the public API (`mpo_from_terms` with the model's local identity,
`QMB` for states and observables), with a README stating its question, inputs and outputs, and its
models move into `QMB` only once they are maintained with tests.
