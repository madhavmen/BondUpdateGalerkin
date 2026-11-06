# Telum.jl

`Telum.jl` (TEnsor Library for Universal Many-body simulation) is a non-Abelian symmetric tensor network library.

## Installation

Once this repository is pushed to its own remote, install it with:

```julia
using Pkg
Pkg.add(url="<Telum-repo-url>")
```

Because `Telum.jl` depends on `LurCGT.jl`, local development typically looks
like this:

```julia
using Pkg
Pkg.develop(path="../LurCGT.jl")
Pkg.develop(path=".")
```

Typical usage:

```julia
using LurCGT
using Telum
```

The proposed eager-default, opt-in lazy-evaluation design is documented in
[docs/LAZY_EVALUATION_PLAN.md](docs/LAZY_EVALUATION_PLAN.md).

## Testing

```julia
using Pkg
Pkg.develop(path="../LurCGT.jl")
Pkg.test()
```
