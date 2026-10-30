# Module-boundary check (implementation plan, Section 2.2). A module may reference only the
# modules it is allowed to depend on. Inside BUG the layers are include directories, not modules,
# so the second testset checks names instead: `BUG/Core` never uses a name that only
# `Randomised`, `TimeEvolution`, `ITE` or `solve.jl` define (the sweeps reach the randomised
# expansion through a `BasisExpansion`).

using Test

const SRC_DIR = normpath(joinpath(@__DIR__, "..", "..", "src"))

const FORBIDDEN_REFERENCES = Dict(
    "BUG" => ["QMB", "QTTS", "Baselines"],
    "QMB" => ["QTTS", "Baselines"],
    "QTTS" => ["QMB", "Baselines"],
)

# Matches `using`/`import` lines naming a forbidden module, or a qualified access `Module.name`.
function reference_pattern(modules)
    names = join(modules, "|")
    return Regex("\\b(?:using|import)\\b[^\\n]*\\b(?:$names)\\b|\\b(?:$names)\\.[A-Za-z_]")
end

function julia_files(dir)
    files = String[]
    for (root, _, names) in walkdir(dir), name in names
        endswith(name, ".jl") && push!(files, joinpath(root, name))
    end
    return files
end

@testset "module boundaries" begin
    @testset "pattern catches forbidden references" begin
        pattern = reference_pattern(["QMB"])
        @test occursin(pattern, "using ..QMB")
        @test occursin(pattern, "import ..QMB: xx_mpo")
        @test occursin(pattern, "H = QMB.xx_mpo(L)")
        @test !occursin(pattern, "using ..BUG")
        @test !occursin(pattern, "QMBState = 1")
    end

    for (owner, forbidden) in FORBIDDEN_REFERENCES
        pattern = reference_pattern(forbidden)
        for file in julia_files(joinpath(SRC_DIR, owner))
            offending = match(pattern, read(file, String))
            offending === nothing ||
                @error "Forbidden cross-module reference" owner file offending.match
            @test offending === nothing
        end
    end
end

# ── layering inside BUG ──
# Names are read from the parsed code, so docstrings and comments never count as a use.

"Strip `where` clauses and return-type annotations off a definition's left-hand side."
function strip_signature(lhs)
    while lhs isa Expr && lhs.head in (:where, :(::))
        lhs = lhs.args[1]
    end
    return lhs
end

"The bare name of a type in a `struct`/`abstract type` header (`Name`, `Name{T}`, `Name <: Super`)."
function type_name(header)
    while header isa Expr && header.head in (:<:, :curly)
        header = header.args[1]
    end
    return header isa Symbol ? [header] : Symbol[]
end

"""
The names a top-level expression defines. A qualified method (`Base.length(...)`, `BUG.f(...)`)
extends another module's function, so it defines no new name here.
"""
function defined_names(expr)
    expr isa Expr || return Symbol[]
    # A docstring or a macro (`@inline`, `Base.@kwdef`) wraps the definition: look inside.
    expr.head === :macrocall && return defined_names(expr.args[end])
    expr.head === :struct && return type_name(expr.args[2])
    expr.head === :abstract && return type_name(expr.args[1])
    expr.head === :const && return defined_names(expr.args[1])
    expr.head in (:function, :(=)) || return Symbol[]
    lhs = strip_signature(expr.args[1])
    lhs isa Symbol && return [lhs]                                  # a global or `function f end`
    lhs isa Expr && lhs.head === :tuple && return filter(arg -> arg isa Symbol, lhs.args)
    lhs isa Expr && lhs.head === :call || return Symbol[]
    callee = lhs.args[1]
    callee isa Expr && callee.head === :curly && (callee = callee.args[1])   # `T{P}(...)`
    return callee isa Symbol ? [callee] : Symbol[]
end

"Every name defined at the top level of the given files."
function defined_names_in(files)
    names = Set{Symbol}()
    for file in files, expr in Meta.parseall(read(file, String)).args
        union!(names, defined_names(expr))
    end
    return names
end

"""
Every symbol a file's code mentions. Field names and quoted symbols (`a.b`, `:b`) parse to
`QuoteNode`s, so they are not counted as names.
"""
function used_symbols(file)
    found = Set{Symbol}()
    collect!(x::Symbol) = push!(found, x)
    collect!(x::Expr) = foreach(collect!, x.args)
    collect!(x) = nothing
    collect!(Meta.parseall(read(file, String)))
    return found
end

@testset "BUG/Core uses no name of a higher BUG layer" begin
    bug_dir = joinpath(SRC_DIR, "BUG")
    core_files = julia_files(joinpath(bug_dir, "Core"))
    core_names = defined_names_in(core_files)
    higher_files = vcat(
        julia_files(joinpath(bug_dir, "Randomised")),
        julia_files(joinpath(bug_dir, "TimeEvolution")),
        julia_files(joinpath(bug_dir, "ITE")),
        [joinpath(bug_dir, "solve.jl")],
    )
    # A method a higher layer adds to a Core function (`apply_one_site(op::RootOp, ...)`) is Core's name.
    higher_only = setdiff(defined_names_in(higher_files), core_names)
    @test :dd_cbe_expand in higher_only && :cbe_bug_step! in higher_only     # the check is not vacuous
    for file in core_files
        offending = intersect(used_symbols(file), higher_only)
        isempty(offending) || @error "BUG/Core uses a name of a higher layer" file offending =
            sort!(collect(offending))
        @test isempty(offending)
    end
end
