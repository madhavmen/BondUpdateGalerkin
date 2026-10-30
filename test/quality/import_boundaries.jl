# Module-boundary check (implementation plan, Section 2.2). A module may reference only the
# modules it is allowed to depend on. Layering inside BUG (Core <- Randomised <- TimeEvolution,
# ITE) is not visible at module level and is enforced in review.

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
