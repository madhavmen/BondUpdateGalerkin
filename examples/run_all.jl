# Run every example at small sizes, each in its own process; exit non-zero if any fails.
#
#   julia --project examples/run_all.jl
#
# Every example checks its own answer and exits non-zero on failure. Results go to a temporary
# directory, never into the source tree. Used by the manual pre-commit hook `julia-examples`.

const EXAMPLES_DIR = @__DIR__
const PROJECT_DIR = dirname(EXAMPLES_DIR)

output_dir = mktempdir()
# (script, extra arguments after the output directory)
examples = [
    ("QMB/xx_quickstart.jl", ["12"]),                   # chain length 12
    ("QMB/xx_domain_wall.jl", ["12", "2.0"]),           # chain length 12, final time 2
    ("QTTS/heston_quickstart.jl", String[]),
    ("QTTS/gp_ground_state.jl", String[]),
]

failed = String[]
for (script, arguments) in examples
    println("== $script")
    script_path = joinpath(EXAMPLES_DIR, script)
    command = `$(Base.julia_cmd()) --project=$PROJECT_DIR $script_path $output_dir $arguments`
    success(pipeline(command; stdout = stdout, stderr = stderr)) || push!(failed, script)
end

isempty(failed) || (println("FAILED: ", join(failed, ", ")); exit(1))
println("all $(length(examples)) examples passed")
