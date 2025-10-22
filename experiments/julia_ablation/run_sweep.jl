#!/usr/bin/env julia

using Dates
using TOML

include("../../native/julia/SpiralTransformerAblation.jl")
using .SpiralTransformerAblation

function load_defaults(cfg)
    return AblationSpec(
        d_model = Int(get(cfg, "d_model", 128)),
        n_layers = Int(get(cfg, "n_layers", 4)),
        n_heads = Int(get(cfg, "n_heads", 4)),
        ff_multiplier = Float64(get(cfg, "ff_multiplier", 4.0)),
        seq_len = Int(get(cfg, "seq_len", 32)),
        gate_scale = Float64(get(cfg, "gate_scale", 1.0)),
        seed = Int(get(cfg, "seed", 2025)),
        repeat = Int(get(cfg, "repeat", 1)),
    )
end

function expand_overrides(grid, repeats)
    overrides = Vector{Dict{String,Any}}()
    for (index, base_override) in enumerate(grid)
        for r in 1:repeats
            entry = Dict{String,Any}(base_override)
            entry["repeat"] = r
            entry["seed_offset"] = (index - 1) * repeats + (r - 1)
            push!(overrides, entry)
        end
    end
    return overrides
end

function write_results(path::AbstractString, rows)
    isempty(rows) && return
    headers = collect(propertynames(first(rows)))
    open(path, "w") do io
        println(io, join(string.(headers), ","))
        for row in rows
            values = [getfield(row, header) for header in headers]
            println(io, join(string.(values), ","))
        end
    end
end

function main(args)
    config_path = length(args) >= 1 ? args[1] : joinpath(@__DIR__, "sweep_config.toml")
    config = TOML.parsefile(config_path)
    defaults_cfg = get(config, "defaults", Dict{String,Any}())
    parameters_cfg = get(config, "parameters", Dict{String,Any}())
    experiment_cfg = get(config, "experiment", Dict{String,Any}())

    base = load_defaults(defaults_cfg)
    grid = parameter_grid(Dict{String,Any}(parameters_cfg))
    repeats = Int(get(experiment_cfg, "repeats", 1))
    overrides = expand_overrides(grid, repeats)
    results = run_parameter_sweep(overrides; base = base)

    results_dir_raw = get(experiment_cfg, "results_dir", "../../results/ablation")
    results_dir = normpath(joinpath(@__DIR__, results_dir_raw))
    mkpath(results_dir)
    prefix = get(experiment_cfg, "file_prefix", "ablation")
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    output_path = joinpath(results_dir, "$(prefix)_$(timestamp).csv")
    write_results(output_path, results)
    println("Wrote $(length(results)) rows to $(output_path)")
end

main(ARGS)
