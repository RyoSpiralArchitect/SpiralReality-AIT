module SpiralTransformerAblation

export AblationSpec, parameter_grid, run_parameter_sweep

include("SpiralTransformerJulia.jl")
using .SpiralTransformerJulia

using LinearAlgebra
using Random
using Statistics
using Base.Iterators

Base.@kwdef struct AblationSpec
    d_model::Int = 128
    n_layers::Int = 4
    n_heads::Int = 4
    ff_multiplier::Float64 = 4.0
    seq_len::Int = 32
    gate_scale::Float64 = 1.0
    seed::Int = 2025
    repeat::Int = 1
end

function _merge_spec(base::AblationSpec, overrides::Dict{String,Any})
    return AblationSpec(
        d_model = Int(get(overrides, "d_model", base.d_model)),
        n_layers = Int(get(overrides, "n_layers", base.n_layers)),
        n_heads = Int(get(overrides, "n_heads", base.n_heads)),
        ff_multiplier = Float64(get(overrides, "ff_multiplier", base.ff_multiplier)),
        seq_len = Int(get(overrides, "seq_len", base.seq_len)),
        gate_scale = Float64(get(overrides, "gate_scale", base.gate_scale)),
        seed = Int(get(overrides, "seed", base.seed)),
        repeat = Int(get(overrides, "repeat", base.repeat)),
    )
end

function _materialise_spec(base::AblationSpec, overrides::Dict{String,Any})
    merged = _merge_spec(base, overrides)
    offset = Int(get(overrides, "seed_offset", 0))
    return AblationSpec(
        d_model = merged.d_model,
        n_layers = merged.n_layers,
        n_heads = merged.n_heads,
        ff_multiplier = merged.ff_multiplier,
        seq_len = merged.seq_len,
        gate_scale = merged.gate_scale,
        seed = merged.seed + offset,
        repeat = merged.repeat,
    )
end

function parameter_grid(parameter_values::Dict{String,Any})
    keys = collect(keys(parameter_values))
    isempty(keys) && return [Dict{String,Any}()]
    values = Tuple(parameter_values[k] for k in keys)
    combos = product(values...)
    grid = Vector{Dict{String,Any}}()
    for combo in combos
        row = Dict{String,Any}()
        for (idx, key) in enumerate(keys)
            row[key] = combo[idx]
        end
        push!(grid, row)
    end
    return grid
end

function _random_inputs(spec::AblationSpec)
    rng = MersenneTwister(spec.seed)
    X = randn(rng, spec.seq_len, spec.d_model)
    gates = collect(range(0.0, step = spec.gate_scale / max(spec.seq_len - 1, 1), length = spec.seq_len))
    return X, gates
end

function _attn_stats(attn_layers)
    isempty(attn_layers) && return (0.0, 0.0)
    means = [mean(abs.(layer)) for layer in attn_layers]
    return mean(means), std(means)
end

function _mask_stats(mask)
    isempty(mask) && return (0.0, 0.0)
    return mean(mask), std(mask)
end

function _output_stats(out)
    isempty(out) && return (0.0, 0.0)
    vals = vec(abs.(out))
    return mean(vals), std(vals)
end

function _run_single(spec::AblationSpec)
    adapter = SpiralTransformerJulia.create_adapter(
        d_model = spec.d_model,
        n_layers = spec.n_layers,
        n_heads = spec.n_heads,
        ff_multiplier = spec.ff_multiplier,
        seed = spec.seed,
    )
    X, gate_pos = _random_inputs(spec)
    output = SpiralTransformerJulia.forward(adapter, X, gate_pos)
    attn_mean, attn_std = _attn_stats(adapter.last_attn)
    mask_mean, mask_std = _mask_stats(adapter.last_gate_mask)
    out_mean, out_std = _output_stats(output)
    return (
        d_model = spec.d_model,
        n_layers = spec.n_layers,
        n_heads = spec.n_heads,
        ff_multiplier = spec.ff_multiplier,
        seq_len = spec.seq_len,
        gate_scale = spec.gate_scale,
        seed = spec.seed,
        repeat = spec.repeat,
        attention_mean = attn_mean,
        attention_std = attn_std,
        gate_mask_mean = mask_mean,
        gate_mask_std = mask_std,
        output_mean = out_mean,
        output_std = out_std,
    )
end

function run_parameter_sweep(overrides_list::Vector{Dict{String,Any}}; base::AblationSpec = AblationSpec())
    results = NamedTuple[]
    for overrides in overrides_list
        spec = _materialise_spec(base, overrides)
        push!(results, _run_single(spec))
    end
    return results
end

end
