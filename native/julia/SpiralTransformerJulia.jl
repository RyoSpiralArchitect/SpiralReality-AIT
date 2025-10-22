module SpiralTransformerJulia

export JuliaTransformerAdapter, create_adapter, BACKEND_KIND, DEFAULT_DEVICE, AVAILABLE_DEVICES

using LinearAlgebra
using Random
using Statistics

const BACKEND_KIND = "julia-transformer"
const DEFAULT_DEVICE = "cpu"
const AVAILABLE_DEVICES = (DEFAULT_DEVICE,)

mutable struct JuliaTransformerAdapter
    d_model::Int
    n_layers::Int
    n_heads::Int
    head_dim::Int
    ff_dim::Int
    Wq::Vector{Matrix{Float64}}
    Wk::Vector{Matrix{Float64}}
    Wv::Vector{Matrix{Float64}}
    Wo::Vector{Matrix{Float64}}
    Wff1::Vector{Matrix{Float64}}
    bff1::Vector{Vector{Float64}}
    Wff2::Vector{Matrix{Float64}}
    bff2::Vector{Vector{Float64}}
    ln1_gamma::Vector{Vector{Float64}}
    ln1_beta::Vector{Vector{Float64}}
    ln2_gamma::Vector{Vector{Float64}}
    ln2_beta::Vector{Vector{Float64}}
    gate_bias::Vector{Vector{Float64}}
    ff_gate::Vector{Vector{Float64}}
    last_attn::Vector{Matrix{Float64}}
    last_gate_mask::Matrix{Float64}
    device::String
end

function JuliaTransformerAdapter(; d_model::Int = 128, n_layers::Int = 4, n_heads::Int = 4, ff_multiplier::Float64 = 4.0, seed::Int = 2025)
    n_heads > 0 || throw(ArgumentError("n_heads must be positive"))
    d_model % n_heads == 0 || throw(ArgumentError("d_model must be divisible by n_heads"))
    rng = MersenneTwister(seed)
    head_dim = Int(d_model ÷ n_heads)
    ff_dim = max(head_dim * n_heads, Int(round(d_model * ff_multiplier)))
    scale = 1.0 / sqrt(d_model)
    randn_matrix(r, c) = randn(rng, r, c) .* scale
    randn_vector(len) = randn(rng, len) .* scale
    Wq = [randn_matrix(d_model, d_model) for _ in 1:n_layers]
    Wk = [randn_matrix(d_model, d_model) for _ in 1:n_layers]
    Wv = [randn_matrix(d_model, d_model) for _ in 1:n_layers]
    Wo = [randn_matrix(d_model, d_model) for _ in 1:n_layers]
    Wff1 = [randn_matrix(d_model, ff_dim) for _ in 1:n_layers]
    bff1 = [randn_vector(ff_dim) for _ in 1:n_layers]
    Wff2 = [randn_matrix(ff_dim, d_model) for _ in 1:n_layers]
    bff2 = [randn_vector(d_model) for _ in 1:n_layers]
    ln1_gamma = [ones(Float64, d_model) for _ in 1:n_layers]
    ln1_beta = [zeros(Float64, d_model) for _ in 1:n_layers]
    ln2_gamma = [ones(Float64, d_model) for _ in 1:n_layers]
    ln2_beta = [zeros(Float64, d_model) for _ in 1:n_layers]
    gate_bias = [zeros(Float64, 2) for _ in 1:n_layers]
    ff_gate = [zeros(Float64, 2) for _ in 1:n_layers]
    last_attn = Matrix{Float64}[]
    last_gate_mask = zeros(Float64, 0, 0)
    device = String(DEFAULT_DEVICE)
    return JuliaTransformerAdapter(d_model, n_layers, n_heads, head_dim, ff_dim, Wq, Wk, Wv, Wo, Wff1, bff1, Wff2, bff2,
        ln1_gamma, ln1_beta, ln2_gamma, ln2_beta, gate_bias, ff_gate, last_attn, last_gate_mask, device)
end

function _layer_norm(row::Vector{Float64}, gamma::Vector{Float64}, beta::Vector{Float64})
    μ = mean(row)
    σ = sqrt(var(row) + 1e-5)
    out = similar(row)
    for i in eachindex(row)
        g = i <= length(gamma) ? gamma[i] : gamma[end]
        b = i <= length(beta) ? beta[i] : beta[end]
        out[i] = g * ((row[i] - μ) / σ) + b
    end
    return out
end

function _layer_norm(mat::Matrix{Float64}, gamma::Vector{Float64}, beta::Vector{Float64})
    out = similar(mat)
    for i in axes(mat, 1)
        out[i, :] = _layer_norm(vec(mat[i, :]), gamma, beta)
    end
    return out
end

function _ensure_matrix(mask)
    mask === nothing && return zeros(Float64, 0, 0)
    mask isa Matrix{Float64} && return copy(mask)
    rows = Vector{Float64}[]
    for row in mask
        push!(rows, [Float64(x) for x in row])
    end
    if isempty(rows)
        return zeros(Float64, 0, 0)
    end
    return reduce(vcat, (row' for row in rows))
end

function forward(adapter::JuliaTransformerAdapter, X::AbstractMatrix{<:Real}, gate_pos::AbstractVector{<:Real}; gate_mask=nothing)
    seq_len = size(X, 1)
    if seq_len == 0
        adapter.last_attn = Matrix{Float64}[]
        adapter.last_gate_mask = zeros(Float64, 0, 0)
        return zeros(Float64, 0, adapter.d_model)
    end
    H = Array{Float64}(X)
    gate_vals = [Float64(x) for x in gate_pos]
    mask = _ensure_matrix(gate_mask)
    if isempty(mask)
        mask = zeros(Float64, length(gate_vals), length(gate_vals))
        for i in 1:length(gate_vals), j in 1:length(gate_vals)
            mask[i, j] = min(gate_vals[i], gate_vals[j])
        end
    end
    outer = zeros(Float64, size(mask))
    for i in 1:size(outer, 1), j in 1:size(outer, 2)
        if i <= length(gate_vals) && j <= length(gate_vals)
            outer[i, j] = gate_vals[i] * gate_vals[j]
        end
    end
    gate_mean = isempty(gate_vals) ? 0.0 : mean(gate_vals)
    gate_std = isempty(gate_vals) ? 0.0 : sqrt(var(gate_vals))
    mask_energy = isempty(mask) ? 0.0 : mean(mask)

    adapter.last_attn = Matrix{Float64}[]
    for layer in 1:adapter.n_layers
        norm_in = _layer_norm(H, adapter.ln1_gamma[layer], adapter.ln1_beta[layer])
        Q = norm_in * adapter.Wq[layer]
        K = norm_in * adapter.Wk[layer]
        V = norm_in * adapter.Wv[layer]
        seq_len = size(Q, 1)
        key_len = size(K, 1)
        attn_heads = [zeros(Float64, seq_len, key_len) for _ in 1:adapter.n_heads]
        scale = 1.0 / sqrt(adapter.head_dim)
        for head in 1:adapter.n_heads
            qcols = ((head - 1) * adapter.head_dim + 1):(head * adapter.head_dim)
            for i in 1:seq_len
                for j in 1:key_len
                    qv = view(Q, i, qcols)
                    kv = view(K, j, qcols)
                    dot = sum(qv .* kv)
                    bias = adapter.gate_bias[layer][1] * ((i <= size(outer, 1) && j <= size(outer, 2)) ? outer[i, j] : 0.0) +
                           adapter.gate_bias[layer][2] * ((i <= size(mask, 1) && j <= size(mask, 2)) ? mask[i, j] : 0.0)
                    attn_heads[head][i, j] = dot * scale + bias
                end
                row = view(attn_heads[head], i, :)
                maxval = maximum(row)
                row .= exp.(row .- maxval)
                denom = sum(row) + 1e-12
                row ./= denom
            end
        end
        context = zeros(Float64, seq_len, adapter.d_model)
        for head in 1:adapter.n_heads
            cols = ((head - 1) * adapter.head_dim + 1):(head * adapter.head_dim)
            for i in 1:seq_len
                for j in 1:key_len
                    weight = attn_heads[head][i, j]
                    context[i, cols] .+= weight .* view(V, j, cols)
                end
            end
        end
        attn_out = context * adapter.Wo[layer]
        H .+= attn_out
        ff_in = _layer_norm(H, adapter.ln2_gamma[layer], adapter.ln2_beta[layer])
        ff_hidden = tanh.(ff_in * adapter.Wff1[layer] .+ adapter.bff1[layer]')
        ff_out = ff_hidden * adapter.Wff2[layer] .+ adapter.bff2[layer]'
        modulation = 1.0 + adapter.ff_gate[layer][1] * gate_mean + adapter.ff_gate[layer][2] * (gate_std + mask_energy)
        H .+= modulation .* ff_out
        attn_mean = zeros(Float64, seq_len, key_len)
        for head in 1:adapter.n_heads
            attn_mean .+= attn_heads[head]
        end
        attn_mean ./= adapter.n_heads
        push!(adapter.last_attn, attn_mean)
    end
    adapter.last_gate_mask = copy(mask)
    return H
end

function tune_from_boundary!(adapter::JuliaTransformerAdapter, base_gate, targets; lr::Float64 = 1e-3)
    base_vals = [Float64(x) for x in base_gate]
    target_vals = [Float64(x) for x in targets]
    if isempty(base_vals) || length(base_vals) != length(target_vals)
        return
    end
    diffs = base_vals .- target_vals
    err_mean = mean(diffs)
    err_std = sqrt(var(diffs))
    for layer in 1:adapter.n_layers
        adapter.gate_bias[layer][1] -= lr * err_mean
        adapter.gate_bias[layer][2] -= lr * err_std
        adapter.ff_gate[layer][1] -= lr * err_mean
        adapter.ff_gate[layer][2] -= lr * err_std
    end
    return nothing
end

function export_state(adapter::JuliaTransformerAdapter)
    return Dict(
        "d_model" => adapter.d_model,
        "n_layers" => adapter.n_layers,
        "n_heads" => adapter.n_heads,
        "Wq" => adapter.Wq,
        "Wk" => adapter.Wk,
        "Wv" => adapter.Wv,
        "Wo" => adapter.Wo,
        "Wff1" => adapter.Wff1,
        "bff1" => adapter.bff1,
        "Wff2" => adapter.Wff2,
        "bff2" => adapter.bff2,
        "ln1_gamma" => adapter.ln1_gamma,
        "ln1_beta" => adapter.ln1_beta,
        "ln2_gamma" => adapter.ln2_gamma,
        "ln2_beta" => adapter.ln2_beta,
        "gate_bias" => adapter.gate_bias,
        "ff_gate" => adapter.ff_gate,
    )
end

function load_state!(adapter::JuliaTransformerAdapter, state)
    adapter.d_model = Int(get(state, "d_model", adapter.d_model))
    adapter.n_layers = Int(get(state, "n_layers", adapter.n_layers))
    adapter.n_heads = Int(get(state, "n_heads", adapter.n_heads))
    adapter.head_dim = adapter.d_model ÷ max(1, adapter.n_heads)
    adapter.Wq = [Matrix{Float64}(mat) for mat in state["Wq"]]
    adapter.Wk = [Matrix{Float64}(mat) for mat in state["Wk"]]
    adapter.Wv = [Matrix{Float64}(mat) for mat in state["Wv"]]
    adapter.Wo = [Matrix{Float64}(mat) for mat in state["Wo"]]
    adapter.Wff1 = [Matrix{Float64}(mat) for mat in state["Wff1"]]
    adapter.bff1 = [Vector{Float64}(vec) for vec in state["bff1"]]
    adapter.Wff2 = [Matrix{Float64}(mat) for mat in state["Wff2"]]
    adapter.bff2 = [Vector{Float64}(vec) for vec in state["bff2"]]
    adapter.ln1_gamma = [Vector{Float64}(vec) for vec in state["ln1_gamma"]]
    adapter.ln1_beta = [Vector{Float64}(vec) for vec in state["ln1_beta"]]
    adapter.ln2_gamma = [Vector{Float64}(vec) for vec in state["ln2_gamma"]]
    adapter.ln2_beta = [Vector{Float64}(vec) for vec in state["ln2_beta"]]
    adapter.gate_bias = [Vector{Float64}(vec) for vec in state["gate_bias"]]
    adapter.ff_gate = [Vector{Float64}(vec) for vec in state["ff_gate"]]
    adapter.ff_dim = size(adapter.Wff1[1], 2)
    adapter.last_attn = Matrix{Float64}[]
    adapter.last_gate_mask = zeros(Float64, 0, 0)
    return adapter
end

device_inventory(adapter::JuliaTransformerAdapter) = AVAILABLE_DEVICES

available_devices(::JuliaTransformerAdapter) = AVAILABLE_DEVICES

function set_device!(adapter::JuliaTransformerAdapter, device::AbstractString)
    adapter.device = String(device)
    return adapter.device
end

create_adapter(; kwargs...) = JuliaTransformerAdapter(; kwargs...)

end

