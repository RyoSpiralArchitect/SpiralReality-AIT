module SpiralBoundaryJulia

export JuliaBoundaryStudent, create_student, AVAILABLE_DEVICES, DEFAULT_DEVICE, BACKEND_KIND

using Base: Set
using Unicode

const BACKEND_KIND = "julia"
const DEFAULT_DEVICE = "cpu"
const AVAILABLE_DEVICES = ("cpu",)

mutable struct PairStats
    boundary::Float64
    total::Float64
end

mutable struct JuliaBoundaryStudent
    pair_stats::Dict{Tuple{String,String},PairStats}
    bias::Float64
    smoothing::Float64
    threshold::Float64
    fallback_bias::Float64
    device::String
end

function JuliaBoundaryStudent()
    JuliaBoundaryStudent(Dict{Tuple{String,String},PairStats}(), 0.0, 0.5, 0.55, 0.12, String(DEFAULT_DEVICE))
end

function configure!(student::JuliaBoundaryStudent, cfg)
    if haskey(cfg, "compiled_threshold")
        student.threshold = Float64(cfg["compiled_threshold"])
    elseif haskey(cfg, "boundary_threshold")
        student.threshold = Float64(cfg["boundary_threshold"])
    end
    if haskey(cfg, "compiled_smoothing")
        student.smoothing = max(1e-6, Float64(cfg["compiled_smoothing"]))
    end
    if haskey(cfg, "fallback_bias")
        student.fallback_bias = Float64(cfg["fallback_bias"])
    end
    return student
end

collect_chars(text::AbstractString) = [String(g) for g in graphemes(text)]

function collect_boundaries(segments::Vector{String}, total_chars::Int)
    boundaries = Int[]
    cursor = 0
    for (idx, segment) in enumerate(segments)
        cursor += length(collect_chars(segment))
        cursor = min(cursor, total_chars)
        if idx < length(segments) && cursor > 0
            push!(boundaries, cursor)
        end
    end
    return boundaries
end

function update_counts!(student::JuliaBoundaryStudent, chars, boundaries::Vector{Int})
    isempty(chars) && return
    boundary_set = Set(boundaries)
    for i in 1:(length(chars)-1)
        key = (chars[i], chars[i+1])
        stats = get!(student.pair_stats, key) do
            PairStats(0.0, 0.0)
        end
        stats.total += 1.0
        if i in boundary_set
            stats.boundary += 1.0
        end
    end
end

function fallback_probability(student::JuliaBoundaryStudent, prev::String, next::String)
    base = student.fallback_bias
    if occursin(r"^[\s]$", prev)
        base += 0.35
    end
    if occursin(r"[!?,.;:()\[\]{}<>\"'`~+\-*/\\|]", prev)
        base += 0.25
    end
    if occursin(r"^[\s]$", next)
        base += 0.1
    end
    if occursin(r"[!?,.;:()\[\]{}<>\"'`~+\-*/\\|]", next)
        base += 0.15
    end
    if length(prev) > 1 || length(next) > 1
        base += 0.05
    end
    return clamp(base, 0.01, 0.95)
end

function pair_probability(student::JuliaBoundaryStudent, prev::String, next::String)
    key = (prev, next)
    if haskey(student.pair_stats, key)
        stats = student.pair_stats[key]
        return clamp((stats.boundary + student.smoothing) / (stats.total + 2 * student.smoothing), 1e-4, 1 - 1e-4)
    else
        return fallback_probability(student, prev, next)
    end
end

function compute_bias!(student::JuliaBoundaryStudent)
    totals = reduce((a, b) -> (a[1]+b[1], a[2]+b[2]),
                    ((s.boundary, s.total) for s in values(student.pair_stats)); init=(0.0, 0.0))
    boundary_pairs, total_pairs = totals
    if total_pairs <= 0.0
        student.bias = student.fallback_bias
    else
        student.bias = (boundary_pairs + student.smoothing) / (total_pairs + 2 * student.smoothing)
    end
    student.threshold = clamp((student.threshold + student.bias)/2, 0.2, 0.8)
    return student
end

function train!(student::JuliaBoundaryStudent, texts, segments, cfg)
    configure!(student, cfg)
    student.pair_stats = Dict{Tuple{String,String},PairStats}()
    total_tokens = 0
    total_pairs = 0.0
    total_boundaries = 0.0

    texts_vec = [String(t) for t in texts]
    segments_vec = [[String(x) for x in seg] for seg in segments]
    length(texts_vec) == length(segments_vec) || throw(ArgumentError("texts and segments must align"))

    for (text, segs) in zip(texts_vec, segments_vec)
        chars = collect_chars(text)
        total_tokens += length(chars)
        bounds = collect_boundaries(segs, length(chars))
        update_counts!(student, chars, bounds)
    end

    for stats in values(student.pair_stats)
        total_pairs += stats.total
        total_boundaries += stats.boundary
    end

    compute_bias!(student)
    return Dict(
        "backend" => string(BACKEND_KIND, ":", student.device),
        "examples" => length(texts_vec),
        "tokens" => total_tokens,
        "pairs_tracked" => length(student.pair_stats),
        "training_pairs" => total_pairs,
        "boundary_pairs" => total_boundaries,
        "threshold" => student.threshold,
        "smoothing" => student.smoothing,
        "device" => student.device,
    )
end

function boundary_probs(student::JuliaBoundaryStudent, text::AbstractString)
    chars = collect_chars(text)
    probs = Float64[]
    for i in 1:(length(chars)-1)
        push!(probs, pair_probability(student, chars[i], chars[i+1]))
    end
    return probs
end

function decode(student::JuliaBoundaryStudent, text::AbstractString)
    chars = collect_chars(text)
    probs = boundary_probs(student, text)
    tokens = String[]
    buffer = IOBuffer()
    for (idx, ch) in enumerate(chars)
        print(buffer, ch)
        boundary = idx <= length(probs) && probs[idx] >= student.threshold
        if boundary
            push!(tokens, String(take!(buffer)))
        end
    end
    remaining = String(take!(buffer))
    if !isempty(remaining)
        push!(tokens, remaining)
    end
    return tokens
end

function export_state(student::JuliaBoundaryStudent)
    stats = Dict{String,Any}()
    for (key, value) in student.pair_stats
        stats[string(key[1], "|", key[2])] = Dict(
            "boundary" => value.boundary,
            "total" => value.total,
        )
    end
    return Dict(
        "pair_stats" => stats,
        "bias" => student.bias,
        "threshold" => student.threshold,
        "smoothing" => student.smoothing,
        "device" => student.device,
        "backend" => BACKEND_KIND,
    )
end

function load_state!(student::JuliaBoundaryStudent, state)
    student.pair_stats = Dict{Tuple{String,String},PairStats}()
    stats = get(state, "pair_stats", Dict())
    for (key, values) in stats
        parts = split(String(key), "|")
        if length(parts) == 2
            student.pair_stats[(parts[1], parts[2])] = PairStats(Float64(values["boundary"]), Float64(values["total"]))
        end
    end
    student.bias = Float64(get(state, "bias", student.fallback_bias))
    student.threshold = Float64(get(state, "threshold", student.threshold))
    student.smoothing = Float64(get(state, "smoothing", student.smoothing))
    student.device = String(get(state, "device", student.device))
    compute_bias!(student)
    return student
end

function available_devices(student::JuliaBoundaryStudent)
    return AVAILABLE_DEVICES
end

function preferred_device(student::JuliaBoundaryStudent)
    return student.device
end

function to_device!(student::JuliaBoundaryStudent, device::AbstractString)
    if device ∈ AVAILABLE_DEVICES
        student.device = String(device)
        return true
    else
        return false
    end
end

function create_student()
    return JuliaBoundaryStudent()
end

function Base.getproperty(student::JuliaBoundaryStudent, name::Symbol)
    if name === :configure
        return cfg -> begin configure!(student, cfg); nothing end
    elseif name === :attach_phase
        return _ -> nothing
    elseif name === :attach_encoder
        return _ -> nothing
    elseif name === :train
        return (texts, segments, cfg) -> train!(student, texts, segments, cfg)
    elseif name === :boundary_probs
        return text -> boundary_probs(student, text)
    elseif name === :decode
        return text -> decode(student, text)
    elseif name === :export_state
        return () -> export_state(student)
    elseif name === :load_state
        return state -> begin load_state!(student, state); nothing end
    elseif name === :available_devices
        return () -> available_devices(student)
    elseif name === :preferred_device
        return () -> preferred_device(student)
    elseif name === :to_device
        return device -> to_device!(student, device)
    else
        return getfield(student, name)
    end
end

function Base.propertynames(student::JuliaBoundaryStudent, private::Bool=false)
    return (:configure, :attach_phase, :attach_encoder, :train, :boundary_probs, :decode,
            :export_state, :load_state, :available_devices, :preferred_device, :to_device)
end

end # module
