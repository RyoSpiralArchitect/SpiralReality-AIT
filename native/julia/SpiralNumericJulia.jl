module SpiralNumericJulia

using LinearAlgebra
using Statistics

export matmul, dot, mean_reduce, std_reduce, sum_reduce, tanh_map, exp_map,
       log_map, logaddexp_map, median_all, median_reduce, abs_map, clip_map,
       sqrt_map, diff_vec, argsort_indices, argmax_index, trace_value,
       norm_value, inv_matrix, slogdet_pair

function _as_array(data)
    if data isa Number
        return [Float64(data)]
    end
    return Array{Float64}(data)
end

function _as_vector(data)
    return vec(_as_array(data))
end

function _as_matrix(data)
    arr = _as_array(data)
    if ndims(arr) == 1
        return reshape(arr, :, 1)
    end
    return arr
end

function matmul(a, b)
    A = _as_array(a)
    B = _as_array(b)
    if ndims(A) == 1 && ndims(B) == 1
        return LinearAlgebra.dot(vec(A), vec(B))
    elseif ndims(A) == 2 && ndims(B) == 1
        return Array{Float64}(A * vec(B))
    elseif ndims(A) == 2 && ndims(B) == 2
        return Array{Float64}(A * B)
    elseif ndims(A) == 1 && ndims(B) == 2
        result = vec(A)' * B
        return Array{Float64}(vec(result))
    else
        throw(ArgumentError("Unsupported operand shapes for matmul"))
    end
end

function dot(a, b)
    return LinearAlgebra.dot(_as_vector(a), _as_vector(b))
end

function mean_reduce(data, axis)
    arr = _as_array(data)
    if axis === nothing
        return Statistics.mean(arr)
    elseif axis == 0
        vals = Statistics.mean(arr, dims=1)
        return vec(vals)
    elseif axis == 1
        vals = Statistics.mean(arr, dims=2)
        return vec(vals)
    else
        throw(ArgumentError("Unsupported axis for mean"))
    end
end

function std_reduce(data, axis)
    arr = _as_array(data)
    if axis === nothing
        return Statistics.std(vec(arr); corrected=false)
    elseif axis == 0
        vals = Statistics.std(arr, dims=1; corrected=false)
        return vec(vals)
    elseif axis == 1
        vals = Statistics.std(arr, dims=2; corrected=false)
        return vec(vals)
    else
        throw(ArgumentError("Unsupported axis for std"))
    end
end

function sum_reduce(data, axis, keepdims::Bool)
    arr = _as_array(data)
    if axis === nothing
        return sum(arr)
    elseif axis == 0
        vals = sum(arr, dims=1)
        return keepdims ? Array(vals) : vec(vals)
    elseif axis == 1
        vals = sum(arr, dims=2)
        return keepdims ? Array(vals) : vec(vals)
    else
        throw(ArgumentError("Unsupported axis for sum"))
    end
end

function tanh_map(data)
    return tanh.(_as_array(data))
end

function exp_map(data)
    return exp.(_as_array(data))
end

function log_map(data)
    return log.(_as_array(data))
end

function logaddexp_map(a, b)
    A = _as_array(a)
    B = _as_array(b)
    m = max.(A, B)
    return m .+ log.(exp.(A .- m) .+ exp.(B .- m))
end

function _median(values::Vector{Float64})
    len = length(values)
    len == 0 && return 0.0
    sorted_vals = sort(values)
    if isodd(len)
        return sorted_vals[(len + 1) ÷ 2]
    else
        mid = len ÷ 2
        return 0.5 * (sorted_vals[mid] + sorted_vals[mid + 1])
    end
end

function median_reduce(data, axis)
    arr = _as_array(data)
    if axis === nothing
        return _median(vec(arr))
    elseif axis == 0
        ndims(arr) == 1 && return [_median(vec(arr))]
        ndims(arr) == 0 && return [Float64(arr)]
        cols = eachcol(arr)
        return [_median(collect(col)) for col in cols]
    elseif axis == 1
        ndims(arr) == 1 && return [_median(vec(arr))]
        return [_median(collect(row)) for row in eachrow(arr)]
    else
        throw(ArgumentError("Unsupported axis for median"))
    end
end

function median_all(data, axis=nothing)
    return median_reduce(data, axis)
end

function abs_map(data)
    return abs.(_as_array(data))
end

function clip_map(data, lo::Float64, hi::Float64)
    return clamp.(_as_array(data), lo, hi)
end

function sqrt_map(data)
    return sqrt.(_as_array(data))
end

function diff_vec(data, n::Int=1)
    n < 0 && throw(ArgumentError("diff order must be non-negative"))
    arr = _as_array(data)
    result = Array{Float64}(arr)
    n == 0 && return result
    for _ in 1:n
        if ndims(result) == 1
            length(result) <= 1 && return Float64[]
            result = diff(result)
        elseif ndims(result) == 2
            size(result, 2) <= 1 && return [Float64[] for _ in 1:size(result, 1)]
            result = Array{Float64}(diff(result, dims=2))
        else
            throw(ArgumentError("diff is only supported for 1D or 2D inputs"))
        end
    end
    return result
end

function argsort_indices(data)
    return collect(sortperm(_as_vector(data)) .- 1)
end

function argmax_index(data)
    _, idx = findmax(_as_vector(data))
    return idx - 1
end

function trace_value(data)
    mat = _as_matrix(data)
    if size(mat, 2) == 1 && size(mat, 1) == 1
        return mat[1, 1]
    end
    return tr(mat)
end

function norm_value(data)
    return LinearAlgebra.norm(_as_vector(data))
end

function inv_matrix(data)
    arr = _as_matrix(data)
    if size(arr, 1) != size(arr, 2)
        throw(ArgumentError("Matrix must be square"))
    end
    return Array{Float64}(inv(arr))
end

function slogdet_pair(data)
    arr = _as_matrix(data)
    if size(arr, 1) != size(arr, 2)
        throw(ArgumentError("Matrix must be square"))
    end
    logdet, sign = logabsdet(arr)
    return (sign, logdet)
end

end # module
