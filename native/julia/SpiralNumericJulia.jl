module SpiralNumericJulia

using LinearAlgebra
using Statistics

export matmul, dot, mean_reduce, std_reduce, var_reduce, sum_reduce,
       min_reduce, max_reduce, tanh_map, exp_map, log_map, logaddexp_map,
       median_all, median_reduce, abs_map, clip_map, sqrt_map, diff_vec,
       maximum_map, minimum_map, argsort_indices, argmax_index, trace_value,
       norm_value, inv_matrix, solve_matrix, cholesky_lower, slogdet_pair

const INV_ABS_TOL = 1.0e-12
const INV_REL_TOL = 1.0e-9

function _as_array(data)
    if data isa Number
        return fill(promote_type(Float64, typeof(data))(data), 1)
    end
    return Array(data)
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

function _wrap_scalar_keepdims(value::Float64, arr, keepdims::Bool)
    if !keepdims
        return value
    end
    if ndims(arr) <= 1
        return [value]
    end
    return [[value]]
end

function _wrap_axis_values(values::Vector{Float64}, keepdims::Bool, column::Bool)
    if !keepdims
        return values
    end
    if column
        return [[v] for v in values]
    end
    return [values]
end

function _var_vector(values::Vector{Float64}, ddof::Int)
    n = length(values)
    if n == 0
        return ddof <= 0 ? 0.0 : NaN
    end
    mean_val = sum(values) / n
    accum = sum((v - mean_val) ^ 2 for v in values)
    denom = n - ddof
    if denom <= 0
        return NaN
    end
    return accum / denom
end

function _std_vector(values::Vector{Float64}, ddof::Int)
    variance = _var_vector(values, ddof)
    return sqrt(variance)
end

function mean_reduce(data, axis, keepdims::Bool=false)
    arr = _as_array(data)
    if axis === nothing
        values = collect(vec(arr))
        if isempty(values)
            return _wrap_scalar_keepdims(0.0, arr, keepdims)
        end
        mean_val = sum(values) / length(values)
        return _wrap_scalar_keepdims(mean_val, arr, keepdims)
    elseif axis == 0
        if ndims(arr) == 1
            values = collect(arr)
            if isempty(values)
                return _wrap_axis_values([0.0], keepdims, false)
            end
            mean_val = sum(values) / length(values)
            return _wrap_axis_values([mean_val], keepdims, false)
        end
        vals = Float64[]
        for col in eachcol(arr)
            column = collect(col)
            if isempty(column)
                push!(vals, 0.0)
            else
                push!(vals, sum(column) / length(column))
            end
        end
        return _wrap_axis_values(vals, keepdims, false)
    elseif axis == 1
        if ndims(arr) == 1
            throw(ArgumentError("axis=1 requires a 2D input"))
        end
        vals = Float64[]
        for row in eachrow(arr)
            items = collect(row)
            if isempty(items)
                push!(vals, 0.0)
            else
                push!(vals, sum(items) / length(items))
            end
        end
        return _wrap_axis_values(vals, keepdims, true)
    else
        throw(ArgumentError("Unsupported axis for mean"))
    end
end

function std_reduce(data, axis, ddof::Int=0, keepdims::Bool=false)
    arr = _as_array(data)
    if axis === nothing
        values = collect(vec(arr))
        if isempty(values)
            return _wrap_scalar_keepdims(ddof <= 0 ? 0.0 : NaN, arr, keepdims)
        end
        std_val = _std_vector(values, ddof)
        return _wrap_scalar_keepdims(std_val, arr, keepdims)
    elseif axis == 0
        if ndims(arr) == 1
            std_val = _std_vector(collect(arr), ddof)
            return _wrap_axis_values([std_val], keepdims, false)
        end
        vals = Float64[]
        for col in eachcol(arr)
            push!(vals, _std_vector(collect(col), ddof))
        end
        return _wrap_axis_values(vals, keepdims, false)
    elseif axis == 1
        if ndims(arr) == 1
            throw(ArgumentError("axis=1 requires a 2D input"))
        end
        vals = Float64[]
        for row in eachrow(arr)
            push!(vals, _std_vector(collect(row), ddof))
        end
        return _wrap_axis_values(vals, keepdims, true)
    else
        throw(ArgumentError("Unsupported axis for std"))
    end
end

function var_reduce(data, axis, ddof::Int=0, keepdims::Bool=false)
    arr = _as_array(data)
    if axis === nothing
        values = collect(vec(arr))
        if isempty(values)
            return _wrap_scalar_keepdims(ddof <= 0 ? 0.0 : NaN, arr, keepdims)
        end
        var_val = _var_vector(values, ddof)
        return _wrap_scalar_keepdims(var_val, arr, keepdims)
    elseif axis == 0
        if ndims(arr) == 1
            values = collect(arr)
            var_val = _var_vector(values, ddof)
            return _wrap_axis_values([var_val], keepdims, false)
        end
        vals = Float64[]
        for col in eachcol(arr)
            column = collect(col)
            push!(vals, _var_vector(column, ddof))
        end
        return _wrap_axis_values(vals, keepdims, false)
    elseif axis == 1
        if ndims(arr) == 1
            throw(ArgumentError("axis=1 requires a 2D input"))
        end
        vals = Float64[]
        for row in eachrow(arr)
            items = collect(row)
            push!(vals, _var_vector(items, ddof))
        end
        return _wrap_axis_values(vals, keepdims, true)
    else
        throw(ArgumentError("Unsupported axis for var"))
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

function min_reduce(data, axis, keepdims::Bool=false)
    arr = _as_array(data)
    if axis === nothing
        values = collect(vec(arr))
        if isempty(values)
            throw(ArgumentError("minimum of empty array"))
        end
        min_val = minimum(values)
        return _wrap_scalar_keepdims(min_val, arr, keepdims)
    elseif axis == 0
        if ndims(arr) == 1
            values = collect(arr)
            if isempty(values)
                throw(ArgumentError("minimum of empty array"))
            end
            min_val = minimum(values)
            return _wrap_axis_values([min_val], keepdims, false)
        end
        if size(arr, 1) == 0
            throw(ArgumentError("minimum of empty array"))
        end
        vals = Float64[]
        for col in eachcol(arr)
            column = collect(col)
            if isempty(column)
                throw(ArgumentError("minimum of empty array"))
            end
            push!(vals, minimum(column))
        end
        return _wrap_axis_values(vals, keepdims, false)
    elseif axis == 1
        if ndims(arr) == 1
            throw(ArgumentError("axis=1 requires a 2D input"))
        end
        vals = Float64[]
        if size(arr, 1) == 0
            return _wrap_axis_values(vals, keepdims, true)
        end
        for row in eachrow(arr)
            items = collect(row)
            if isempty(items)
                throw(ArgumentError("minimum of empty array"))
            end
            push!(vals, minimum(items))
        end
        return _wrap_axis_values(vals, keepdims, true)
    else
        throw(ArgumentError("Unsupported axis for min"))
    end
end

function max_reduce(data, axis, keepdims::Bool=false)
    arr = _as_array(data)
    if axis === nothing
        values = collect(vec(arr))
        if isempty(values)
            throw(ArgumentError("maximum of empty array"))
        end
        max_val = maximum(values)
        return _wrap_scalar_keepdims(max_val, arr, keepdims)
    elseif axis == 0
        if ndims(arr) == 1
            values = collect(arr)
            if isempty(values)
                throw(ArgumentError("maximum of empty array"))
            end
            max_val = maximum(values)
            return _wrap_axis_values([max_val], keepdims, false)
        end
        if size(arr, 1) == 0
            throw(ArgumentError("maximum of empty array"))
        end
        vals = Float64[]
        for col in eachcol(arr)
            column = collect(col)
            if isempty(column)
                throw(ArgumentError("maximum of empty array"))
            end
            push!(vals, maximum(column))
        end
        return _wrap_axis_values(vals, keepdims, false)
    elseif axis == 1
        if ndims(arr) == 1
            throw(ArgumentError("axis=1 requires a 2D input"))
        end
        vals = Float64[]
        if size(arr, 1) == 0
            return _wrap_axis_values(vals, keepdims, true)
        end
        for row in eachrow(arr)
            items = collect(row)
            if isempty(items)
                throw(ArgumentError("maximum of empty array"))
            end
            push!(vals, maximum(items))
        end
        return _wrap_axis_values(vals, keepdims, true)
    else
        throw(ArgumentError("Unsupported axis for max"))
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

function maximum_map(a, b)
    A = _as_array(a)
    B = _as_array(b)
    return max.(A, B)
end

function minimum_map(a, b)
    A = _as_array(a)
    B = _as_array(b)
    return min.(A, B)
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
    n = size(arr, 1)
    if n == 0
        return Array{eltype(arr)}(undef, 0, 0)
    end
    work_T = eltype(arr)
    if work_T <: Complex
        work_T = promote_type(work_T, ComplexF64)
    else
        work_T = promote_type(work_T, Float64)
    end
    working = Array{work_T}(arr)
    scales = similar(working, work_T, n)
    for (idx, row) in enumerate(eachrow(working))
        row_max = maximum(abs, row)
        scales[idx] = row_max == zero(work_T) ? one(work_T) : row_max
    end
    identity_block = Matrix{work_T}(I, n, n)
    augmented = hcat(working ./ scales, identity_block)
    for col in 1:n
        pivot_sub = abs.(augmented[col:n, col])
        pivot_offset = findmax(pivot_sub)[2] - 1
        pivot_idx = col + pivot_offset
        pivot_val = augmented[pivot_idx, col]
        column_norm = maximum(abs, augmented[:, col])
        tol = INV_ABS_TOL + INV_REL_TOL * max(one(work_T), column_norm)
        if abs(pivot_val) <= tol
            throw(ArgumentError("Singular matrix: pivot $(pivot_val) at column $(col - 1)"))
        end
        if pivot_idx != col
            augmented[[col, pivot_idx], :] = augmented[[pivot_idx, col], :]
        end
        augmented[col, :] ./= augmented[col, col]
        for row in 1:n
            if row == col
                continue
            end
            factor = augmented[row, col]
            if factor != zero(work_T)
                augmented[row, :] .-= factor .* augmented[col, :]
            end
        end
    end
    inv_block = augmented[:, n + 1:end]
    inv_block ./= scales'
    return Array{eltype(arr)}(inv_block)
end

function solve_matrix(coeffs, rhs)
    a = _as_matrix(coeffs)
    if size(a, 1) != size(a, 2)
        throw(ArgumentError("Coefficient matrix must be square"))
    end
    b = _as_array(rhs)
    if ndims(b) == 1
        if length(b) != size(a, 1)
            throw(ArgumentError("Right-hand side dimension mismatch"))
        end
        return Array{Float64}(a \ b)
    else
        if size(b, 1) != size(a, 1)
            throw(ArgumentError("Right-hand side dimension mismatch"))
        end
        return Array{Float64}(a \ b)
    end
end

function cholesky_lower(data)
    arr = _as_matrix(data)
    if size(arr, 1) != size(arr, 2)
        throw(ArgumentError("Matrix must be square"))
    end
    factor = cholesky(Symmetric(arr))
    return Array{Float64}(factor.L)
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
