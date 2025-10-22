#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

namespace {

enum class ShapeKind { Scalar, Vector, Matrix };

struct ParsedSequence {
    ShapeKind kind = ShapeKind::Scalar;
    double scalar = 0.0;
    Vector vector;
    Matrix matrix;
};

bool is_sequence_like(const py::handle &obj) {
    if (!py::isinstance<py::sequence>(obj)) {
        return false;
    }
    if (py::isinstance<py::str>(obj) || py::isinstance<py::bytes>(obj)) {
        return false;
    }
    return true;
}

ParsedSequence parse_sequence(const py::handle &obj) {
    ParsedSequence parsed;
    if (!is_sequence_like(obj)) {
        parsed.kind = ShapeKind::Scalar;
        parsed.scalar = py::cast<double>(obj);
        return parsed;
    }

    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    std::vector<py::object> items;
    items.reserve(py::len(seq));
    for (const py::handle &item : seq) {
        items.emplace_back(py::reinterpret_borrow<py::object>(item));
    }
    if (items.empty()) {
        parsed.kind = ShapeKind::Vector;
        parsed.vector = Vector();
        return parsed;
    }

    bool has_nested = false;
    for (const py::object &item : items) {
        if (is_sequence_like(item)) {
            has_nested = true;
            break;
        }
    }

    if (!has_nested) {
        parsed.kind = ShapeKind::Vector;
        parsed.vector.reserve(items.size());
        for (const py::object &item : items) {
            parsed.vector.push_back(py::cast<double>(item));
        }
        return parsed;
    }

    Matrix matrix;
    matrix.reserve(items.size());
    for (const py::object &row_obj : items) {
        if (!is_sequence_like(row_obj)) {
            // Fallback to treating as a vector if rows are not sequences.
            parsed.kind = ShapeKind::Vector;
            parsed.vector.reserve(items.size());
            for (const py::object &item : items) {
                parsed.vector.push_back(py::cast<double>(item));
            }
            return parsed;
        }
        py::sequence row_seq = py::reinterpret_borrow<py::sequence>(row_obj);
        Vector row;
        row.reserve(py::len(row_seq));
        for (const py::handle &value : row_seq) {
            row.push_back(py::cast<double>(value));
        }
        matrix.push_back(std::move(row));
    }
    parsed.kind = ShapeKind::Matrix;
    parsed.matrix = std::move(matrix);
    return parsed;
}

Vector flatten(const ParsedSequence &data) {
    if (data.kind == ShapeKind::Scalar) {
        return Vector{data.scalar};
    }
    if (data.kind == ShapeKind::Vector) {
        return data.vector;
    }
    Vector flat;
    for (const auto &row : data.matrix) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

Matrix ensure_matrix(const ParsedSequence &data) {
    if (data.kind == ShapeKind::Matrix) {
        return data.matrix;
    }
    if (data.kind == ShapeKind::Vector) {
        return Matrix{data.vector};
    }
    return Matrix{Vector{data.scalar}};
}

py::object to_python(double value) {
    return py::float_(value);
}

py::object to_python(const Vector &vec) {
    py::list list;
    for (double value : vec) {
        list.append(py::float_(value));
    }
    return list;
}

py::object to_python(const Matrix &mat) {
    py::list rows;
    for (const auto &row : mat) {
        rows.append(to_python(row));
    }
    return rows;
}

std::size_t column_count(const Matrix &mat) {
    std::size_t cols = std::numeric_limits<std::size_t>::max();
    for (const auto &row : mat) {
        cols = std::min(cols, row.size());
    }
    if (cols == std::numeric_limits<std::size_t>::max()) {
        return 0;
    }
    return cols;
}

Vector column_values(const Matrix &mat, std::size_t column) {
    Vector values;
    for (const auto &row : mat) {
        if (column < row.size()) {
            values.push_back(row[column]);
        }
    }
    return values;
}

double compute_mean(const Vector &values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double compute_std(const Vector &values) {
    if (values.empty()) {
        return 0.0;
    }
    double mean = compute_mean(values);
    double accum = 0.0;
    for (double value : values) {
        double diff = value - mean;
        accum += diff * diff;
    }
    return std::sqrt(accum / static_cast<double>(values.size()));
}

double compute_sum(const Vector &values) {
    return std::accumulate(values.begin(), values.end(), 0.0);
}

double compute_median(Vector values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

Vector mean_axis0(const Matrix &mat) {
    std::size_t cols = column_count(mat);
    Vector out(cols, 0.0);
    if (cols == 0) {
        return out;
    }
    for (std::size_t col = 0; col < cols; ++col) {
        out[col] = compute_mean(column_values(mat, col));
    }
    return out;
}

Vector mean_axis1(const Matrix &mat) {
    Vector out;
    out.reserve(mat.size());
    for (const auto &row : mat) {
        out.push_back(compute_mean(row));
    }
    return out;
}

Vector std_axis0(const Matrix &mat) {
    std::size_t cols = column_count(mat);
    Vector out(cols, 0.0);
    if (cols == 0) {
        return out;
    }
    for (std::size_t col = 0; col < cols; ++col) {
        out[col] = compute_std(column_values(mat, col));
    }
    return out;
}

Vector std_axis1(const Matrix &mat) {
    Vector out;
    out.reserve(mat.size());
    for (const auto &row : mat) {
        out.push_back(compute_std(row));
    }
    return out;
}

Vector sum_axis0(const Matrix &mat) {
    std::size_t cols = column_count(mat);
    Vector out(cols, 0.0);
    if (cols == 0) {
        return out;
    }
    for (std::size_t col = 0; col < cols; ++col) {
        out[col] = compute_sum(column_values(mat, col));
    }
    return out;
}

Vector sum_axis1(const Matrix &mat) {
    Vector out;
    out.reserve(mat.size());
    for (const auto &row : mat) {
        out.push_back(compute_sum(row));
    }
    return out;
}

Vector median_axis0(const Matrix &mat) {
    std::size_t cols = column_count(mat);
    Vector out(cols, 0.0);
    if (cols == 0) {
        return out;
    }
    for (std::size_t col = 0; col < cols; ++col) {
        out[col] = compute_median(column_values(mat, col));
    }
    return out;
}

Vector median_axis1(const Matrix &mat) {
    Vector out;
    out.reserve(mat.size());
    for (const auto &row : mat) {
        out.push_back(compute_median(row));
    }
    return out;
}

Vector diff_vector(Vector values, int order) {
    for (int step = 0; step < order; ++step) {
        if (values.size() <= 1) {
            return Vector();
        }
        Vector next;
        next.reserve(values.size() - 1);
        for (std::size_t i = 0; i + 1 < values.size(); ++i) {
            next.push_back(values[i + 1] - values[i]);
        }
        values = std::move(next);
    }
    return values;
}

Matrix diff_matrix(const Matrix &mat, int order) {
    Matrix out;
    out.reserve(mat.size());
    for (const auto &row : mat) {
        out.push_back(diff_vector(row, order));
    }
    return out;
}

Vector elementwise(const Vector &vec, const std::function<double(double)> &fn) {
    Vector out;
    out.reserve(vec.size());
    for (double value : vec) {
        out.push_back(fn(value));
    }
    return out;
}

Matrix elementwise(const Matrix &mat, const std::function<double(double)> &fn) {
    Matrix out;
    out.reserve(mat.size());
    for (const auto &row : mat) {
        out.push_back(elementwise(row, fn));
    }
    return out;
}

template <typename Fn>
py::object apply_unary(const ParsedSequence &data, Fn fn) {
    switch (data.kind) {
    case ShapeKind::Scalar:
        return to_python(fn(data.scalar));
    case ShapeKind::Vector:
        return to_python(elementwise(data.vector, fn));
    case ShapeKind::Matrix:
        return to_python(elementwise(data.matrix, fn));
    }
    throw std::runtime_error("Unhandled shape kind");
}

template <typename Fn>
py::object apply_binary(const ParsedSequence &lhs, const ParsedSequence &rhs, Fn fn) {
    if (lhs.kind == ShapeKind::Scalar && rhs.kind == ShapeKind::Scalar) {
        return to_python(fn(lhs.scalar, rhs.scalar));
    }
    if (lhs.kind == ShapeKind::Scalar) {
        return apply_unary(rhs, [&](double value) { return fn(lhs.scalar, value); });
    }
    if (rhs.kind == ShapeKind::Scalar) {
        return apply_unary(lhs, [&](double value) { return fn(value, rhs.scalar); });
    }
    if (lhs.kind == ShapeKind::Vector && rhs.kind == ShapeKind::Vector) {
        std::size_t length = std::min(lhs.vector.size(), rhs.vector.size());
        Vector out;
        out.reserve(length);
        for (std::size_t i = 0; i < length; ++i) {
            out.push_back(fn(lhs.vector[i], rhs.vector[i]));
        }
        return to_python(out);
    }
    if (lhs.kind == ShapeKind::Matrix && rhs.kind == ShapeKind::Matrix) {
        std::size_t rows = std::min(lhs.matrix.size(), rhs.matrix.size());
        Matrix out;
        out.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            std::size_t cols = std::min(lhs.matrix[row].size(), rhs.matrix[row].size());
            Vector out_row;
            out_row.reserve(cols);
            for (std::size_t col = 0; col < cols; ++col) {
                out_row.push_back(fn(lhs.matrix[row][col], rhs.matrix[row][col]));
            }
            out.push_back(std::move(out_row));
        }
        return to_python(out);
    }
    throw std::invalid_argument("Mismatched shapes for binary operation");
}

double trace_value(const Matrix &mat) {
    if (mat.empty()) {
        return 0.0;
    }
    std::size_t rows = mat.size();
    std::size_t cols = 0;
    for (const auto &row : mat) {
        cols = std::max(cols, row.size());
    }
    std::size_t diag = std::min(rows, cols);
    double sum = 0.0;
    for (std::size_t i = 0; i < diag; ++i) {
        if (i < mat[i].size()) {
            sum += mat[i][i];
        }
    }
    return sum;
}

Matrix identity_matrix(std::size_t n) {
    Matrix ident(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        ident[i][i] = 1.0;
    }
    return ident;
}

Matrix augment_matrix(const Matrix &mat, const Matrix &identity) {
    Matrix augmented = mat;
    for (std::size_t i = 0; i < augmented.size(); ++i) {
        augmented[i].insert(augmented[i].end(), identity[i].begin(), identity[i].end());
    }
    return augmented;
}

Matrix inverse_matrix(const Matrix &mat) {
    std::size_t n = mat.size();
    for (const auto &row : mat) {
        if (row.size() != n) {
            throw std::invalid_argument("Matrix must be square");
        }
    }
    Matrix augmented = augment_matrix(mat, identity_matrix(n));
    for (std::size_t i = 0; i < n; ++i) {
        double pivot = augmented[i][i];
        if (std::abs(pivot) < 1e-12) {
            std::size_t swap_row = n;
            for (std::size_t j = i + 1; j < n; ++j) {
                if (std::abs(augmented[j][i]) > 1e-12) {
                    swap_row = j;
                    break;
                }
            }
            if (swap_row == n) {
                throw std::invalid_argument("Matrix is singular");
            }
            std::swap(augmented[i], augmented[swap_row]);
            pivot = augmented[i][i];
        }
        double pivot_inv = 1.0 / pivot;
        for (double &value : augmented[i]) {
            value *= pivot_inv;
        }
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            double factor = augmented[j][i];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t k = 0; k < augmented[j].size(); ++k) {
                augmented[j][k] -= factor * augmented[i][k];
            }
        }
    }
    Matrix inverse(n, Vector(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        inverse[i].assign(augmented[i].begin() + static_cast<std::ptrdiff_t>(n), augmented[i].end());
    }
    return inverse;
}

std::pair<double, double> slogdet_matrix(Matrix mat) {
    std::size_t n = mat.size();
    for (const auto &row : mat) {
        if (row.size() != n) {
            throw std::invalid_argument("Matrix must be square");
        }
    }
    double sign = 1.0;
    double log_abs_det = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t pivot_row = i;
        double max_val = std::abs(mat[i][i]);
        for (std::size_t r = i + 1; r < n; ++r) {
            double candidate = std::abs(mat[r][i]);
            if (candidate > max_val) {
                max_val = candidate;
                pivot_row = r;
            }
        }
        if (max_val < 1e-12) {
            return {0.0, -std::numeric_limits<double>::infinity()};
        }
        if (pivot_row != i) {
            std::swap(mat[i], mat[pivot_row]);
            sign *= -1.0;
        }
        double pivot = mat[i][i];
        if (pivot < 0) {
            sign *= -1.0;
        }
        log_abs_det += std::log(std::abs(pivot));
        for (std::size_t row = i + 1; row < n; ++row) {
            double factor = mat[row][i] / pivot;
            for (std::size_t col = i; col < n; ++col) {
                mat[row][col] -= factor * mat[i][col];
            }
        }
    }
    return {sign, log_abs_det};
}

}  // namespace

py::object matmul(py::handle a, py::handle b) {
    ParsedSequence lhs = parse_sequence(a);
    ParsedSequence rhs = parse_sequence(b);
    if (lhs.kind == ShapeKind::Scalar && rhs.kind == ShapeKind::Scalar) {
        return to_python(lhs.scalar * rhs.scalar);
    }
    if (lhs.kind == ShapeKind::Vector && rhs.kind == ShapeKind::Vector) {
        std::size_t length = std::min(lhs.vector.size(), rhs.vector.size());
        double result = 0.0;
        for (std::size_t i = 0; i < length; ++i) {
            result += lhs.vector[i] * rhs.vector[i];
        }
        return to_python(result);
    }
    if (lhs.kind == ShapeKind::Matrix && rhs.kind == ShapeKind::Vector) {
        Vector result;
        result.reserve(lhs.matrix.size());
        for (const auto &row : lhs.matrix) {
            std::size_t length = std::min(row.size(), rhs.vector.size());
            double value = 0.0;
            for (std::size_t i = 0; i < length; ++i) {
                value += row[i] * rhs.vector[i];
            }
            result.push_back(value);
        }
        return to_python(result);
    }
    if (lhs.kind == ShapeKind::Matrix && rhs.kind == ShapeKind::Matrix) {
        std::size_t m = lhs.matrix.size();
        std::size_t n = lhs.matrix.empty() ? 0 : lhs.matrix.front().size();
        std::size_t p = rhs.matrix.empty() ? 0 : rhs.matrix.front().size();
        Matrix result(m, Vector(p, 0.0));
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t k = 0; k < n; ++k) {
                if (k >= rhs.matrix.size()) {
                    continue;
                }
                double aik = lhs.matrix[i][k];
                std::size_t cols = std::min(p, rhs.matrix[k].size());
                for (std::size_t j = 0; j < cols; ++j) {
                    result[i][j] += aik * rhs.matrix[k][j];
                }
            }
        }
        return to_python(result);
    }
    if (lhs.kind == ShapeKind::Vector && rhs.kind == ShapeKind::Matrix) {
        std::size_t rows = rhs.matrix.size();
        Vector result;
        result.reserve(rows ? rhs.matrix.front().size() : 0);
        if (!rhs.matrix.empty()) {
            std::size_t cols = rhs.matrix.front().size();
            result.assign(cols, 0.0);
            for (std::size_t k = 0; k < std::min(lhs.vector.size(), rhs.matrix.size()); ++k) {
                for (std::size_t j = 0; j < std::min(cols, rhs.matrix[k].size()); ++j) {
                    result[j] += lhs.vector[k] * rhs.matrix[k][j];
                }
            }
        }
        return to_python(result);
    }
    if (lhs.kind == ShapeKind::Scalar) {
        return apply_unary(rhs, [&](double value) { return lhs.scalar * value; });
    }
    if (rhs.kind == ShapeKind::Scalar) {
        return apply_unary(lhs, [&](double value) { return value * rhs.scalar; });
    }
    throw std::invalid_argument("Unsupported operands for matmul");
}

py::object dot(py::handle a, py::handle b) {
    ParsedSequence lhs = parse_sequence(a);
    ParsedSequence rhs = parse_sequence(b);
    Vector left = flatten(lhs);
    Vector right = flatten(rhs);
    std::size_t length = std::min(left.size(), right.size());
    double result = 0.0;
    for (std::size_t i = 0; i < length; ++i) {
        result += left[i] * right[i];
    }
    return to_python(result);
}

py::object mean(py::handle data, py::object axis) {
    ParsedSequence parsed = parse_sequence(data);
    if (axis.is_none()) {
        Vector values = flatten(parsed);
        return to_python(compute_mean(values));
    }
    int axis_value = axis.cast<int>();
    if (parsed.kind == ShapeKind::Scalar) {
        if (axis_value == 0) {
            return to_python(Vector{parsed.scalar});
        }
        throw std::invalid_argument("axis out of bounds for scalar input");
    }
    if (parsed.kind == ShapeKind::Vector) {
        if (axis_value == 0) {
            return to_python(Vector{compute_mean(parsed.vector)});
        }
        throw std::invalid_argument("axis out of bounds for 1D input");
    }
    if (axis_value == 0) {
        return to_python(mean_axis0(parsed.matrix));
    }
    if (axis_value == 1) {
        return to_python(mean_axis1(parsed.matrix));
    }
    throw std::invalid_argument("Unsupported axis for mean");
}

py::object std(py::handle data, py::object axis) {
    ParsedSequence parsed = parse_sequence(data);
    if (axis.is_none()) {
        return to_python(compute_std(flatten(parsed)));
    }
    int axis_value = axis.cast<int>();
    if (parsed.kind == ShapeKind::Scalar) {
        if (axis_value == 0) {
            return to_python(Vector{0.0});
        }
        throw std::invalid_argument("axis out of bounds for scalar input");
    }
    if (parsed.kind == ShapeKind::Vector) {
        if (axis_value == 0) {
            return to_python(Vector{compute_std(parsed.vector)});
        }
        throw std::invalid_argument("axis out of bounds for 1D input");
    }
    if (axis_value == 0) {
        return to_python(std_axis0(parsed.matrix));
    }
    if (axis_value == 1) {
        return to_python(std_axis1(parsed.matrix));
    }
    throw std::invalid_argument("Unsupported axis for std");
}

py::object sum(py::handle data, py::object axis, bool keepdims) {
    ParsedSequence parsed = parse_sequence(data);
    if (axis.is_none()) {
        return to_python(compute_sum(flatten(parsed)));
    }
    int axis_value = axis.cast<int>();
    if (parsed.kind == ShapeKind::Scalar) {
        if (axis_value == 0) {
            return keepdims ? to_python(Matrix{Vector{parsed.scalar}}) : to_python(Vector{parsed.scalar});
        }
        throw std::invalid_argument("axis out of bounds for scalar input");
    }
    if (parsed.kind == ShapeKind::Vector) {
        if (axis_value != 0) {
            throw std::invalid_argument("axis out of bounds for 1D input");
        }
        double value = compute_sum(parsed.vector);
        if (keepdims) {
            return to_python(Matrix{Vector{value}});
        }
        return to_python(Vector{value});
    }
    if (axis_value == 0) {
        Vector values = sum_axis0(parsed.matrix);
        if (keepdims) {
            return to_python(Matrix{values});
        }
        return to_python(values);
    }
    if (axis_value == 1) {
        Vector values = sum_axis1(parsed.matrix);
        if (keepdims) {
            Matrix mat;
            mat.reserve(values.size());
            for (double value : values) {
                mat.push_back(Vector{value});
            }
            return to_python(mat);
        }
        return to_python(values);
    }
    throw std::invalid_argument("Unsupported axis for sum");
}

py::object tanh(py::handle data) {
    return apply_unary(parse_sequence(data), [](double value) { return std::tanh(value); });
}

py::object exp(py::handle data) {
    return apply_unary(parse_sequence(data), [](double value) { return std::exp(value); });
}

py::object log(py::handle data) {
    return apply_unary(parse_sequence(data), [](double value) { return std::log(value); });
}

py::object logaddexp(py::handle a, py::handle b) {
    auto combine = [](double x, double y) {
        double m = std::max(x, y);
        return m + std::log(std::exp(x - m) + std::exp(y - m));
    };
    return apply_binary(parse_sequence(a), parse_sequence(b), combine);
}

py::object median(py::handle data, py::object axis) {
    ParsedSequence parsed = parse_sequence(data);
    if (axis.is_none()) {
        return to_python(compute_median(flatten(parsed)));
    }
    int axis_value = axis.cast<int>();
    if (parsed.kind == ShapeKind::Scalar) {
        if (axis_value == 0) {
            return to_python(Vector{parsed.scalar});
        }
        throw std::invalid_argument("axis out of bounds for scalar input");
    }
    if (parsed.kind == ShapeKind::Vector) {
        if (axis_value == 0) {
            return to_python(Vector{compute_median(parsed.vector)});
        }
        throw std::invalid_argument("axis out of bounds for 1D input");
    }
    if (axis_value == 0) {
        return to_python(median_axis0(parsed.matrix));
    }
    if (axis_value == 1) {
        return to_python(median_axis1(parsed.matrix));
    }
    throw std::invalid_argument("Unsupported axis for median");
}

py::object abs(py::handle data) {
    return apply_unary(parse_sequence(data), [](double value) { return std::abs(value); });
}

py::object clip(py::handle data, double lo, double hi) {
    if (lo > hi) {
        std::swap(lo, hi);
    }
    return apply_unary(parse_sequence(data), [&](double value) { return std::clamp(value, lo, hi); });
}

py::object sqrt(py::handle data) {
    return apply_unary(parse_sequence(data), [](double value) { return std::sqrt(value); });
}

py::object diff(py::handle data, int order) {
    if (order < 0) {
        throw std::invalid_argument("diff order must be non-negative");
    }
    ParsedSequence parsed = parse_sequence(data);
    if (order == 0) {
        switch (parsed.kind) {
        case ShapeKind::Scalar:
            return to_python(parsed.scalar);
        case ShapeKind::Vector:
            return to_python(parsed.vector);
        case ShapeKind::Matrix:
            return to_python(parsed.matrix);
        }
    }
    if (parsed.kind == ShapeKind::Scalar) {
        return to_python(Vector());
    }
    if (parsed.kind == ShapeKind::Vector) {
        return to_python(diff_vector(parsed.vector, order));
    }
    return to_python(diff_matrix(parsed.matrix, order));
}

py::object argsort(py::handle data) {
    Vector values = flatten(parse_sequence(data));
    std::vector<std::size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](std::size_t lhs, std::size_t rhs) {
        return values[lhs] < values[rhs];
    });
    py::list result;
    for (std::size_t index : indices) {
        result.append(py::int_(static_cast<long>(index)));
    }
    return result;
}

py::object argmax(py::handle data) {
    Vector values = flatten(parse_sequence(data));
    if (values.empty()) {
        return py::int_(0);
    }
    std::size_t best = 0;
    double best_value = values[0];
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > best_value) {
            best = i;
            best_value = values[i];
        }
    }
    return py::int_(static_cast<long>(best));
}

py::object trace(py::handle data) {
    Matrix mat = ensure_matrix(parse_sequence(data));
    return to_python(trace_value(mat));
}

py::object linalg_norm(py::handle data) {
    Vector values = flatten(parse_sequence(data));
    double sum_sq = 0.0;
    for (double value : values) {
        sum_sq += value * value;
    }
    return to_python(std::sqrt(sum_sq));
}

py::object linalg_inv(py::handle data) {
    Matrix mat = ensure_matrix(parse_sequence(data));
    if (mat.empty()) {
        return to_python(Matrix());
    }
    return to_python(inverse_matrix(mat));
}

py::object linalg_slogdet(py::handle data) {
    Matrix mat = ensure_matrix(parse_sequence(data));
    if (mat.empty()) {
        return py::make_tuple(1.0, -std::numeric_limits<double>::infinity());
    }
    auto result = slogdet_matrix(mat);
    return py::make_tuple(result.first, result.second);
}

PYBIND11_MODULE(spiral_numeric_cpp, m) {
    m.doc() = "High-performance numeric helpers for Spiral Reality";
    m.def("matmul", &matmul, "Matrix multiplication");
    m.def("dot", &dot, "Dot product");
    m.def("mean", &mean, "Mean reduction", py::arg("data"), py::arg("axis") = py::none());
    m.def("std", &std, "Standard deviation", py::arg("data"), py::arg("axis") = py::none());
    m.def("sum", &sum, "Sum reduction", py::arg("data"), py::arg("axis") = py::none(), py::arg("keepdims") = false);
    m.def("tanh", &tanh, "Hyperbolic tangent");
    m.def("exp", &exp, "Exponential");
    m.def("log", &log, "Natural logarithm");
    m.def("logaddexp", &logaddexp, "Log-add-exp");
    m.def("median", &median, "Median", py::arg("data"), py::arg("axis") = py::none());
    m.def("abs", &abs, "Absolute value");
    m.def("clip", &clip, "Clip values", py::arg("data"), py::arg("lo"), py::arg("hi"));
    m.def("sqrt", &sqrt, "Square root");
    m.def("diff", &diff, "Discrete difference", py::arg("data"), py::arg("order") = 1);
    m.def("argsort", &argsort, "Argsort indices");
    m.def("argmax", &argmax, "Argmax index");
    m.def("trace", &trace, "Trace value");
    m.def("linalg_norm", &linalg_norm, "Vector norm");
    m.def("linalg_inv", &linalg_inv, "Matrix inverse");
    m.def("linalg_slogdet", &linalg_slogdet, "Sign and log determinant");
}
