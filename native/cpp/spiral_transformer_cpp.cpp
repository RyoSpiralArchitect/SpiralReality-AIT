#include <pybind11/attr.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace py = pybind11;

namespace {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

constexpr const char *kBackendKind = "cpp-transformer";

std::size_t slice_length(const Vector &vec, std::size_t offset, std::size_t width) {
    if (offset >= vec.size()) {
        return 0;
    }
    return std::min(width, vec.size() - offset);
}

double dot_simd(const double *lhs, const double *rhs, std::size_t length) {
    double result = 0.0;
#if defined(__AVX2__)
    std::size_t i = 0;
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= length; i += 4) {
        __m256d va = _mm256_loadu_pd(lhs + i);
        __m256d vb = _mm256_loadu_pd(rhs + i);
        acc = _mm256_fmadd_pd(va, vb, acc);
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc);
    result += tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < length; ++i) {
        result += lhs[i] * rhs[i];
    }
#else
    for (std::size_t i = 0; i < length; ++i) {
        result += lhs[i] * rhs[i];
    }
#endif
    return result;
}

double dot_simd_slice(const Vector &lhs, std::size_t lhs_offset, const Vector &rhs, std::size_t rhs_offset,
                      std::size_t width) {
    const std::size_t lhs_len = slice_length(lhs, lhs_offset, width);
    const std::size_t rhs_len = slice_length(rhs, rhs_offset, width);
    const std::size_t length = std::min(lhs_len, rhs_len);
    if (length == 0) {
        return 0.0;
    }
    return dot_simd(lhs.data() + lhs_offset, rhs.data() + rhs_offset, length);
}

void axpy_simd(double weight, const double *src, double *dst, std::size_t length) {
#if defined(__AVX2__)
    std::size_t i = 0;
    const __m256d scale = _mm256_set1_pd(weight);
    for (; i + 4 <= length; i += 4) {
        __m256d vdst = _mm256_loadu_pd(dst + i);
        __m256d vsrc = _mm256_loadu_pd(src + i);
        vdst = _mm256_fmadd_pd(vsrc, scale, vdst);
        _mm256_storeu_pd(dst + i, vdst);
    }
    for (; i < length; ++i) {
        dst[i] += weight * src[i];
    }
#else
    for (std::size_t i = 0; i < length; ++i) {
        dst[i] += weight * src[i];
    }
#endif
}

void axpy_simd_slice(double weight, const Vector &src, std::size_t src_offset, std::size_t width, Vector &dst,
                     std::size_t dst_offset) {
    const std::size_t src_len = slice_length(src, src_offset, width);
    const std::size_t dst_len = slice_length(dst, dst_offset, width);
    const std::size_t length = std::min(src_len, dst_len);
    if (length == 0) {
        return;
    }
    axpy_simd(weight, src.data() + src_offset, dst.data() + dst_offset, length);
}

void scale_slice(Vector &vec, std::size_t offset, std::size_t width, double factor) {
    if (factor == 1.0) {
        return;
    }
    const std::size_t length = slice_length(vec, offset, width);
    for (std::size_t i = 0; i < length; ++i) {
        vec[offset + i] *= factor;
    }
}

#ifdef SPIRAL_HAS_CUDA
constexpr bool kSupportsCuda = true;
#else
constexpr bool kSupportsCuda = false;
#endif

#ifdef SPIRAL_HAS_HIP
constexpr bool kSupportsRocm = true;
#else
constexpr bool kSupportsRocm = false;
#endif

#ifdef SPIRAL_HAS_METAL
constexpr bool kSupportsMps = true;
#else
constexpr bool kSupportsMps = false;
#endif

std::vector<std::string> compute_available_devices() {
    std::vector<std::string> devices{"cpu"};
    if (kSupportsCuda) {
        devices.emplace_back("cuda");
    }
    if (kSupportsRocm) {
        devices.emplace_back("rocm");
    }
    if (kSupportsMps) {
        devices.emplace_back("mps");
    }
    return devices;
}

std::string trim_copy(const std::string &value) {
    const std::string whitespace = " \t\r\n";
    const std::size_t first = value.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return std::string();
    }
    const std::size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

std::string to_lower_copy(const std::string &value) {
    std::string lowered = value;
    for (char &ch : lowered) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return lowered;
}

std::string first_non_cpu_device(const std::vector<std::string> &devices) {
    for (const auto &device : devices) {
        if (to_lower_copy(device) != "cpu") {
            return device;
        }
    }
    if (!devices.empty()) {
        return devices.front();
    }
    return std::string("cpu");
}

std::string match_device_token(const std::vector<std::string> &devices, const std::string &token) {
    if (token.empty()) {
        return std::string();
    }
    std::string lowered = to_lower_copy(token);
    const std::size_t colon = lowered.find(':');
    if (colon != std::string::npos) {
        lowered = lowered.substr(0, colon);
    }
    for (const auto &candidate : devices) {
        if (to_lower_copy(candidate) == lowered) {
            return candidate;
        }
    }
    return std::string();
}

std::string resolve_device_request(const std::vector<std::string> &devices, const std::string &request,
                                   bool strict) {
    const std::string trimmed = trim_copy(request);
    const std::string lowered = to_lower_copy(trimmed);
    if (trimmed.empty() || lowered == "auto" || lowered == "default") {
        return first_non_cpu_device(devices);
    }
    if (lowered == "gpu" || lowered == "accelerator" || lowered == "best") {
        return first_non_cpu_device(devices);
    }
    const std::string matched = match_device_token(devices, lowered);
    if (!matched.empty()) {
        return matched;
    }
    if (strict) {
        throw std::invalid_argument("Unsupported device: " + request);
    }
    return std::string();
}

std::string environment_device_preference(const std::vector<std::string> &devices) {
    const char *keys[] = {"SPIRAL_TRANSFORMER_DEVICE", "SPIRAL_DEVICE", "SPIRAL_DEFAULT_DEVICE"};
    for (const char *key : keys) {
        if (key == nullptr) {
            continue;
        }
        const char *value = std::getenv(key);
        if (value == nullptr) {
            continue;
        }
        const std::string requested = trim_copy(std::string(value));
        if (requested.empty()) {
            continue;
        }
        try {
            const std::string resolved = resolve_device_request(devices, requested, false);
            if (!resolved.empty()) {
                return resolved;
            }
        } catch (const std::exception &) {
            // Ignore invalid environment overrides and fall back to discovery.
        }
    }
    return std::string();
}

std::string select_default_device(const std::vector<std::string> &devices) {
    const std::string env_pref = environment_device_preference(devices);
    if (!env_pref.empty()) {
        return env_pref;
    }
    return first_non_cpu_device(devices);
}

Matrix zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, Vector(cols, 0.0));
}

Matrix matmul(const Matrix &a, const Matrix &b) {
    if (a.empty() || b.empty()) {
        return Matrix();
    }
    std::size_t m = a.size();
    std::size_t n = a.front().size();
    std::size_t p = b.front().size();
    Matrix result(m, Vector(p, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            double aik = a[i][k];
            for (std::size_t j = 0; j < p; ++j) {
                result[i][j] += aik * b[k][j];
            }
        }
    }
    return result;
}

Matrix add(const Matrix &a, const Matrix &b) {
    if (a.empty()) {
        return b;
    }
    Matrix result = a;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            result[i][j] += (i < b.size() && j < b[i].size()) ? b[i][j] : 0.0;
        }
    }
    return result;
}

Matrix add_bias(const Matrix &a, const Vector &bias) {
    if (a.empty()) {
        return a;
    }
    Matrix result = a;
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a[i].size(); ++j) {
            double bval = (j < bias.size()) ? bias[j] : bias.empty() ? 0.0 : bias.back();
            result[i][j] += bval;
        }
    }
    return result;
}

Matrix scale_matrix(const Matrix &a, double factor) {
    Matrix result = a;
    for (auto &row : result) {
        for (auto &val : row) {
            val *= factor;
        }
    }
    return result;
}

Matrix tanh_matrix(const Matrix &a) {
    Matrix result = a;
    for (auto &row : result) {
        for (auto &val : row) {
            val = std::tanh(val);
        }
    }
    return result;
}

Vector random_vector(std::size_t size, double scale, std::mt19937 &rng) {
    std::normal_distribution<double> dist(0.0, scale);
    Vector out(size, 0.0);
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = dist(rng);
    }
    return out;
}

Matrix random_matrix(std::size_t rows, std::size_t cols, double scale, std::mt19937 &rng) {
    Matrix m(rows, Vector(cols, 0.0));
    std::normal_distribution<double> dist(0.0, scale);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            m[i][j] = dist(rng);
        }
    }
    return m;
}

Vector layer_norm_stats(const Vector &row) {
    Vector stats(2, 0.0);
    if (row.empty()) {
        return stats;
    }
    double mean = 0.0;
    for (double v : row) {
        mean += v;
    }
    mean /= static_cast<double>(row.size());
    double var = 0.0;
    for (double v : row) {
        double diff = v - mean;
        var += diff * diff;
    }
    var /= static_cast<double>(row.size());
    stats[0] = mean;
    stats[1] = std::sqrt(var + 1e-5);
    return stats;
}

Matrix layer_norm(const Matrix &h, const Vector &gamma, const Vector &beta) {
    Matrix result = h;
    for (std::size_t i = 0; i < h.size(); ++i) {
        const Vector stats = layer_norm_stats(h[i]);
        double mean = stats[0];
        double denom = stats[1];
        for (std::size_t j = 0; j < h[i].size(); ++j) {
            double normed = (h[i][j] - mean) / denom;
            double g = (j < gamma.size()) ? gamma[j] : gamma.empty() ? 1.0 : gamma.back();
            double b = (j < beta.size()) ? beta[j] : beta.empty() ? 0.0 : beta.back();
            result[i][j] = g * normed + b;
        }
    }
    return result;
}

Matrix to_matrix(const py::array_t<double> &array) {
    Matrix result;
    if (!array) {
        return result;
    }
    py::buffer_info info = array.request();
    if (info.ndim == 1) {
        Vector row(info.shape[0], 0.0);
        const double *ptr = static_cast<double *>(info.ptr);
        for (ssize_t i = 0; i < info.shape[0]; ++i) {
            row[i] = ptr[i];
        }
        result.push_back(std::move(row));
        return result;
    }
    if (info.ndim != 2) {
        return result;
    }
    const double *ptr = static_cast<double *>(info.ptr);
    for (ssize_t i = 0; i < info.shape[0]; ++i) {
        Vector row(info.shape[1], 0.0);
        for (ssize_t j = 0; j < info.shape[1]; ++j) {
            row[j] = ptr[i * info.shape[1] + j];
        }
        result.push_back(std::move(row));
    }
    return result;
}

Vector to_vector(const py::array_t<double> &array) {
    Vector result;
    if (!array) {
        return result;
    }
    py::buffer_info info = array.request();
    if (info.ndim == 0) {
        return result;
    }
    if (info.ndim == 1) {
        result.resize(info.shape[0]);
        const double *ptr = static_cast<double *>(info.ptr);
        for (ssize_t i = 0; i < info.shape[0]; ++i) {
            result[i] = ptr[i];
        }
        return result;
    }
    const double *ptr = static_cast<double *>(info.ptr);
    std::size_t count = 1;
    for (ssize_t d = 0; d < info.ndim; ++d) {
        count *= static_cast<std::size_t>(info.shape[d]);
    }
    result.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = ptr[i];
    }
    return result;
}

Matrix from_nested_iterable(const py::object &obj) {
    Matrix result;
    if (obj.is_none()) {
        return result;
    }
    try {
        for (const py::handle &row_obj : obj) {
            Vector row;
            for (const py::handle &val : py::reinterpret_borrow<py::iterable>(row_obj)) {
                row.push_back(py::cast<double>(val));
            }
            result.push_back(std::move(row));
        }
    } catch (const py::cast_error &) {
        result.clear();
    }
    return result;
}

py::array_t<double> to_array(const Matrix &matrix) {
    if (matrix.empty()) {
        return py::array_t<double>({0, 0});
    }
    ssize_t rows = static_cast<ssize_t>(matrix.size());
    ssize_t cols = static_cast<ssize_t>(matrix.front().size());
    py::array_t<double> result({rows, cols});
    py::buffer_info info = result.request();
    double *ptr = static_cast<double *>(info.ptr);
    for (ssize_t i = 0; i < rows; ++i) {
        for (ssize_t j = 0; j < cols; ++j) {
            ptr[i * cols + j] = (j < static_cast<ssize_t>(matrix[i].size())) ? matrix[i][j] : 0.0;
        }
    }
    return result;
}

py::array_t<double> to_vector_array(const Vector &vec) {
    ssize_t size = static_cast<ssize_t>(vec.size());
    py::array_t<double> result(size);
    py::buffer_info info = result.request();
    double *ptr = static_cast<double *>(info.ptr);
    for (ssize_t i = 0; i < size; ++i) {
        ptr[i] = vec[i];
    }
    return result;
}

}  // namespace

class CppTransformerAdapter {
   public:
    CppTransformerAdapter(int d_model = 128, int n_layers = 4, int n_heads = 4,
                          double ff_multiplier = 4.0, int seed = 2025)
        : d_model_(d_model),
          n_layers_(n_layers),
          n_heads_(n_heads),
          ff_multiplier_(ff_multiplier),
          head_dim_(d_model / std::max(1, n_heads)),
          device_("cpu"),
          backend_(kBackendKind),
          available_devices_(compute_available_devices()) {
        if (n_heads <= 0) {
            throw std::invalid_argument("n_heads must be positive");
        }
        if (d_model % n_heads != 0) {
            throw std::invalid_argument("d_model must be divisible by n_heads");
        }
        head_dim_ = d_model / n_heads;
        ff_dim_ = std::max(head_dim_ * n_heads, static_cast<int>(std::round(d_model * ff_multiplier_)));
        std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
        double scale = 1.0 / std::sqrt(static_cast<double>(d_model));
        if (available_devices_.empty()) {
            available_devices_.push_back("cpu");
        }
        device_ = select_default_device(available_devices_);
        for (int layer = 0; layer < n_layers_; ++layer) {
            Wq_.push_back(random_matrix(d_model_, d_model_, scale, rng));
            Wk_.push_back(random_matrix(d_model_, d_model_, scale, rng));
            Wv_.push_back(random_matrix(d_model_, d_model_, scale, rng));
            Wo_.push_back(random_matrix(d_model_, d_model_, scale, rng));
            Wff1_.push_back(random_matrix(d_model_, ff_dim_, scale, rng));
            bff1_.push_back(random_vector(ff_dim_, scale, rng));
            Wff2_.push_back(random_matrix(ff_dim_, d_model_, scale, rng));
            bff2_.push_back(random_vector(d_model_, scale, rng));
            ln1_gamma_.push_back(Vector(d_model_, 1.0));
            ln1_beta_.push_back(Vector(d_model_, 0.0));
            ln2_gamma_.push_back(Vector(d_model_, 1.0));
            ln2_beta_.push_back(Vector(d_model_, 0.0));
            gate_bias_.push_back({0.0, 0.0});
            ff_gate_.push_back({0.0, 0.0});
        }
        last_attn_ = py::list();
        last_gate_mask_ = py::array_t<double>({0, 0});
    }

    py::array_t<double> forward(const py::array_t<double> &X_in, const py::array_t<double> &gate_pos_in,
                                 py::object gate_mask_obj = py::none()) {
        Matrix X = to_matrix(X_in);
        if (X.empty()) {
            last_attn_ = py::list();
            last_gate_mask_ = py::array_t<double>({0, 0});
            return py::array_t<double>({0, d_model_});
        }
        Vector gate_pos_vec;
        {
            py::buffer_info info = gate_pos_in.request();
            const double *ptr = static_cast<double *>(info.ptr);
            for (ssize_t i = 0; i < info.shape[0]; ++i) {
                gate_pos_vec.push_back(ptr[i]);
            }
        }
        Matrix gate_mask;
        if (!gate_mask_obj.is_none()) {
            if (py::isinstance<py::array>(gate_mask_obj)) {
                gate_mask = to_matrix(py::cast<py::array_t<double>>(gate_mask_obj));
            } else {
                gate_mask = from_nested_iterable(gate_mask_obj);
            }
        }
        if (gate_mask.empty()) {
            std::size_t len = gate_pos_vec.size();
            gate_mask = Matrix(len, Vector(len, 0.0));
            for (std::size_t i = 0; i < len; ++i) {
                for (std::size_t j = 0; j < len; ++j) {
                    gate_mask[i][j] = std::min(gate_pos_vec[i], gate_pos_vec[j]);
                }
            }
        }
        Matrix outer = gate_mask;
        for (std::size_t i = 0; i < outer.size(); ++i) {
            for (std::size_t j = 0; j < outer[i].size(); ++j) {
                outer[i][j] = (i < gate_pos_vec.size() && j < gate_pos_vec.size())
                                   ? gate_pos_vec[i] * gate_pos_vec[j]
                                   : 0.0;
            }
        }

        double gate_mean = 0.0;
        double gate_std = 0.0;
        if (!gate_pos_vec.empty()) {
            for (double v : gate_pos_vec) {
                gate_mean += v;
            }
            gate_mean /= static_cast<double>(gate_pos_vec.size());
            double var = 0.0;
            for (double v : gate_pos_vec) {
                double diff = v - gate_mean;
                var += diff * diff;
            }
            var /= static_cast<double>(gate_pos_vec.size());
            gate_std = std::sqrt(std::max(0.0, var));
        }
        double mask_energy = 0.0;
        std::size_t mask_count = 0;
        for (const auto &row : gate_mask) {
            for (double v : row) {
                mask_energy += v;
                ++mask_count;
            }
        }
        if (mask_count > 0) {
            mask_energy /= static_cast<double>(mask_count);
        }

        Matrix H = X;
        last_attn_ = py::list();
        for (int layer = 0; layer < n_layers_; ++layer) {
            Matrix norm_in = layer_norm(H, ln1_gamma_[layer], ln1_beta_[layer]);
            Matrix Q = matmul(norm_in, Wq_[layer]);
            Matrix K = matmul(norm_in, Wk_[layer]);
            Matrix V = matmul(norm_in, Wv_[layer]);

            std::size_t seq_len = Q.size();
            std::size_t key_len = K.size();
            double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));

            std::vector<Matrix> attn_heads(n_heads_, Matrix(seq_len, Vector(key_len, 0.0)));
            Matrix context(seq_len, Vector(d_model_, 0.0));
            std::vector<double> score_buffer(key_len, 0.0);
            const std::size_t block = std::max<std::size_t>(1, std::min<std::size_t>(key_len, static_cast<std::size_t>(64)));
            for (int head = 0; head < n_heads_; ++head) {
                std::size_t head_offset = static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim_);
                Matrix &head_weights = attn_heads[head];
                for (std::size_t i = 0; i < seq_len; ++i) {
                    Vector &weights_row = head_weights[i];
                    std::fill(weights_row.begin(), weights_row.end(), 0.0);
                    Vector &context_row = context[i];
                    double m_i = -std::numeric_limits<double>::infinity();
                    double l_i = 0.0;
                    for (std::size_t start = 0; start < key_len; start += block) {
                        const std::size_t end = std::min(start + block, key_len);
                        double block_max = -std::numeric_limits<double>::infinity();
                        for (std::size_t j = start; j < end; ++j) {
                            double dot = dot_simd_slice(Q[i], head_offset, K[j], head_offset, head_dim_);
                            double bias = gate_bias_[layer][0] *
                                              ((i < outer.size() && j < outer[i].size()) ? outer[i][j] : 0.0) +
                                          gate_bias_[layer][1] *
                                              ((i < gate_mask.size() && j < gate_mask[i].size()) ? gate_mask[i][j] : 0.0);
                            double score = dot * scale + bias;
                            score_buffer[j] = score;
                            block_max = std::max(block_max, score);
                        }
                        const double new_m = std::max(m_i, block_max);
                        const double exp_scale = std::isinf(m_i) ? 0.0 : std::exp(m_i - new_m);
                        if (!std::isinf(m_i)) {
                            for (double &weight : weights_row) {
                                weight *= exp_scale;
                            }
                            scale_slice(context_row, head_offset, head_dim_, exp_scale);
                            l_i *= exp_scale;
                        }
                        m_i = new_m;
                        for (std::size_t j = start; j < end; ++j) {
                            double weight = std::exp(score_buffer[j] - m_i);
                            weights_row[j] += weight;
                            l_i += weight;
                            axpy_simd_slice(weight, V[j], head_offset, head_dim_, context_row, head_offset);
                        }
                    }
                    const double norm = (l_i <= 0.0) ? 0.0 : 1.0 / l_i;
                    for (double &weight : weights_row) {
                        weight *= norm;
                    }
                    scale_slice(context_row, head_offset, head_dim_, norm);
                }
            }

            Matrix attn_out = matmul(context, Wo_[layer]);
            H = add(H, attn_out);

            Matrix ff_in = layer_norm(H, ln2_gamma_[layer], ln2_beta_[layer]);
            Matrix ff_hidden = add_bias(matmul(ff_in, Wff1_[layer]), bff1_[layer]);
            ff_hidden = tanh_matrix(ff_hidden);
            Matrix ff_out = add_bias(matmul(ff_hidden, Wff2_[layer]), bff2_[layer]);
            double modulation = 1.0 + ff_gate_[layer][0] * gate_mean + ff_gate_[layer][1] * (gate_std + mask_energy);
            H = add(H, scale_matrix(ff_out, modulation));

            Matrix attn_mean(seq_len, Vector(key_len, 0.0));
            for (std::size_t i = 0; i < seq_len; ++i) {
                for (std::size_t j = 0; j < key_len; ++j) {
                    double sum = 0.0;
                    for (int head = 0; head < n_heads_; ++head) {
                        sum += attn_heads[head][i][j];
                    }
                    attn_mean[i][j] = sum / static_cast<double>(std::max(1, n_heads_));
                }
            }
            last_attn_.attr("append")(to_array(attn_mean));
        }
        last_gate_mask_ = to_array(gate_mask);
        return to_array(H);
    }

    void tune_from_boundary(const py::iterable &base_gate, const py::iterable &targets, double lr = 1e-3) {
        Vector base_vals;
        Vector target_vals;
        for (const auto &item : base_gate) {
            base_vals.push_back(py::cast<double>(item));
        }
        for (const auto &item : targets) {
            target_vals.push_back(py::cast<double>(item));
        }
        if (base_vals.empty() || base_vals.size() != target_vals.size()) {
            return;
        }
        Vector diffs(base_vals.size(), 0.0);
        for (std::size_t i = 0; i < base_vals.size(); ++i) {
            diffs[i] = base_vals[i] - target_vals[i];
        }
        double err_mean = 0.0;
        for (double v : diffs) {
            err_mean += v;
        }
        err_mean /= static_cast<double>(diffs.size());
        double var = 0.0;
        for (double v : diffs) {
            double diff = v - err_mean;
            var += diff * diff;
        }
        var /= static_cast<double>(diffs.size());
        double err_std = std::sqrt(std::max(0.0, var));
        for (int layer = 0; layer < n_layers_; ++layer) {
            gate_bias_[layer][0] -= lr * err_mean;
            gate_bias_[layer][1] -= lr * err_std;
            ff_gate_[layer][0] -= lr * err_mean;
            ff_gate_[layer][1] -= lr * err_std;
        }
    }

    py::dict export_state() const {
        py::dict state;
        state["d_model"] = d_model_;
        state["n_layers"] = n_layers_;
        state["n_heads"] = n_heads_;
        auto matrix_to_list = [](const Matrix &m) {
            py::list rows;
            for (const auto &row : m) {
                py::list lst;
                for (double v : row) {
                    lst.append(v);
                }
                rows.append(lst);
            }
            return rows;
        };
        auto vector_to_list = [](const Vector &v) {
            py::list lst;
            for (double val : v) {
                lst.append(val);
            }
            return lst;
        };
        py::list Wq_list, Wk_list, Wv_list, Wo_list, Wff1_list, Wff2_list, bff1_list, bff2_list, ln1_gamma_list,
            ln1_beta_list, ln2_gamma_list, ln2_beta_list, gate_bias_list, ff_gate_list;
        for (int layer = 0; layer < n_layers_; ++layer) {
            Wq_list.append(matrix_to_list(Wq_[layer]));
            Wk_list.append(matrix_to_list(Wk_[layer]));
            Wv_list.append(matrix_to_list(Wv_[layer]));
            Wo_list.append(matrix_to_list(Wo_[layer]));
            Wff1_list.append(matrix_to_list(Wff1_[layer]));
            Wff2_list.append(matrix_to_list(Wff2_[layer]));
            bff1_list.append(vector_to_list(bff1_[layer]));
            bff2_list.append(vector_to_list(bff2_[layer]));
            ln1_gamma_list.append(vector_to_list(ln1_gamma_[layer]));
            ln1_beta_list.append(vector_to_list(ln1_beta_[layer]));
            ln2_gamma_list.append(vector_to_list(ln2_gamma_[layer]));
            ln2_beta_list.append(vector_to_list(ln2_beta_[layer]));
            gate_bias_list.append(vector_to_list(gate_bias_[layer]));
            ff_gate_list.append(vector_to_list(ff_gate_[layer]));
        }
        state["Wq"] = Wq_list;
        state["Wk"] = Wk_list;
        state["Wv"] = Wv_list;
        state["Wo"] = Wo_list;
        state["Wff1"] = Wff1_list;
        state["bff1"] = bff1_list;
        state["Wff2"] = Wff2_list;
        state["bff2"] = bff2_list;
        state["ln1_gamma"] = ln1_gamma_list;
        state["ln1_beta"] = ln1_beta_list;
        state["ln2_gamma"] = ln2_gamma_list;
        state["ln2_beta"] = ln2_beta_list;
        state["gate_bias"] = gate_bias_list;
        state["ff_gate"] = ff_gate_list;
        return state;
    }

    void load_state(const py::dict &state) {
        auto get_int = [&state](const char *key, int fallback) {
            if (state.contains(key)) {
                return py::cast<int>(state[key]);
            }
            return fallback;
        };
        d_model_ = get_int("d_model", d_model_);
        n_layers_ = get_int("n_layers", n_layers_);
        n_heads_ = get_int("n_heads", n_heads_);
        head_dim_ = d_model_ / std::max(1, n_heads_);
        auto to_matrix_list = [](const py::handle &obj) {
            std::vector<Matrix> mats;
            for (const auto &mat_obj : obj) {
                Matrix mat;
                for (const auto &row_obj : py::reinterpret_borrow<py::iterable>(mat_obj)) {
                    Vector row;
                    for (const auto &val : py::reinterpret_borrow<py::iterable>(row_obj)) {
                        row.push_back(py::cast<double>(val));
                    }
                    mat.push_back(std::move(row));
                }
                mats.push_back(std::move(mat));
            }
            return mats;
        };
        auto to_vector_list = [](const py::handle &obj) {
            std::vector<Vector> vecs;
            for (const auto &vec_obj : obj) {
                Vector vec;
                for (const auto &val : py::reinterpret_borrow<py::iterable>(vec_obj)) {
                    vec.push_back(py::cast<double>(val));
                }
                vecs.push_back(std::move(vec));
            }
            return vecs;
        };
        Wq_ = to_matrix_list(state["Wq"]);
        Wk_ = to_matrix_list(state["Wk"]);
        Wv_ = to_matrix_list(state["Wv"]);
        Wo_ = to_matrix_list(state["Wo"]);
        Wff1_ = to_matrix_list(state["Wff1"]);
        Wff2_ = to_matrix_list(state["Wff2"]);
        bff1_ = to_vector_list(state["bff1"]);
        bff2_ = to_vector_list(state["bff2"]);
        ln1_gamma_ = to_vector_list(state["ln1_gamma"]);
        ln1_beta_ = to_vector_list(state["ln1_beta"]);
        ln2_gamma_ = to_vector_list(state["ln2_gamma"]);
        ln2_beta_ = to_vector_list(state["ln2_beta"]);
        gate_bias_ = to_vector_list(state["gate_bias"]);
        ff_gate_ = to_vector_list(state["ff_gate"]);
    }

    py::list device_inventory() const {
        py::list result;
        for (const auto &dev : available_devices_) {
            result.append(dev);
        }
        return result;
    }

    void set_device(const std::string &device) {
        const std::string trimmed = trim_copy(device);
        if (trimmed.empty()) {
            device_ = select_default_device(available_devices_);
            return;
        }
        const std::string lowered = to_lower_copy(trimmed);
        if (lowered == "auto" || lowered == "default") {
            device_ = select_default_device(available_devices_);
            return;
        }
        if (lowered == "gpu" || lowered == "accelerator" || lowered == "best") {
            device_ = first_non_cpu_device(available_devices_);
            return;
        }
        const std::string matched = match_device_token(available_devices_, lowered);
        if (!matched.empty()) {
            device_ = matched;
            return;
        }
        throw std::invalid_argument("Unsupported device: " + device);
    }

    py::list last_attn() const { return last_attn_; }

    py::array_t<double> last_gate_mask() const { return last_gate_mask_; }

    std::string device() const { return device_; }

    std::string backend() const { return backend_; }

   private:
    int d_model_;
    int n_layers_;
    int n_heads_;
    double ff_multiplier_;
    int head_dim_;
    int ff_dim_;
    std::string device_;
    std::string backend_;
    std::vector<std::string> available_devices_;
    std::vector<Matrix> Wq_;
    std::vector<Matrix> Wk_;
    std::vector<Matrix> Wv_;
    std::vector<Matrix> Wo_;
    std::vector<Matrix> Wff1_;
    std::vector<Vector> bff1_;
    std::vector<Matrix> Wff2_;
    std::vector<Vector> bff2_;
    std::vector<Vector> ln1_gamma_;
    std::vector<Vector> ln1_beta_;
    std::vector<Vector> ln2_gamma_;
    std::vector<Vector> ln2_beta_;
    std::vector<Vector> gate_bias_;
    std::vector<Vector> ff_gate_;
    py::list last_attn_;
    py::array_t<double> last_gate_mask_;
};

PYBIND11_MODULE(_spiral_transformer_cpp, m) {
    const std::vector<std::string> devices = compute_available_devices();
    py::tuple device_tuple(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
        device_tuple[i] = py::str(devices[i]);
    }
    const std::string default_device = select_default_device(devices);
    m.attr("BACKEND_KIND") = kBackendKind;
    m.attr("DEFAULT_DEVICE") = default_device;
    m.attr("AVAILABLE_DEVICES") = device_tuple;

    py::class_<CppTransformerAdapter>(m, "CppTransformerAdapter")
        .def(py::init<int, int, int, double, int>(),
             py::arg("d_model") = 128,
             py::arg("n_layers") = 4,
             py::arg("n_heads") = 4,
             py::arg("ff_multiplier") = 4.0,
             py::arg("seed") = 2025)
        .def("forward", &CppTransformerAdapter::forward, py::arg("X"), py::arg("gate_pos"), py::arg("gate_mask") = py::none())
        .def("tune_from_boundary", &CppTransformerAdapter::tune_from_boundary, py::arg("base_gate"), py::arg("targets"), py::arg("lr") = 1e-3)
        .def("export_state", &CppTransformerAdapter::export_state)
        .def("load_state", &CppTransformerAdapter::load_state)
        .def_property_readonly("device", &CppTransformerAdapter::device)
        .def_property_readonly("backend", &CppTransformerAdapter::backend)
        .def_property_readonly("device_inventory", &CppTransformerAdapter::device_inventory)
        .def("set_device", &CppTransformerAdapter::set_device, py::arg("device"))
        .def_property_readonly("last_attn", &CppTransformerAdapter::last_attn)
        .def_property_readonly("last_gate_mask", &CppTransformerAdapter::last_gate_mask);
}

