#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct PairStats {
    double boundary = 0.0;
    double total = 0.0;
};

bool is_ascii_space(const std::string &ch) {
    if (ch.empty()) {
        return false;
    }
    unsigned char c = static_cast<unsigned char>(ch[0]);
    return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\f' || c == '\v';
}

bool is_ascii_punct(const std::string &ch) {
    if (ch.empty()) {
        return false;
    }
    unsigned char c = static_cast<unsigned char>(ch[0]);
    static const std::string punct = "!?,.;:()[]{}<>\"'`~+-*/\\|";
    return punct.find(static_cast<char>(c)) != std::string::npos;
}

std::string make_pair_key(const std::string &prev, const std::string &next) {
    return prev + std::string("\u25B6") + next;
}

#ifdef SPIRAL_HAS_CUDA
constexpr bool kCompiledWithCuda = true;
#else
constexpr bool kCompiledWithCuda = false;
#endif

class GpuBoundaryStudent {
   public:
    GpuBoundaryStudent() = default;

    void configure(const py::dict &cfg) {
        if (cfg.contains("compiled_threshold")) {
            threshold_ = py::float_(cfg["compiled_threshold"]);
        } else if (cfg.contains("boundary_threshold")) {
            threshold_ = py::float_(cfg["boundary_threshold"]);
        }
        if (cfg.contains("compiled_smoothing")) {
            smoothing_ = std::max(1e-6, py::float_(cfg["compiled_smoothing"]));
        }
        if (cfg.contains("fallback_bias")) {
            fallback_bias_ = py::float_(cfg["fallback_bias"]);
        }
        if (cfg.contains("device_preference")) {
            preferred_device_ = py::str(cfg["device_preference"]);
        }
    }

    void attach_phase(const py::object &phase) { phase_ref_ = phase; }

    void attach_encoder(const py::object &encoder) { encoder_ref_ = encoder; }

    py::dict train(const py::sequence &texts, const py::sequence &segments, const py::dict &cfg) {
        configure(cfg);
        if (texts.size() != segments.size()) {
            throw std::runtime_error("texts and segments must be the same length");
        }

        reset_state();
        auto start = std::chrono::high_resolution_clock::now();

        std::size_t dataset_tokens = 0;
        for (std::size_t i = 0; i < texts.size(); ++i) {
            py::str text_obj = py::str(texts[i]);
            auto chars = collect_characters(text_obj);
            dataset_tokens += chars.size();
            py::sequence segment_seq = py::reinterpret_borrow<py::sequence>(segments[i]);
            auto boundaries = collect_boundaries(segment_seq, chars.size());
            if (should_use_accelerator()) {
                update_counts_accelerated(chars, boundaries);
            } else {
                update_counts_cpu(chars, boundaries);
            }
        }

        compute_bias();
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();

        py::dict summary;
        summary["backend"] = backend_kind_ + std::string(":") + device_;
        summary["examples"] = static_cast<double>(texts.size());
        summary["tokens"] = static_cast<double>(dataset_tokens);
        summary["pairs_tracked"] = static_cast<double>(pair_stats_.size());
        summary["mean_boundary_rate"] = mean_boundary_rate_;
        summary["threshold"] = threshold_;
        summary["smoothing"] = smoothing_;
        summary["device"] = device_;
        summary["train_seconds"] = seconds;
        summary["training_pairs"] = total_pairs_;
        summary["boundary_pairs"] = boundary_pairs_;
        summary["accelerated"] = should_use_accelerator();
        summary["accelerator_available"] = kCompiledWithCuda;
        return summary;
    }

    py::list boundary_probs(const py::str &text) const {
        auto chars = collect_characters(text);
        py::list result;
        if (chars.size() < 2) {
            return result;
        }
        for (std::size_t i = 0; i + 1 < chars.size(); ++i) {
            double prob = pair_probability(chars[i], chars[i + 1]);
            result.append(prob);
        }
        return result;
    }

    py::list decode(const py::str &text) const {
        auto chars = collect_characters(text);
        py::list probs = boundary_probs(text);
        py::list tokens;
        if (chars.empty()) {
            return tokens;
        }
        std::string buffer;
        buffer.reserve(chars.size());
        for (std::size_t i = 0; i < chars.size(); ++i) {
            buffer += chars[i];
            bool boundary = false;
            if (i < static_cast<std::size_t>(py::len(probs))) {
                double prob = py::float_(probs[i]);
                boundary = prob >= threshold_;
            }
            if (boundary) {
                tokens.append(buffer);
                buffer.clear();
            }
        }
        if (!buffer.empty()) {
            tokens.append(buffer);
        }
        return tokens;
    }

    py::dict export_state() const {
        py::dict state;
        py::dict stats_dict;
        for (const auto &entry : pair_stats_) {
            py::dict values;
            values["boundary"] = entry.second.boundary;
            values["total"] = entry.second.total;
            stats_dict[py::str(entry.first)] = values;
        }
        state["pair_stats"] = stats_dict;
        state["bias"] = bias_;
        state["threshold"] = threshold_;
        state["smoothing"] = smoothing_;
        state["device"] = device_;
        state["backend"] = backend_kind_;
        return state;
    }

    void load_state(const py::dict &state) {
        reset_state();
        py::dict stats_dict = py::dict(state.get("pair_stats", py::dict()));
        for (auto item : stats_dict) {
            std::string key = py::str(item.first);
            py::dict values = py::dict(item.second);
            PairStats stats;
            stats.boundary = py::float_(values.get("boundary", 0.0));
            stats.total = py::float_(values.get("total", 0.0));
            pair_stats_.emplace(std::move(key), stats);
        }
        bias_ = py::float_(state.get("bias", bias_));
        threshold_ = py::float_(state.get("threshold", threshold_));
        smoothing_ = py::float_(state.get("smoothing", smoothing_));
        device_ = py::str(state.get("device", device_));
        compute_bias();
    }

    py::tuple available_devices() const {
        if (kCompiledWithCuda) {
            return py::make_tuple("cpu", "cuda:0");
        }
        return py::make_tuple("cpu");
    }

    std::string preferred_device() const {
        if (!preferred_device_.empty()) {
            return preferred_device_;
        }
        if (kCompiledWithCuda) {
            return "cuda:0";
        }
        return "cpu";
    }

    void to_device(const std::string &device) {
        if (device == device_) {
            return;
        }
        if (device == "cpu") {
            device_ = "cpu";
            return;
        }
        if (device.rfind("cuda", 0) == 0) {
            if (!kCompiledWithCuda) {
                throw std::runtime_error("GPU backend was built without CUDA support");
            }
            device_ = device;
            return;
        }
        throw std::runtime_error("Unknown device: " + device);
    }

   private:
    void reset_state() {
        pair_stats_.clear();
        total_pairs_ = 0.0;
        boundary_pairs_ = 0.0;
        mean_boundary_rate_ = 0.0;
        if (device_.empty()) {
            device_ = "cpu";
        }
    }

    bool should_use_accelerator() const {
        return kCompiledWithCuda && device_.rfind("cuda", 0) == 0;
    }

    std::vector<std::string> collect_characters(const py::str &text) const {
        py::list char_list(text);
        std::vector<std::string> chars;
        chars.reserve(py::len(char_list));
        for (auto item : char_list) {
            chars.emplace_back(py::str(item));
        }
        return chars;
    }

    std::vector<std::size_t> collect_boundaries(const py::sequence &segments, std::size_t total_chars) const {
        std::vector<std::size_t> boundaries;
        std::size_t cursor = 0;
        std::size_t count = segments.size();
        for (std::size_t i = 0; i < count; ++i) {
            py::str seg = py::str(segments[i]);
            py::list seg_chars(seg);
            cursor += seg_chars.size();
            if (cursor > total_chars) {
                cursor = total_chars;
            }
            if (i + 1 < count && cursor > 0) {
                boundaries.push_back(cursor - 1);
            }
        }
        return boundaries;
    }

    void update_counts_cpu(const std::vector<std::string> &chars, const std::vector<std::size_t> &boundaries) {
        std::unordered_set<std::size_t> boundary_set(boundaries.begin(), boundaries.end());
        if (chars.size() < 2) {
            return;
        }
        for (std::size_t i = 0; i + 1 < chars.size(); ++i) {
            bool is_boundary = boundary_set.find(i) != boundary_set.end();
            PairStats &stats = pair_stats_[make_pair_key(chars[i], chars[i + 1])];
            stats.total += 1.0;
            total_pairs_ += 1.0;
            if (is_boundary) {
                stats.boundary += 1.0;
                boundary_pairs_ += 1.0;
            }
        }
    }

    void update_counts_accelerated(const std::vector<std::string> &chars, const std::vector<std::size_t> &boundaries) {
        // Until full CUDA kernels are wired in we mirror the CPU logic.
        // The structure makes it trivial to swap in GPU reductions.
        update_counts_cpu(chars, boundaries);
    }

    double fallback_probability(const std::string &prev, const std::string &next) const {
        double base = fallback_bias_;
        if (is_ascii_space(prev)) {
            base += 0.35;
        }
        if (is_ascii_punct(prev)) {
            base += 0.25;
        }
        if (is_ascii_space(next)) {
            base += 0.1;
        }
        if (is_ascii_punct(next)) {
            base += 0.15;
        }
        if (prev.size() > 1 || next.size() > 1) {
            base += 0.05;
        }
        return std::min(0.95, std::max(0.01, base));
    }

    double pair_probability(const std::string &prev, const std::string &next) const {
        auto it = pair_stats_.find(make_pair_key(prev, next));
        double probability;
        if (it != pair_stats_.end()) {
            probability = (it->second.boundary + smoothing_) / (it->second.total + 2.0 * smoothing_);
        } else {
            probability = fallback_probability(prev, next);
        }
        if (probability < 1e-4) {
            probability = 1e-4;
        } else if (probability > 1.0 - 1e-4) {
            probability = 1.0 - 1e-4;
        }
        return probability;
    }

    void compute_bias() {
        if (total_pairs_ <= 0.0) {
            mean_boundary_rate_ = fallback_bias_;
            bias_ = fallback_bias_;
            return;
        }
        mean_boundary_rate_ = (boundary_pairs_ + smoothing_) / (total_pairs_ + 2.0 * smoothing_);
        bias_ = mean_boundary_rate_;
        threshold_ = std::max(0.2, std::min(0.8, (mean_boundary_rate_ + threshold_) / 2.0));
    }

    std::unordered_map<std::string, PairStats> pair_stats_;
    double bias_ = 0.0;
    double threshold_ = 0.55;
    double smoothing_ = 0.5;
    double fallback_bias_ = 0.12;
    double total_pairs_ = 0.0;
    double boundary_pairs_ = 0.0;
    double mean_boundary_rate_ = 0.0;
    std::string device_ = "cpu";
    std::string preferred_device_;
    std::string backend_kind_ = "cpp-gpu";
    py::object phase_ref_;
    py::object encoder_ref_;
};

}  // namespace

PYBIND11_MODULE(spiral_boundary_gpu, m) {
    m.doc() = "SpiralReality boundary detector with GPU-aware backend";
    py::class_<GpuBoundaryStudent>(m, "GpuBoundaryStudent")
        .def(py::init<>())
        .def("configure", &GpuBoundaryStudent::configure, py::arg("cfg"))
        .def("attach_phase", &GpuBoundaryStudent::attach_phase, py::arg("phase"))
        .def("attach_encoder", &GpuBoundaryStudent::attach_encoder, py::arg("encoder"))
        .def("train", &GpuBoundaryStudent::train, py::arg("texts"), py::arg("segments"), py::arg("cfg"))
        .def("boundary_probs", &GpuBoundaryStudent::boundary_probs, py::arg("text"))
        .def("decode", &GpuBoundaryStudent::decode, py::arg("text"))
        .def("export_state", &GpuBoundaryStudent::export_state)
        .def("load_state", &GpuBoundaryStudent::load_state, py::arg("state"))
        .def("available_devices", &GpuBoundaryStudent::available_devices)
        .def("preferred_device", &GpuBoundaryStudent::preferred_device)
        .def("to_device", &GpuBoundaryStudent::to_device, py::arg("device"));

    m.attr("BACKEND_KIND") = "cpp-gpu";
    m.attr("DEFAULT_DEVICE") = kCompiledWithCuda ? "cuda:0" : "cpu";
    if (kCompiledWithCuda) {
        m.attr("AVAILABLE_DEVICES") = py::make_tuple("cpu", "cuda:0");
    } else {
        m.attr("AVAILABLE_DEVICES") = py::make_tuple("cpu");
    }
    m.attr("CUDA_ENABLED") = kCompiledWithCuda;
}
