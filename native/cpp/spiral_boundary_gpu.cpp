#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
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

constexpr const char *kBackendKind = "cpp-accelerator";

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

std::string resolve_device_request(const std::vector<std::string> &devices, const std::string &request, bool strict) {
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
    const char *keys[] = {"SPIRAL_BOUNDARY_DEVICE", "SPIRAL_DEVICE", "SPIRAL_DEFAULT_DEVICE"};
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

class CppBoundaryStudent {
   public:
    CppBoundaryStudent() : available_devices_(compute_available_devices()) {
        device_ = select_default_device(available_devices_);
    }

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
            std::string candidate = trim_copy(py::str(cfg["device_preference"]));
            if (!candidate.empty()) {
                const std::string resolved = resolve_device_request(available_devices_, candidate, false);
                preferred_device_ = resolved.empty() ? candidate : resolved;
            }
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
        select_device();
        auto start = std::chrono::high_resolution_clock::now();

        std::size_t dataset_tokens = 0;
        for (std::size_t i = 0; i < texts.size(); ++i) {
            py::str text_obj = py::str(texts[i]);
            auto chars = collect_characters(text_obj);
            dataset_tokens += chars.size();
            py::sequence segment_seq = py::reinterpret_borrow<py::sequence>(segments[i]);
            auto boundaries = collect_boundaries(segment_seq, chars.size());
            update_counts(chars, boundaries);
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
        summary["accelerated"] = false;
        summary["accelerator_available"] = has_any_accelerator();
        summary["available_devices"] = available_devices();
        summary["selected_device"] = device_;
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
        std::string loaded_device = py::str(state.get("device", device_));
        const std::string resolved = resolve_device_request(available_devices_, loaded_device, false);
        if (!resolved.empty()) {
            device_ = resolved;
        } else if (is_device_available(loaded_device)) {
            device_ = loaded_device;
        }
        compute_bias();
    }

    py::tuple available_devices() const {
        return py::cast(available_devices_);
    }

    std::string preferred_device() const {
        if (!preferred_device_.empty()) {
            const std::string resolved = resolve_device_request(available_devices_, preferred_device_, false);
            if (!resolved.empty()) {
                return resolved;
            }
        }
        return select_default_device(available_devices_);
    }

    bool to_device(const std::string &device) {
        std::string resolved;
        try {
            resolved = resolve_device_request(available_devices_, device, false);
        } catch (const std::exception &) {
            resolved.clear();
        }
        if (resolved.empty()) {
            return false;
        }
        device_ = resolved;
        return true;
    }

   private:
    void reset_state() {
        pair_stats_.clear();
        bias_ = 0.0;
        total_pairs_ = 0.0;
        boundary_pairs_ = 0.0;
        mean_boundary_rate_ = 0.0;
    }

    void select_device() {
        if (!preferred_device_.empty()) {
            const std::string resolved = resolve_device_request(available_devices_, preferred_device_, false);
            if (!resolved.empty()) {
                device_ = resolved;
                return;
            }
        }
        const std::string env_pref = environment_device_preference(available_devices_);
        if (!env_pref.empty()) {
            device_ = env_pref;
            return;
        }
        if (!is_device_available(device_)) {
            device_ = select_default_device(available_devices_);
        }
    }

    bool has_any_accelerator() const { return available_devices_.size() > 1; }

    bool is_device_available(const std::string &device) const {
        return std::find(available_devices_.begin(), available_devices_.end(), device) != available_devices_.end();
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

    void update_counts(const std::vector<std::string> &chars, const std::vector<std::size_t> &boundaries) {
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
        return std::clamp(base, 0.01, 0.95);
    }

    double pair_probability(const std::string &prev, const std::string &next) const {
        auto it = pair_stats_.find(make_pair_key(prev, next));
        double probability;
        if (it != pair_stats_.end()) {
            probability = (it->second.boundary + smoothing_) / (it->second.total + 2.0 * smoothing_);
        } else {
            probability = fallback_probability(prev, next);
        }
        return std::clamp(probability, 1e-4, 1.0 - 1e-4);
    }

    void compute_bias() {
        if (total_pairs_ <= 0.0) {
            mean_boundary_rate_ = fallback_bias_;
            bias_ = fallback_bias_;
            return;
        }
        mean_boundary_rate_ = (boundary_pairs_ + smoothing_) / (total_pairs_ + 2.0 * smoothing_);
        bias_ = mean_boundary_rate_;
        threshold_ = std::clamp((mean_boundary_rate_ + threshold_) / 2.0, 0.2, 0.8);
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
    std::string backend_kind_ = kBackendKind;
    std::vector<std::string> available_devices_;
    py::object phase_ref_;
    py::object encoder_ref_;
};

}  // namespace

PYBIND11_MODULE(spiral_boundary_gpu, m) {
    m.doc() = "SpiralReality boundary detector (accelerator-capable stub) implemented in C++";
    py::class_<CppBoundaryStudent>(m, "CppBoundaryStudent")
        .def(py::init<>())
        .def("configure", &CppBoundaryStudent::configure, py::arg("cfg"))
        .def("attach_phase", &CppBoundaryStudent::attach_phase, py::arg("phase"))
        .def("attach_encoder", &CppBoundaryStudent::attach_encoder, py::arg("encoder"))
        .def("train", &CppBoundaryStudent::train, py::arg("texts"), py::arg("segments"), py::arg("cfg"))
        .def("boundary_probs", &CppBoundaryStudent::boundary_probs, py::arg("text"))
        .def("decode", &CppBoundaryStudent::decode, py::arg("text"))
        .def("export_state", &CppBoundaryStudent::export_state)
        .def("load_state", &CppBoundaryStudent::load_state, py::arg("state"))
        .def("available_devices", &CppBoundaryStudent::available_devices)
        .def("preferred_device", &CppBoundaryStudent::preferred_device)
        .def("to_device", &CppBoundaryStudent::to_device, py::arg("device"));

    auto devices = compute_available_devices();
    const std::string default_device = select_default_device(devices);
    py::tuple devices_tuple = py::cast(devices);
    m.attr("BACKEND_KIND") = kBackendKind;
    m.attr("DEFAULT_DEVICE") = default_device;
    m.attr("AVAILABLE_DEVICES") = devices_tuple;
}
