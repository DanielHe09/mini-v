#pragma once

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

struct GenerateParams {
    std::optional<int> max_tokens;
    std::optional<double> temperature;
};

struct GenerateResult {
    bool ok = false;
    std::string message;
    std::string output;
    /// When !ok: true means client should treat as 503 (missing/wrong LLAMA_*), false as 502.
    bool misconfigured = false;
    std::string model_label;

    /// Time from `request_started_at` (passed to generate) until subprocess inference begins (after lock).
    std::chrono::nanoseconds since_request_at_model_start{0};
    /// Subprocess inference only (`run_llama_completion`).
    std::chrono::nanoseconds model_duration{0};
    bool model_timings_valid = false;
};

// Owns serialization of inference (one completion at a time) and delegates to the llama subprocess backend.
class ModelRunner {
public:
    GenerateResult generate(std::string_view prompt, const GenerateParams& params,
                            std::chrono::steady_clock::time_point request_started_at);

private:
    std::mutex mutex_;
};
