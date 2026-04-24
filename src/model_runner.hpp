#pragma once

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
};

// Owns serialization of inference (one completion at a time) and delegates to the llama subprocess backend.
class ModelRunner {
public:
    GenerateResult generate(std::string_view prompt, const GenerateParams& params);

private:
    std::mutex mutex_;
};
