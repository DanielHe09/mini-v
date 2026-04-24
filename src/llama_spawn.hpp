#pragma once

#include <optional>
#include <string>
#include <string_view>

struct LlamaRunResult {
    bool ok = false;
    std::string message;
    std::string output;
};

// Runs `llama-cli` (or LLAMA_CLI) as a subprocess: one prompt in, completion on stdout.
// Configure with env: LLAMA_MODEL (required, path to .gguf), LLAMA_CLI (optional, default llama-cli).
LlamaRunResult run_llama_completion(std::string_view prompt, int max_tokens,
                                     std::optional<double> temperature);
