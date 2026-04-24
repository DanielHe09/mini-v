#pragma once

#include <optional>
#include <string>
#include <string_view>

struct LlamaRunResult {
    bool ok = false;
    std::string message;
    std::string output;
};

// Runs `llama-completion` (or LLAMA_CLI) as a subprocess: one prompt in, completion on stdout.
// Newer llama.cpp: `-no-cnv` is not supported on `llama-cli` — use `llama-completion` (default if LLAMA_CLI unset).
// Configure with env: LLAMA_MODEL (required, path to .gguf), LLAMA_CLI (optional, default llama-completion).
LlamaRunResult run_llama_completion(std::string_view prompt, int max_tokens,
                                     std::optional<double> temperature);
