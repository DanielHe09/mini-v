#pragma once

#include "llama_spawn.hpp"

#include <optional>
#include <string>
#include <string_view>

// Calls a long-running llama-server instance instead of spawning a llama CLI
// process. Configure with LLAMA_SERVER_URL, defaulting to http://127.0.0.1:8080.
LlamaRunResult run_llama_server_completion(std::string_view prompt, int max_tokens,
                                           std::optional<double> temperature);
