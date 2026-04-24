#include "model_runner.hpp"

#include "llama_spawn.hpp"

#include <cstdlib>
#include <string>

namespace {

std::string basename_path(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

} // namespace

GenerateResult ModelRunner::generate(const std::string_view prompt, const GenerateParams& params) {
    const int n_predict = params.max_tokens.value_or(256);

    const char* model_env = std::getenv("LLAMA_MODEL");
    const std::string model_label =
        (model_env && *model_env) ? basename_path(std::string(model_env)) : "unset";

    LlamaRunResult gen;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        gen = run_llama_completion(prompt, n_predict, params.temperature);
    }

    GenerateResult out;
    out.model_label = model_label;

    if (!gen.ok) {
        out.ok = false;
        out.message = std::move(gen.message);
        out.misconfigured = (out.message.compare(0, 11, "LLAMA_MODEL") == 0) ||
                            (out.message.compare(0, 9, "LLAMA_CLI") == 0);
        return out;
    }

    out.ok = true;
    out.output = std::move(gen.output);
    return out;
}
