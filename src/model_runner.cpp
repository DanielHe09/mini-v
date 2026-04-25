#include "model_runner.hpp"

#include "llama_spawn.hpp"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::string basename_path(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

} // namespace

ModelRunner::ModelRunner() : worker_(&ModelRunner::worker_loop, this) {}

ModelRunner::~ModelRunner() {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

ModelRunner::RequestPtr ModelRunner::submit(std::string prompt, GenerateParams params,
                                            const std::chrono::steady_clock::time_point request_started_at) {
    RequestPtr request;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            throw std::runtime_error("model runner is stopping");
        }
        request = std::make_shared<InferenceRequest>(next_request_id_++, std::move(prompt), std::move(params),
                                                     request_started_at);
        pending_.push(request);
    }
    cv_.notify_one();
    return request;
}

void ModelRunner::worker_loop() {
    for (;;) {
        RequestPtr request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stopping_ || !pending_.empty(); });
            if (stopping_ && pending_.empty()) {
                return;
            }
            request = std::move(pending_.front());
            pending_.pop();
        }

        try {
            request->result = run_request(*request);
            request->result_promise_.set_value(request->result);
        } catch (...) {
            request->result_promise_.set_exception(std::current_exception());
        }
    }
}

GenerateResult ModelRunner::run_request(const InferenceRequest& request) {
    const int n_predict = request.params.max_tokens.value_or(256);

    const char* model_env = std::getenv("LLAMA_MODEL");
    const std::string model_label =
        (model_env && *model_env) ? basename_path(std::string(model_env)) : "unset";

    const auto t_model_start = std::chrono::steady_clock::now();
    const auto since_request_at_model_start =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_model_start - request.request_started_at);
    LlamaRunResult gen = run_llama_completion(request.prompt, n_predict, request.params.temperature);
    const auto t_model_end = std::chrono::steady_clock::now();
    const auto model_duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_model_end - t_model_start);

    GenerateResult out;
    out.model_label = model_label;
    out.since_request_at_model_start = since_request_at_model_start;
    out.model_duration = model_duration;
    out.model_timings_valid = true;

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
