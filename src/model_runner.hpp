#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

struct InferenceRequest {
    using Id = std::uint64_t;

    Id id = 0;
    std::string prompt;
    GenerateParams params;
    GenerateResult result;

private:
    friend class ModelRunner;

    std::promise<GenerateResult> result_promise_;

public:
    std::shared_future<GenerateResult> result_future;
    std::chrono::steady_clock::time_point request_started_at;

    InferenceRequest(Id id_, std::string prompt_, GenerateParams params_,
                     std::chrono::steady_clock::time_point request_started_at_)
        : id(id_),
          prompt(std::move(prompt_)),
          params(std::move(params_)),
          result_future(result_promise_.get_future().share()),
          request_started_at(request_started_at_) {}

    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
};

// Owns request scheduling for inference and delegates execution to the llama subprocess backend.
class ModelRunner {
public:
    using RequestPtr = std::shared_ptr<InferenceRequest>;

    ModelRunner();
    ~ModelRunner();

    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    RequestPtr submit(std::string prompt, GenerateParams params,
                      std::chrono::steady_clock::time_point request_started_at);

private:
    void worker_loop();
    GenerateResult run_request(const InferenceRequest& request);
    std::vector<RequestPtr> collect_batch(std::unique_lock<std::mutex>& lock);

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<RequestPtr> pending_;
    bool stopping_ = false;
    InferenceRequest::Id next_request_id_ = 1;
    std::thread worker_;

    static constexpr std::chrono::milliseconds kBatchWindow{50};
};
