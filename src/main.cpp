// Crow = small C++ HTTP framework (think Starlette/Flask-level, not batteries-included like FastAPI).
// nlohmann::json = JSON as nested maps/arrays (think Python dict + json.loads / dumps).

#include "model_runner.hpp"

#include <crow.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>

// Roughly: int(os.environ.get("PORT", "18080")) with validation.
// getenv returns char* or nullptr; strtoul parses unsigned integer from C string.
static uint16_t port_from_env() {
    const char* p = std::getenv("PORT");
    if (!p || !*p) {
        return 18080;
    }
    char* end = nullptr;
    const unsigned long v = std::strtoul(p, &end, 10);
    if (end == p || *end != '\0' || v == 0 || v > 65535) {
        std::cerr << "PORT must be 1-65535, ignoring invalid value.\n";
        return 18080;
    }
    return static_cast<uint16_t>(v);
}

static double ns_to_ms(const std::chrono::nanoseconds ns) {
    return std::chrono::duration<double, std::milli>(ns).count();
}

static void log_generate_latency(const std::chrono::steady_clock::time_point& t_request,
                                 const std::chrono::steady_clock::time_point& t_response, const int status,
                                 const GenerateResult* const gen_timing) {
    const auto total_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_response - t_request);
    std::cerr << std::fixed << std::setprecision(3) << "[mini-v] POST /generate status=" << status
              << " total_ms=" << ns_to_ms(total_ns);
    if (gen_timing != nullptr && gen_timing->model_timings_valid) {
        const auto model_end_ns = gen_timing->since_request_at_model_start + gen_timing->model_duration;
        const auto post_model_ns = total_ns - model_end_ns;
        std::cerr << " model_start_ms=" << ns_to_ms(gen_timing->since_request_at_model_start)
                  << " model_ms=" << ns_to_ms(gen_timing->model_duration)
                  << " model_end_ms=" << ns_to_ms(model_end_ns)
                  << " post_model_ms=" << ns_to_ms(post_model_ns);
    } else {
        std::cerr << " model_start_ms=- model_ms=- model_end_ms=- post_model_ms=-";
    }
    std::cerr << '\n';
}

int main() {
    // One app object; routes register on it (like `app = FastAPI()` then decorators).
    crow::SimpleApp app;

    ModelRunner model_runner;

    // CROW_ROUTE is a macro that expands to "register this path on app".
    // Mental model vs FastAPI:
    //   @app.post("/generate")
    //   async def generate(req: Request): ...
    //
    // `.methods(crow::HTTPMethod::Post)` = only POST (like methods=["POST"]).
    //
    // `([](const crow::request& req) { ... })` is a lambda: anonymous function object
    // passed as the handler. [=] would capture locals by copy; here we only use `req`.
    // C++ lambdas used as callbacks are extremely common in C++ web code.
    CROW_ROUTE(app, "/generate")
        .methods(crow::HTTPMethod::Post)([&model_runner](const crow::request& req) {
            const std::chrono::steady_clock::time_point t_request = std::chrono::steady_clock::now();

            auto json_error = [](int status, const char* message) {
                crow::response res(status, nlohmann::json{{"error", message}}.dump());
                res.set_header("Content-Type", "application/json");
                return res;
            };

            auto json_error_string = [](int status, const std::string& message) {
                crow::response res(status, nlohmann::json{{"error", message}}.dump());
                res.set_header("Content-Type", "application/json");
                return res;
            };

            auto respond = [&](crow::response&& res, const int status, const GenerateResult* gen_timing) {
                const std::chrono::steady_clock::time_point t_response = std::chrono::steady_clock::now();
                log_generate_latency(t_request, t_response, status, gen_timing);
                return std::move(res);
            };

            // Parse JSON body (expect a single object, like a FastAPI JSON body model).
            nlohmann::json body = nlohmann::json::object();
            if (!req.body.empty()) {
                try {
                    body = nlohmann::json::parse(req.body);
                } catch (const nlohmann::json::parse_error&) {
                    return respond(json_error(400, "invalid json"), 400, nullptr);
                }
            }

            if (!body.is_object()) {
                return respond(json_error(400, "body must be a JSON object"), 400, nullptr);
            }

            //extract some basic parameters like prompt, max_tokens, temperature. Last 2 are optional
            if (!body.contains("prompt") || !body["prompt"].is_string()) {
                return respond(json_error(400, R"(field "prompt" (string) is required)"), 400, nullptr);
            }
            const std::string prompt = body["prompt"].get<std::string>();

            std::optional<int> max_tokens;
            if (body.contains("max_tokens")) {
                const auto& v = body["max_tokens"];
                if (!v.is_number()) {
                    return respond(json_error(400, R"(field "max_tokens" must be a number)"), 400, nullptr);
                }
                const double x = v.get<double>();
                if (!std::isfinite(x) || x < 0.0 || x > 1'000'000.0 || x != std::floor(x)) {
                    return respond(json_error(400,
                                              R"(field "max_tokens" must be a non-negative integer at most 1000000)"),
                                   400, nullptr);
                }
                max_tokens = static_cast<int>(x);
            }

            std::optional<double> temperature;
            if (body.contains("temperature")) {
                const auto& v = body["temperature"];
                if (!v.is_number()) {
                    return respond(json_error(400, R"(field "temperature" must be a number)"), 400, nullptr);
                }
                const double t = v.get<double>();
                if (!std::isfinite(t) || t < 0.0 || t > 2.0) {
                    return respond(
                        json_error(400, R"(field "temperature" must be a finite number between 0 and 2)"), 400,
                        nullptr);
                }
                temperature = t;
            }

            // Echo only the fields we understood (like returning validated params + result).
            nlohmann::json params = {{"prompt", prompt}};
            if (max_tokens.has_value()) {
                params["max_tokens"] = *max_tokens;
            }
            if (temperature.has_value()) {
                params["temperature"] = *temperature;
            }

            const GenerateResult gen =
                model_runner.generate(prompt, GenerateParams{max_tokens, temperature}, t_request);

            if (!gen.ok) {
                const int status = gen.misconfigured ? 503 : 502;
                return respond(json_error_string(status, gen.message), status, &gen);
            }

            const nlohmann::json response = {
                {"ok", true},
                {"model", gen.model_label},
                {"params", params},
                {"output", gen.output},
            };

            crow::response res(response.dump());
            res.set_header("Content-Type", "application/json");
            return respond(std::move(res), 200, &gen);
        });

    const uint16_t port = port_from_env();
    // .run() blocks forever serving requests (like uvicorn.run(...)).
    // If the port is taken, bind() throws std::system_error (we catch so the process exits cleanly).
    try {
        app.port(port).multithreaded().run();
    } catch (const std::system_error& e) {
        std::cerr << "listen failed on port " << port << ": " << e.what() << '\n'
                  << "Another process may be bound here. Try PORT=18081 or run: lsof -iTCP:"
                  << port << " -sTCP:LISTEN\n";
        return 1;
    }
    return 0;
}
