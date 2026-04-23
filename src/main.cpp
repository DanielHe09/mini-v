// Crow = small C++ HTTP framework (think Starlette/Flask-level, not batteries-included like FastAPI).
// nlohmann::json = JSON as nested maps/arrays (think Python dict + json.loads / dumps).

#include <crow.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
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

int main() {
    // One app object; routes register on it (like `app = FastAPI()` then decorators).
    crow::SimpleApp app;

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
        .methods(crow::HTTPMethod::Post)([](const crow::request& req) {
            auto json_error = [](int status, const char* message) {
                crow::response res(status, nlohmann::json{{"error", message}}.dump());
                res.set_header("Content-Type", "application/json");
                return res;
            };

            // Parse JSON body (expect a single object, like a FastAPI JSON body model).
            nlohmann::json body = nlohmann::json::object();
            if (!req.body.empty()) {
                try {
                    body = nlohmann::json::parse(req.body);
                } catch (const nlohmann::json::parse_error&) {
                    return json_error(400, "invalid json");
                }
            }

            if (!body.is_object()) {
                return json_error(400, "body must be a JSON object");
            }

            //extract some basic parameters like prompt, max_tokens, temperature. Last 2 are optional
            if (!body.contains("prompt") || !body["prompt"].is_string()) {
                return json_error(400, R"(field "prompt" (string) is required)");
            }
            const std::string prompt = body["prompt"].get<std::string>();

            std::optional<int> max_tokens;
            if (body.contains("max_tokens")) {
                const auto& v = body["max_tokens"];
                if (!v.is_number()) {
                    return json_error(400, R"(field "max_tokens" must be a number)");
                }
                const double x = v.get<double>();
                if (!std::isfinite(x) || x < 0.0 || x > 1'000'000.0 || x != std::floor(x)) {
                    return json_error(400,
                                      R"(field "max_tokens" must be a non-negative integer at most 1000000)");
                }
                max_tokens = static_cast<int>(x);
            }

            std::optional<double> temperature;
            if (body.contains("temperature")) {
                const auto& v = body["temperature"];
                if (!v.is_number()) {
                    return json_error(400, R"(field "temperature" must be a number)");
                }
                const double t = v.get<double>();
                if (!std::isfinite(t) || t < 0.0 || t > 2.0) {
                    return json_error(400, R"(field "temperature" must be a finite number between 0 and 2)");
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

            const nlohmann::json response = {
                {"ok", true},
                {"model", "stub"},
                {"params", params},
                {"output", "hardcoded fake completion for POST /generate"},
            };

            crow::response res(response.dump());
            res.set_header("Content-Type", "application/json");
            return res;
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
