// Crow = small C++ HTTP framework (think Starlette/Flask-level, not batteries-included like FastAPI).
// nlohmann::json = JSON as nested maps/arrays (think Python dict + json.loads / dumps).

#include <crow.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
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
            // Parse JSON body if present (FastAPI would do this via a Pydantic model param).
            nlohmann::json body = nlohmann::json::object();
            if (!req.body.empty()) {
                try {
                    body = nlohmann::json::parse(req.body);
                } catch (const nlohmann::json::parse_error&) {
                    crow::response res(400, R"({"error":"invalid json"})");
                    res.set_header("Content-Type", "application/json");
                    return res;
                }
            }

            // Build a JSON object (initializer list syntax; like a dict literal).
            const nlohmann::json response = {
                {"ok", true},
                {"model", "stub"},
                {"echo", body},
                {"output", "hardcoded fake completion for POST /generate"},
            };

            // .dump() -> JSON string (like json.dumps). crow::response is the HTTP response type.
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
