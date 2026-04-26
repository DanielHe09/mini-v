#include "llama_server_client.hpp"

#include <nlohmann/json.hpp>

#if defined(_WIN32)

LlamaRunResult run_llama_server_completion(std::string_view /*prompt*/, int /*max_tokens*/,
                                           std::optional<double> /*temperature*/) {
    return {false, "llama-server HTTP backend is only implemented on macOS/Linux (not _WIN32)", ""};
}

#else

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>

namespace {

struct ParsedUrl {
    std::string host;
    std::string port;
    std::string path_prefix;
};

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

ParsedUrl parse_url(const std::string& url) {
    constexpr std::string_view scheme = "http://";
    if (url.compare(0, scheme.size(), scheme) != 0) {
        throw std::runtime_error("LLAMA_SERVER_URL must start with http://");
    }

    const std::size_t authority_start = scheme.size();
    const std::size_t path_start = url.find('/', authority_start);
    const std::string authority = url.substr(authority_start, path_start - authority_start);
    if (authority.empty()) {
        throw std::runtime_error("LLAMA_SERVER_URL is missing a host");
    }

    ParsedUrl parsed;
    const std::size_t colon = authority.rfind(':');
    if (colon == std::string::npos) {
        parsed.host = authority;
        parsed.port = "80";
    } else {
        parsed.host = authority.substr(0, colon);
        parsed.port = authority.substr(colon + 1);
        if (parsed.host.empty() || parsed.port.empty()) {
            throw std::runtime_error("LLAMA_SERVER_URL has an invalid host or port");
        }
    }

    parsed.path_prefix = path_start == std::string::npos ? "" : url.substr(path_start);
    while (!parsed.path_prefix.empty() && parsed.path_prefix.back() == '/') {
        parsed.path_prefix.pop_back();
    }
    return parsed;
}

bool write_all(const int fd, const std::string& data) {
    std::string_view remaining(data);
    while (!remaining.empty()) {
        const ssize_t written = ::send(fd, remaining.data(), remaining.size(), 0);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        remaining.remove_prefix(static_cast<std::size_t>(written));
    }
    return true;
}

std::string read_all(const int fd) {
    std::string out;
    char buf[8192];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n > 0) {
            out.append(buf, static_cast<std::size_t>(n));
            continue;
        }
        if (n == 0) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        break;
    }
    return out;
}

std::pair<bool, std::string> http_post(const ParsedUrl& url, const std::string& path,
                                       const std::string& body) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* results = nullptr;
    const int gai = ::getaddrinfo(url.host.c_str(), url.port.c_str(), &hints, &results);
    if (gai != 0) {
        return {false, std::string("resolve llama-server failed: ") + ::gai_strerror(gai)};
    }

    int fd = -1;
    for (addrinfo* ai = results; ai != nullptr; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(results);

    if (fd < 0) {
        return {false, "connect to llama-server failed"};
    }

    std::ostringstream request;
    request << "POST " << path << " HTTP/1.1\r\n"
            << "Host: " << url.host << ":" << url.port << "\r\n"
            << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Connection: close\r\n\r\n"
            << body;

    if (!write_all(fd, request.str())) {
        const std::string message = std::string("write to llama-server failed: ") + std::strerror(errno);
        ::close(fd);
        return {false, message};
    }

    std::string response = read_all(fd);
    ::close(fd);
    return {true, std::move(response)};
}

std::string decode_chunked_body(std::string body) {
    std::string decoded;
    std::size_t pos = 0;
    for (;;) {
        const std::size_t line_end = body.find("\r\n", pos);
        if (line_end == std::string::npos) {
            return body;
        }
        const std::string size_text = body.substr(pos, line_end - pos);
        const std::size_t chunk_size = std::stoul(size_text, nullptr, 16);
        if (chunk_size == 0) {
            return decoded;
        }
        pos = line_end + 2;
        if (pos + chunk_size > body.size()) {
            return body;
        }
        decoded.append(body, pos, chunk_size);
        pos += chunk_size + 2;
    }
}

LlamaRunResult parse_completion_response(const std::string& raw_response) {
    const std::size_t header_end = raw_response.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return {false, "llama-server returned an invalid HTTP response", ""};
    }

    const std::string headers = raw_response.substr(0, header_end);
    std::string body = raw_response.substr(header_end + 4);

    std::istringstream header_stream(headers);
    std::string http_version;
    int status = 0;
    header_stream >> http_version >> status;
    if (status < 200 || status >= 300) {
        return {false, "llama-server returned HTTP " + std::to_string(status) + ": " + body, ""};
    }

    const std::string lower_headers = lower_ascii(headers);
    if (lower_headers.find("transfer-encoding: chunked") != std::string::npos) {
        body = decode_chunked_body(std::move(body));
    }

    try {
        const nlohmann::json json = nlohmann::json::parse(body);
        if (json.contains("content") && json["content"].is_string()) {
            return {true, "", json["content"].get<std::string>()};
        }
        return {false, "llama-server response did not contain string field \"content\"", ""};
    } catch (const nlohmann::json::parse_error& e) {
        return {false, std::string("parse llama-server response failed: ") + e.what(), ""};
    }
}

} // namespace

LlamaRunResult run_llama_server_completion(const std::string_view prompt, const int max_tokens,
                                           const std::optional<double> temperature) {
    const char* env_url = std::getenv("LLAMA_SERVER_URL");
    const std::string server_url = (env_url && *env_url) ? std::string(env_url) : "http://127.0.0.1:8080";

    ParsedUrl parsed;
    try {
        parsed = parse_url(server_url);
    } catch (const std::exception& e) {
        return {false, e.what(), ""};
    }

    const int n_predict = std::max(1, max_tokens);
    nlohmann::json body = {
        {"prompt", std::string(prompt)},
        {"n_predict", n_predict},
    };
    if (temperature.has_value()) {
        body["temperature"] = *temperature;
    }

    const std::string path = parsed.path_prefix + "/completion";
    auto [ok, response] = http_post(parsed, path, body.dump());
    if (!ok) {
        return {false, response, ""};
    }
    return parse_completion_response(response);
}

#endif
