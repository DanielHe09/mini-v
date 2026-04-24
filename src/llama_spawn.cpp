#include "llama_spawn.hpp"

#if defined(_WIN32)

LlamaRunResult run_llama_completion(std::string_view /*prompt*/, int /*max_tokens*/,
                                     std::optional<double> /*temperature*/) {
    return {false, "llama subprocess backend is only implemented on macOS/Linux (not _WIN32)", ""};
}

#else

#include <algorithm>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string read_fd(const int fd) {
    std::string out;
    char buf[8192];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof buf);
        if (n > 0) {
            out.append(buf, static_cast<size_t>(n));
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

} // namespace

LlamaRunResult run_llama_completion(const std::string_view prompt, const int max_tokens,
                                     const std::optional<double> temperature) {
    const char* model_env = std::getenv("LLAMA_MODEL");
    if (!model_env || !*model_env) {
        return {false, "LLAMA_MODEL is not set (absolute path to a .gguf file)", ""};
    }

    struct stat st_model{};
    if (::stat(model_env, &st_model) != 0) {
        return {false, std::string("LLAMA_MODEL path does not exist: ") + model_env + " (" + std::strerror(errno) + ")",
                ""};
    }
    if (!S_ISREG(st_model.st_mode)) {
        return {false, std::string("LLAMA_MODEL is not a regular file: ") + model_env, ""};
    }

    const char* cli_env = std::getenv("LLAMA_CLI");
    const std::string cli =
        (cli_env && *cli_env) ? std::string(cli_env) : std::string("llama-completion");

    if (cli.find('/') != std::string::npos) {
        if (::access(cli.c_str(), X_OK) != 0) {
            return {false, std::string("LLAMA_CLI is not an executable file: ") + cli + " (" + std::strerror(errno) +
                            ")",
                    ""};
        }
    }

    char tmpl[] = "/tmp/mini-v-prompt-XXXXXX";
    const int prompt_fd = ::mkstemp(tmpl);
    if (prompt_fd < 0) {
        return {false, std::string("mkstemp failed: ") + std::strerror(errno), ""};
    }

    {
        std::string_view chunk = prompt;
        while (!chunk.empty()) {
            const ssize_t w = ::write(prompt_fd, chunk.data(), chunk.size());
            if (w < 0) {
                if (errno == EINTR) {
                    continue;
                }
                ::close(prompt_fd);
                ::unlink(tmpl);
                return {false, std::string("write prompt temp file failed: ") + std::strerror(errno), ""};
            }
            chunk.remove_prefix(static_cast<size_t>(w));
        }
    }

    if (::close(prompt_fd) < 0) {
        ::unlink(tmpl);
        return {false, std::string("close prompt temp file failed: ") + std::strerror(errno), ""};
    }

    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    if (::pipe(out_pipe) != 0 || ::pipe(err_pipe) != 0) {
        ::unlink(tmpl);
        return {false, std::string("pipe failed: ") + std::strerror(errno), ""};
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        ::close(err_pipe[0]);
        ::close(err_pipe[1]);
        ::unlink(tmpl);
        return {false, std::string("fork failed: ") + std::strerror(errno), ""};
    }

    if (pid == 0) {
        ::close(out_pipe[0]);
        ::close(err_pipe[0]);
        if (::dup2(out_pipe[1], STDOUT_FILENO) < 0 || ::dup2(err_pipe[1], STDERR_FILENO) < 0) {
            _exit(126);
        }
        ::close(out_pipe[1]);
        ::close(err_pipe[1]);

        const int nullfd = ::open("/dev/null", O_RDONLY);
        if (nullfd >= 0) {
            (void)::dup2(nullfd, STDIN_FILENO);
            ::close(nullfd);
        }

        const int n_predict = std::max(1, max_tokens);
        const std::string n_str = std::to_string(n_predict);
        std::string temp_str;

        std::vector<std::string> storage;
        storage.push_back(cli);
        storage.push_back("-m");
        storage.push_back(model_env);
        storage.push_back("--file");
        storage.push_back(tmpl);
        storage.push_back("-n");
        storage.push_back(n_str);
        storage.push_back("-no-cnv");
        storage.push_back("--simple-io");
        if (temperature.has_value()) {
            storage.push_back("--temp");
            temp_str = std::to_string(*temperature);
            storage.push_back(temp_str);
        }

        std::vector<char*> argv;
        argv.reserve(storage.size() + 1);
        for (auto& s : storage) {
            argv.push_back(s.data());
        }
        argv.push_back(nullptr);

        (void)::execvp(argv[0], argv.data());
        _exit(127);
    }

    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    std::string stdout_str;
    std::string stderr_str;
    std::thread drain_out([&] { stdout_str = read_fd(out_pipe[0]); });
    std::thread drain_err([&] { stderr_str = read_fd(err_pipe[0]); });

    int status = 0;
    const pid_t w = ::waitpid(pid, &status, 0);

    drain_out.join();
    drain_err.join();
    ::close(out_pipe[0]);
    ::close(err_pipe[0]);
    ::unlink(tmpl);

    if (w < 0) {
        return {false, std::string("waitpid failed: ") + std::strerror(errno), ""};
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::ostringstream oss;
        oss << "llama inference process failed";
        if (WIFEXITED(status)) {
            oss << " (exit " << WEXITSTATUS(status) << ")";
        }
        if (!stderr_str.empty()) {
            oss << ": " << stderr_str;
        } else if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
            oss << ": exec failed; check LLAMA_CLI is on PATH or an absolute path";
        }
        return {false, oss.str(), ""};
    }

    while (!stdout_str.empty() &&
           (stdout_str.back() == '\n' || stdout_str.back() == '\r' || stdout_str.back() == ' ')) {
        stdout_str.pop_back();
    }

    return {true, "", stdout_str};
}

#endif
