# mini-v

`mini-v` is a small C++ inference server for running a local `.gguf` model
through a llama.cpp-compatible command-line backend. It exposes a `/generate`
HTTP endpoint, sends each prompt to a separate llama subprocess, and adds a
simple scheduler in front of the model.

The scheduler turns each HTTP call into an inference request object, puts
requests in a queue, processes them on a worker thread, and groups nearby
arrivals with a 50ms micro-batching window.

## What It Demonstrates

- A C++ HTTP inference server using Crow and `nlohmann::json`.
- A first-class `InferenceRequest` object with request id, prompt, generation
  params, result storage, and a `promise`/`shared_future` wait path.
- A `ModelRunner` scheduler that accepts requests, queues them, and has one
  background worker consume them.
- Scheduler-level micro-batching: the worker waits up to 50ms after the first
  request and groups nearby arrivals into one batch cycle.
- Correct response mapping under concurrent traffic: each HTTP handler waits on
  its own request future and receives its own generated output.
- Benchmark tooling for average latency, p95 latency, throughput, and observed
  scheduler batch size.

## Request Lifecycle

1. A client sends `POST /generate` with a prompt and optional generation params.
2. `main.cpp` validates JSON and calls `model_runner.submit(...)`.
3. `ModelRunner` creates an `InferenceRequest`, assigns an id, pushes it onto
   the pending queue, and wakes the worker.
4. The worker waits for the first request, holds a 50ms batching window, and
   collects all requests that arrived during that window.
5. The current backend still executes each request sequentially through
   `llama-completion`, but the scheduler processes the grouped requests in one
   worker cycle and logs `batch_size`.
6. Each request stores its own `GenerateResult`, fulfills its private promise,
   and wakes the HTTP handler waiting on that request's future.

## Architecture

```text
HTTP clients
    |
    v
Crow /generate handlers
    |
    v
ModelRunner::submit(...)
    |
    v
pending_ queue  --50ms window-->  scheduler batch
    |
    v
single worker thread
    |
    v
llama.cpp CLI subprocess per request
    |
    v
promise.set_value(result) -> HTTP handler future.get()
```

This is Option B batching: real scheduler-level batching, while the current
llama.cpp subprocess backend still handles individual prompts. A future backend
could replace the sequential execution loop with true backend batching without
changing the request lifecycle.

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

Configure a local model and optional CLI path:

```sh
export LLAMA_MODEL=/absolute/path/to/model.gguf
export LLAMA_CLI=llama-completion
./build/server
```

Then call the server:

```sh
curl -s http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write one sentence about batching.","max_tokens":16,"temperature":0}'
```

## Benchmarking

The micro-batcher does not force an exact batch size. Instead, the benchmark
changes client concurrency and measures the batch sizes the scheduler actually
forms during the 50ms batching window. In practice, concurrency is the control
knob and observed average batch size is the batching result.

Start the server and capture scheduler logs:

```sh
./build/server 2>server.log
```

Then run the same benchmark at different concurrency levels:

```sh
python3 scripts/bench.py --label batch --requests 50 --concurrency-sweep 1,2,4,8,16 --server-log server.log
```

These runs are effectively testing different observed batch-size regimes:
higher concurrency gives more requests a chance to arrive inside the batching
window, so average batch size should generally rise.

For a single run:

```sh
python3 scripts/bench.py --label batch --requests 50 --concurrency 8 --server-log server.log
```

The script prints a Markdown table with average latency, p95 latency, requests/sec,
and average observed batch size when logs are provided.

## Design Tradeoffs/Learnings

- **CLI subprocess backend vs in-process model runtime:** `mini-v` uses a
  separate llama.cpp-compatible subprocess for each generation. This keeps the
  C++ server simple and isolates model execution, but it adds process startup
  and IPC/file overhead compared with linking directly against an inference
  library.
- **Scheduler-level batching vs true backend batching:** the scheduler groups
  requests that arrive close together, then processes them in one worker cycle.
  The current backend still runs each prompt individually, so this demonstrates
  batching behavior and request mapping without requiring a backend API that
  accepts multiple prompts at once.
- **50ms batching window vs latency:** waiting briefly lets more requests join a
  batch under concurrent traffic, which can improve throughput-oriented behavior.
  The tradeoff is that a lone request may wait up to the batching window before
  model execution starts.
- **One worker thread vs parallel workers:** a single worker makes request
  ordering, logging, and correctness easier to reason about. More workers could
  increase throughput for a backend that supports parallel execution, but would
  add more scheduling and resource-control complexity.