# mini-v

`mini-v` is a small C++ inference server for running a local `.gguf` model
through a long-running `llama-server` backend. It exposes a `/generate` HTTP
endpoint, adds a simple scheduler in front of the model, and forwards grouped
requests to llama.cpp's continuous batching server.

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
- Backend-level continuous batching by forwarding grouped requests concurrently
  to `llama-server`.
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
5. The worker fans out the grouped requests as concurrent `/completion` calls to
   `llama-server`, which keeps the model loaded and performs backend-level
   continuous batching across active requests.
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
concurrent /completion calls
    |
    v
llama-server continuous batching backend
    |
    v
promise.set_value(result) -> HTTP handler future.get()
```

`mini-v` still owns request admission, queueing, response mapping, and benchmark
logging. Actual model execution is delegated to `llama-server`, so grouped
requests can be decoded by a backend that keeps the model loaded and supports
continuous batching.

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

Start `llama-server` with a local model:

```sh
/Users/danielhe/Desktop/mini-v/llama.cpp/build/bin/llama-server \
  -m "/Users/danielhe/Desktop/ML models/gemma-4-E2B-it-Q4_K_M.gguf" \
  --host 127.0.0.1 \
  --port 8080 \
  --parallel 8 \
  --cont-batching \
  -n 16
```

Then start `mini-v` and point it at that backend:

```sh
export LLAMA_SERVER_URL=http://127.0.0.1:8080
./build/server 2>server.log
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

With the `llama-server` backend, `avg batch size` is still the scheduler batch
size observed inside `mini-v`. The actual model-level batching happens inside
`llama-server` through parallel slots and continuous batching.

Start `llama-server` in one terminal:

```sh
/Users/danielhe/Desktop/mini-v/llama.cpp/build/bin/llama-server \
  -m "/Users/danielhe/Desktop/ML models/gemma-4-E2B-it-Q4_K_M.gguf" \
  --host 127.0.0.1 \
  --port 8080 \
  --parallel 8 \
  --cont-batching \
  -n 16
```

Then start `mini-v` in another terminal and capture scheduler logs:

```sh
export LLAMA_SERVER_URL=http://127.0.0.1:8080
./build/server 2>server.log
```

Then run the benchmark sweep:

```sh
python3 scripts/bench.py --label batch --requests 50 --concurrency-sweep 1,2,4,8,16 --server-log server.log
```

If all rows show `success = 0` and `fail = 50`, check that `llama-server` is
still running and reachable at `LLAMA_SERVER_URL`.

These runs are effectively testing different observed batch-size regimes:
higher concurrency gives more requests a chance to arrive inside the batching
window, so average batch size should generally rise.

For a single run:

```sh
python3 scripts/bench.py --label batch --requests 50 --concurrency 8 --server-log server.log
```

The script prints a Markdown table with average latency, p95 latency, requests/sec,
and average observed batch size when logs are provided.

### Sample llama-server Backend Run

This sample run was captured after switching to `llama-server` as the backend.
It reflects end-to-end behavior with scheduler batching in `mini-v` plus
continuous batching in the llama.cpp server.

| run | requests | concurrency | success | fail | avg latency ms | p95 latency ms | req/s | avg batch size | batches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| batch-c1 | 50 | 1 | 50 | 0 | 152.86 | 186.02 | 6.54 | 1.00 | 50 |
| batch-c2 | 50 | 2 | 50 | 0 | 183.05 | 236.90 | 10.92 | 2.00 | 25 |
| batch-c4 | 50 | 4 | 50 | 0 | 239.39 | 498.64 | 16.26 | 3.85 | 13 |
| batch-c8 | 50 | 8 | 50 | 0 | 512.74 | 671.89 | 15.02 | 7.14 | 7 |
| batch-c16 | 50 | 16 | 50 | 0 | 852.73 | 1248.90 | 16.83 | 8.33 | 6 |

Analysis:

- All 50 requests succeeded at every concurrency level, so these numbers reflect
  real end-to-end generation rather than configuration or backend failures.
- Average scheduler batch size increases as concurrency rises, from 1.00 at
  concurrency 1 to 8.33 at concurrency 16. This shows the 50ms micro-batching
  window is grouping nearby requests as intended.
- The number of scheduler batches drops from 50 to 6 across the sweep, meaning
  the worker handles more requests per scheduling cycle under higher load.
- Throughput improves strongly from 6.54 req/s at concurrency 1 to 16.26 req/s
  at concurrency 4, then flattens around 15-17 req/s. This is consistent with
  the backend nearing its capacity while handling more concurrent work.
- Latency rises with concurrency because each request waits behind more queued
  generations. At concurrency 16, p95 latency reaches about 1248.90 ms, showing
  the throughput/latency tradeoff once the system is near saturation.
- This run shows strong throughput with clean request success across the full
  sweep, while keeping latency reasonable at lower to mid concurrency.

## Design Tradeoffs/Learnings

- **llama-server backend:** `mini-v` delegates inference to a long-running
  `llama-server`, which keeps the model loaded and lets llama.cpp handle
  continuous batching internally.
- **Scheduler-level batching vs backend-level batching:** the scheduler groups
  requests that arrive close together, then fans them out concurrently to the
  backend. `mini-v` logs scheduler batch size, while `llama-server` owns actual
  model decoding and parallel slot management.
- **50ms batching window vs latency:** waiting briefly lets more requests join a
  batch under concurrent traffic, which can improve throughput-oriented behavior.
  The tradeoff is that a lone request may wait up to the batching window before
  model execution starts.
- **One scheduler worker vs backend parallelism:** a single scheduler worker
  keeps request grouping and response mapping easy to reason about. Backend
  parallelism now lives in `llama-server`, configured with options such as
  `--parallel` and `--cont-batching`.

## Future Goal: In-Process libllama Backend

A lower-level future version could link `mini-v` directly against `libllama`
instead of proxying to `llama-server`. That would make batching fully
in-process, but it requires owning much more inference machinery:

- Load the GGUF model and create a persistent `llama_context` inside `mini-v`.
- Tokenize prompts and manage one sequence id per active request.
- Build `llama_batch` objects containing tokens from multiple active sequences.
- Run `llama_decode` loops, sample tokens independently for each request, and
  detect EOS or stop conditions per sequence.
- Manage KV cache lifetime, context limits, cancellation, and error handling.
- Preserve the same promise/future response mapping that the current scheduler
  already provides.

That path is more educational because it exposes the mechanics of batched
decoding directly, but `llama-server` is the practical backend for this version.