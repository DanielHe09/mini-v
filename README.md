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