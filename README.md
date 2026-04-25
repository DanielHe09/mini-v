## Benchmarking

Start the server version you want to measure, then run:

```sh
python3 scripts/bench.py --label micro-batched --requests 50 --concurrency 8
```

To include average scheduler batch size, capture server stderr and pass it back:

```sh
./build/server 2>server.log
python3 scripts/bench.py --label micro-batched --requests 50 --concurrency 8 --server-log server.log
```

Use the same script against different builds or branches to compare modes:

```sh
python3 scripts/bench.py --label direct-sync --requests 50 --concurrency 8
python3 scripts/bench.py --label queued-worker --requests 50 --concurrency 8
python3 scripts/bench.py --label micro-batched --requests 50 --concurrency 8 --server-log server.log
```

The script prints a Markdown table with average latency, p95 latency, requests/sec,
and average observed batch size when logs are provided.