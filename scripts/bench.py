#!/usr/bin/env python3
"""Benchmark mini-v /generate latency and throughput.

Run this against an already-running server. To compare direct, queued, and
micro-batched versions, start each version separately and pass a different
--label for each run.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


BATCH_SIZE_RE = re.compile(r"\bbatch_size=(\d+)\b")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * pct)))
    return ordered[index]


def parse_batch_sizes(path: Path | None) -> list[int]:
    if path is None or not path.exists():
        return []

    sizes: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = BATCH_SIZE_RE.search(line)
            if match:
                sizes.append(int(match.group(1)))
    return sizes


def post_generate(url: str, prompt: str, max_tokens: int, temperature: float) -> tuple[float, bool]:
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            ok = 200 <= response.status < 300
            response.read()
    except urllib.error.HTTPError as exc:
        ok = False
        exc.read()
    except urllib.error.URLError:
        ok = False
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, ok


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    url = args.url.rstrip("/") + "/generate"
    prompts = [f"{args.prompt} #{i}" for i in range(args.requests)]

    started = time.perf_counter()
    latencies_ms: list[float] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(post_generate, url, prompt, args.max_tokens, args.temperature)
            for prompt in prompts
        ]
        for future in as_completed(futures):
            latency_ms, ok = future.result()
            latencies_ms.append(latency_ms)
            if not ok:
                failures += 1
    total_s = time.perf_counter() - started

    batch_sizes = parse_batch_sizes(args.server_log)
    successes = args.requests - failures
    return {
        "label": args.label,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successes": successes,
        "failures": failures,
        "avg_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "throughput_rps": successes / total_s if total_s > 0 else 0.0,
        "avg_batch_size": statistics.fmean(batch_sizes) if batch_sizes else None,
        "observed_batches": len(batch_sizes),
    }


def print_markdown_table(row: dict[str, object]) -> None:
    avg_batch_size = row["avg_batch_size"]
    batch_text = "-" if avg_batch_size is None else f"{avg_batch_size:.2f}"
    print("| mode | requests | concurrency | success | fail | avg latency ms | p95 latency ms | req/s | avg batch size | batches |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| {row['label']} | {row['requests']} | {row['concurrency']} | "
        f"{row['successes']} | {row['failures']} | "
        f"{row['avg_latency_ms']:.2f} | {row['p95_latency_ms']:.2f} | "
        f"{row['throughput_rps']:.2f} | {batch_text} | {row['observed_batches']} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark mini-v /generate.")
    parser.add_argument("--url", default="http://127.0.0.1:18080", help="Server base URL.")
    parser.add_argument("--label", default="micro-batched", help="Mode label for the table row.")
    parser.add_argument("--requests", type=int, default=20, help="Total requests to send.")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent client workers.")
    parser.add_argument("--max-tokens", type=int, default=16, help="max_tokens sent to /generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature sent to /generate.")
    parser.add_argument("--prompt", default="Write one short sentence about batching.", help="Prompt prefix.")
    parser.add_argument(
        "--server-log",
        type=Path,
        help="Optional server stderr log containing scheduler batch_size lines.",
    )
    args = parser.parse_args()

    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")

    row = run_benchmark(args)
    print_markdown_table(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
