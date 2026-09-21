# =============================================================================
# load_test.py
# PURPOSE: Fire 100 concurrent HTTP requests at the /predict endpoint and
#          measure how long each one takes (latency).
#
# KEY CONCEPTS:
#   asyncio.gather()  → run many coroutines concurrently on ONE event loop
#   httpx.AsyncClient → async HTTP client (non-blocking, unlike requests lib)
#   Latency metrics   → P50, P95, P99 tell us about real-world performance
# =============================================================================

import asyncio
import time
import httpx
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL        = "http://127.0.0.1:8000/predict"
NUM_REQUESTS   = 100   # Total concurrent requests to fire
TIMEOUT_SECS   = 30    # Give each request up to 30s before declaring failure

# Sample payload: 5 random-ish feature values (matching the model's input shape)
SAMPLE_PAYLOAD = {"features": [1.2, -0.5, 3.1, 0.0, 2.7]}


async def single_request(client: httpx.AsyncClient, request_id: int) -> float:
    """
    Sends ONE POST request and returns how long it took (in milliseconds).

    This is a coroutine (async def).  When we 'await client.post()',
    the event loop suspends THIS coroutine and goes to handle other
    coroutines — no thread is blocked while waiting for the HTTP response.

    Args:
        client     : shared async HTTP client (reuses TCP connections)
        request_id : just for printing, so we know which request finished
    Returns:
        latency in milliseconds for this single request
    """
    start = time.perf_counter()                          # high-precision timer

    response = await client.post(API_URL, json=SAMPLE_PAYLOAD)  # non-blocking!

    elapsed_ms = (time.perf_counter() - start) * 1000   # convert to ms

    # Print result for each request so you can watch them trickle back
    # Notice: they do NOT arrive in order — fast ones return first!
    status = response.status_code
    data   = response.json()
    print(f"  [Request {request_id:>3}] ✅ {status} | "
          f"class={data['predicted_class']} | "
          f"prob={data['probability']:.3f} | "
          f"latency={elapsed_ms:.1f}ms")

    return elapsed_ms


async def run_load_test():
    """
    Fires all NUM_REQUESTS concurrently and prints a latency summary.

    HOW asyncio.gather() creates concurrency:
      - We create 100 coroutine objects (single_request calls).
      - asyncio.gather() starts ALL of them "at once" on the event loop.
      - Each coroutine runs until it hits 'await client.post()', then
        suspends and lets the next coroutine start.
      - The event loop juggles all 100 coroutines until every one finishes.
      - This means all 100 HTTP requests are IN-FLIGHT at the same time,
        not sent one-by-one sequentially.
    """
    print(f"\n{'='*60}")
    print(f"  🔥 Firing {NUM_REQUESTS} concurrent requests at {API_URL}")
    print(f"{'='*60}\n")

    # httpx.AsyncClient is like the 'requests' library but async.
    # Using it as a context manager ensures the TCP connection pool is
    # properly opened at start and closed at the end.
    async with httpx.AsyncClient(timeout=TIMEOUT_SECS) as client:

        wall_clock_start = time.perf_counter()  # total elapsed time for ALL requests

        # Build a list of 100 coroutines (NOT yet running — just defined)
        tasks = [
            single_request(client, request_id=i)
            for i in range(1, NUM_REQUESTS + 1)
        ]

        # asyncio.gather() launches all 100 coroutines concurrently and
        # waits until every single one has completed.
        # 'latencies' is a list of 100 floats (one per request).
        latencies = await asyncio.gather(*tasks)

        total_wall_time = (time.perf_counter() - wall_clock_start) * 1000

    # ── Latency Report ────────────────────────────────────────────────────────
    latencies = np.array(latencies)   # convert to numpy for easy percentile math

    print(f"\n{'='*60}")
    print(f"  📊 LOAD TEST RESULTS ({NUM_REQUESTS} requests)")
    print(f"{'='*60}")
    print(f"  Total wall-clock time : {total_wall_time:.1f} ms")
    print(f"  Requests succeeded    : {len(latencies)} / {NUM_REQUESTS}")
    print(f"  --- Per-Request Latency ---")
    print(f"  Min (fastest)         : {np.min(latencies):.1f} ms")
    print(f"  P50 (median)          : {np.percentile(latencies, 50):.1f} ms")
    print(f"  P95                   : {np.percentile(latencies, 95):.1f} ms")
    print(f"  P99                   : {np.percentile(latencies, 99):.1f} ms")
    print(f"  Max (slowest)         : {np.max(latencies):.1f} ms")
    print(f"{'='*60}\n")

    # WHY P95/P99 matter in production:
    #   Average latency can look fine even if 5% of users wait 10x longer.
    #   P99 = "99% of requests were faster than this value" — this is what
    #   SLAs (Service Level Agreements) are typically defined around.


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # asyncio.run() creates a fresh event loop, runs our async main function,
    # then cleanly closes the loop when done.
    asyncio.run(run_load_test())
