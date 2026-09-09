"""
AeroTwin-4 Performance Benchmarking.

Measures physics update latency (Average, P95, Max) and execution frequency (Hz)
for the EngineRunner simulation system.
"""

import time
import numpy as np
from typing import Dict, Any

from .runner import EngineRunner


def benchmark_runner(runner: EngineRunner = None, steps: int = 5000) -> Dict[str, Any]:
    """
    Benchmark simulation step execution latency over specified number of steps.

    Parameters
    ----------
    runner : EngineRunner, optional
        EngineRunner instance to benchmark.
    steps : int
        Number of step iterations to measure.

    Returns
    -------
    dict
        Benchmark results containing:
        - steps_evaluated
        - total_time_ms
        - avg_latency_ms
        - p95_latency_ms
        - max_latency_ms
        - execution_rate_hz
    """
    if runner is None:
        runner = EngineRunner(dt=0.01)

    runner.reset(seed=42)
    runner.start()

    latencies_ms = []

    # Warmup steps
    for _ in range(50):
        runner.step()

    # Measured steps
    for _ in range(steps):
        t0 = time.perf_counter_ns()
        runner.step()
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)

    latencies = np.array(latencies_ms)
    total_time_s = np.sum(latencies) / 1000.0
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    max_latency = float(np.max(latencies))
    rate_hz = float(steps / total_time_s) if total_time_s > 0 else 0.0

    return {
        "steps_evaluated": steps,
        "total_time_seconds": float(total_time_s),
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "max_latency_ms": max_latency,
        "execution_rate_hz": rate_hz,
    }
