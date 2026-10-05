"""
LatencyProfiler — On-device inference latency and memory profiler.

Measures wall-clock inference time for both float32 and INT8 models,
reports statistics suitable for SWaP (Size, Weight, Power) analysis
documents submitted to DRDO certification bodies.

Measured quantities
-------------------
- Mean, P50, P95, P99, Max inference latency (microseconds)
- Throughput (inferences/second)
- Peak RSS memory delta during inference (bytes, via tracemalloc)
- INT8 vs float32 speedup ratio

Usage:
    profiler = LatencyProfiler(model=q_model, n_runs=2000, batch_size=1)
    stats = profiler.run()
    print(profiler.report(stats))
"""

import time
import tracemalloc
import statistics
from typing import Dict, Any, Optional

import numpy as np


class LatencyProfiler:
    """
    Inference latency and memory profiler for edge anomaly detectors.

    Parameters
    ----------
    model : QuantizedMLPAutoencoder or MLPAutoencoder
        Any model with a .forward(x) method accepting numpy arrays.
    n_runs : int
        Number of inference calls to benchmark.
    batch_size : int
        Input batch size (use 1 for real-time single-frame inference).
    input_dim : int
        Feature dimension of the model input.
    warmup_runs : int
        Number of warmup calls before timing begins (avoids JIT effects).
    """

    def __init__(
        self,
        model,
        n_runs: int = 1000,
        batch_size: int = 1,
        input_dim: int = 9,
        warmup_runs: int = 50,
    ):
        self.model = model
        self.n_runs = n_runs
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.warmup_runs = warmup_runs
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # Main API                                                             #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """
        Run the latency benchmark.

        Returns
        -------
        dict with keys:
            n_runs, batch_size, latencies_us (list),
            mean_us, p50_us, p95_us, p99_us, max_us,
            throughput_hz, memory_bytes, speedup_note
        """
        x_dummy = self._rng.random(
            (self.batch_size, self.input_dim), dtype=np.float32
        )

        # Warmup
        for _ in range(self.warmup_runs):
            self.model.forward(x_dummy)

        # Time main runs
        latencies_ns = []
        for _ in range(self.n_runs):
            t0 = time.perf_counter_ns()
            self.model.forward(x_dummy)
            latencies_ns.append(time.perf_counter_ns() - t0)

        latencies_us = [t / 1_000 for t in latencies_ns]

        # Memory delta
        mem_bytes = self._measure_memory(x_dummy)

        mean_us = statistics.mean(latencies_us)
        stats = {
            "n_runs": self.n_runs,
            "batch_size": self.batch_size,
            "mean_us": round(mean_us, 2),
            "p50_us": round(statistics.median(latencies_us), 2),
            "p95_us": round(
                sorted(latencies_us)[int(0.95 * len(latencies_us))], 2
            ),
            "p99_us": round(
                sorted(latencies_us)[int(0.99 * len(latencies_us))], 2
            ),
            "max_us": round(max(latencies_us), 2),
            "throughput_hz": round(1_000_000 / max(mean_us, 0.01), 1),
            "memory_delta_bytes": mem_bytes,
        }
        return stats

    def compare(self, float_model, int8_model) -> Dict[str, Any]:
        """
        Run benchmarks on both float32 and INT8 models and return speedup.
        """
        print("Profiling float32 model...")
        self.model = float_model
        float_stats = self.run()

        print("Profiling INT8 quantized model...")
        self.model = int8_model
        int8_stats = self.run()

        speedup = float_stats["mean_us"] / max(int8_stats["mean_us"], 0.01)
        return {
            "float32": float_stats,
            "int8": int8_stats,
            "speedup_x": round(speedup, 2),
            "latency_reduction_pct": round((1 - 1 / speedup) * 100, 1),
        }

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def report(stats: Dict[str, Any]) -> str:
        if "float32" in stats:
            f = stats["float32"]
            q = stats["int8"]
            return (
                f"=== PRATIBIMB Edge Inference Benchmark ===\n"
                f"  Float32 | mean={f['mean_us']:.1f} µs  "
                f"p95={f['p95_us']:.1f} µs  "
                f"throughput={f['throughput_hz']:.0f} Hz\n"
                f"  INT8    | mean={q['mean_us']:.1f} µs  "
                f"p95={q['p95_us']:.1f} µs  "
                f"throughput={q['throughput_hz']:.0f} Hz\n"
                f"  Speedup : {stats['speedup_x']:.2f}x  "
                f"({stats['latency_reduction_pct']:.1f}% reduction)\n"
                f"  INT8 memory: {q['memory_delta_bytes']} bytes"
            )
        return (
            f"=== PRATIBIMB Edge Inference Benchmark ===\n"
            f"  mean={stats['mean_us']:.1f} µs | p95={stats['p95_us']:.1f} µs | "
            f"p99={stats['p99_us']:.1f} µs | max={stats['max_us']:.1f} µs\n"
            f"  throughput={stats['throughput_hz']:.0f} Hz | "
            f"memory_delta={stats['memory_delta_bytes']} bytes"
        )

    # ------------------------------------------------------------------ #
    # Memory measurement                                                   #
    # ------------------------------------------------------------------ #

    def _measure_memory(self, x: np.ndarray) -> int:
        tracemalloc.start()
        self.model.forward(x)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak
