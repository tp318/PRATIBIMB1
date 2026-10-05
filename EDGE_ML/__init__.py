"""
EDGE_ML — On-device INT8 Inference Engine for PRATIBIMB.

Provides:
  1. INT8 static quantizer — converts float32 MLPAutoencoder weights to
     INT8 using per-layer symmetric min-max calibration.
  2. QuantizedMLPAutoencoder — INT8 inference engine that runs the full
     forward pass with integer arithmetic, designed for ARM Cortex-A53/A55
     avionics processors (Raspberry Pi CM4, NVIDIA Orin Nano).
  3. LatencyProfiler — measures per-step inference latency and memory
     footprint on the target device.
  4. ONNXExporter — exports the float32 model to ONNX for use with
     ONNX Runtime (with optional INT8 quantization via onnxruntime.quantization).

Target performance (ARM Cortex-A53, 1.2 GHz):
    Float32 forward pass:  ~0.9 ms
    INT8 forward pass:     ~0.2 ms  (≈4.5x speedup)
    Memory (INT8 weights): ~2.4 KB for default model_dim=9, hidden_dim=32

Usage:
    from EDGE_ML import INT8Quantizer, QuantizedMLPAutoencoder, LatencyProfiler

    quantizer = INT8Quantizer()
    q_model = quantizer.quantize(float_model, calibration_data)
    score = q_model.anomaly_score(sensor_window)

    profiler = LatencyProfiler(q_model, n_runs=1000)
    stats = profiler.run()
    print(stats)  # {'mean_ms': 0.21, 'p95_ms': 0.28, 'memory_bytes': 2456}
"""

from .quantizer import INT8Quantizer, QuantizationConfig
from .inference import QuantizedMLPAutoencoder
from .profiler import LatencyProfiler
from .exporter import ONNXExporter

__all__ = [
    "INT8Quantizer",
    "QuantizationConfig",
    "QuantizedMLPAutoencoder",
    "LatencyProfiler",
    "ONNXExporter",
]
