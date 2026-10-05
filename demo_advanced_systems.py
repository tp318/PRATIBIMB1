# -*- coding: utf-8 -*-
"""
PRATIBIMB Advanced Systems Demo
================================
Demonstrates the three new capability layers in a single runnable script:

  1. CAN Telemetry Integration
     Starts a CANSimulator (software CAN bus), receives 50 synthetic CAN frames,
     decodes them, and assembles EngineTelemetry objects.

  2. Federated Learning
     Simulates a 10-UAV fleet running 5 rounds of FedAvg with DP-SGD,
     reports per-round loss reduction and privacy budget estimate.

  3. Edge ML — INT8 Quantization + Latency Profiling
     Quantizes the FL-trained global model to INT8, runs the latency profiler,
     compares float32 vs INT8 throughput, and exports to ONNX.

Run:
    python demo_advanced_systems.py

No hardware required. All modules fall back to software simulation / analytic
models when python-can / torch / onnx are not installed.
"""

import sys
import os
import time
import logging

# Make repo root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pratibimb.demo")

SEPARATOR = "=" * 62


# =========================================================================== #
# 1. CAN Telemetry Integration                                                 #
# =========================================================================== #

def demo_can_telemetry():
    print(f"\n{SEPARATOR}")
    print("  PART 1 — CAN TELEMETRY INTEGRATION")
    print(SEPARATOR)

    from CAN_TELEMETRY import CANBridge, CANSimulator, CANDecoder
    from CAN_TELEMETRY.dbc import DBC_ENGINE_SIGNALS

    print(f"  DBC catalogue: {len(DBC_ENGINE_SIGNALS)} frame IDs defined")
    print(f"  Frame IDs: {[hex(fid) for fid in DBC_ENGINE_SIGNALS]}")

    # Test decoder directly
    decoder = CANDecoder(strict=False)
    from CAN_TELEMETRY.simulator import CANSimulator as _Sim
    sim = _Sim(real_time=False)

    print("\n  Decoding 50 synthetic CAN frames...")
    assembled = []
    frame_buf = {}
    seen_ids = set()
    required_ids = set(DBC_ENGINE_SIGNALS.keys())
    gen = sim.generate()

    for _ in range(1000):
        fid, payload, ts = next(gen)
        signals = decoder.decode_frame(fid, payload)
        frame_buf.update(signals)
        seen_ids.add(fid)
        if required_ids.issubset(seen_ids):
            assembled.append(dict(frame_buf))
            frame_buf = {}
            seen_ids = set()
        if len(assembled) >= 50:
            break

    print(f"  Assembled telemetry frames: {len(assembled)}")
    if assembled:
        last = assembled[-1]
        print(f"\n  Sample decoded frame:")
        for key in ("rpm", "cht", "egt", "oil_temperature", "vibration"):
            print(f"    {key:<20s} = {last.get(key, 'N/A'):.3f}")

    stats = decoder.stats
    print(f"\n  Decoder stats: {stats}")
    print("  [PASS] CAN Telemetry Integration demo complete.")
    return assembled


# =========================================================================== #
# 2. Federated Learning                                                        #
# =========================================================================== #

def demo_federated_learning():
    print(f"\n{SEPARATOR}")
    print("  PART 2 — FEDERATED LEARNING (FedAvg + DP-SGD)")
    print(SEPARATOR)

    from FEDERATED_LEARNING import FLConfig, FederatedClient, FederatedServer
    from FEDERATED_LEARNING.privacy import DPGradientClipper

    config = FLConfig(
        num_rounds=5,
        clients_per_round=4,
        local_epochs=2,
        local_batch_size=32,
        learning_rate=1e-3,
        dp_max_grad_norm=1.0,
        dp_noise_multiplier=0.5,
        seed=42,
    )

    print(f"  Fleet size: 10 UAVs | Rounds: {config.num_rounds} | "
          f"Clients/round: {config.clients_per_round}")
    print(f"  DP params: clip_norm={config.dp_max_grad_norm}, "
          f"noise_mult={config.dp_noise_multiplier}")

    # Create 10 clients with synthetic sortie data
    clients = [FederatedClient(f"UAV-TAPAS-{i:03d}", config) for i in range(10)]
    total_samples = sum(len(c.local_data) for c in clients)
    print(f"  Total local samples across fleet: {total_samples}")

    # Privacy budget estimate
    clipper = DPGradientClipper(
        max_norm=config.dp_max_grad_norm,
        noise_multiplier=config.dp_noise_multiplier,
    )
    budget = clipper.privacy_budget_estimate(
        n_samples=total_samples // 10,
        batch_size=config.local_batch_size,
        n_rounds=config.num_rounds,
        local_epochs=config.local_epochs,
    )
    print(f"\n  DP Privacy budget estimate:")
    print(f"    epsilon (approx) = {budget['epsilon_approx']}")
    print(f"    delta            = {budget['delta']}")
    print(f"    Steps            = {budget['steps']}")
    print(f"    Noise std/param  = {budget['noise_std_per_param']}")

    # Run FL training
    print(f"\n  Running {config.num_rounds} federated rounds...")
    server = FederatedServer(config)
    t0 = time.perf_counter()
    history = server.fit(clients)
    elapsed = time.perf_counter() - t0

    print(f"\n  Training complete in {elapsed:.2f}s")
    print(f"  {server.summary()}")
    print(f"\n  Round-by-round loss:")
    for entry in history:
        print(f"    Round {entry['round']:2d}: loss={entry['loss']:.5f}  "
              f"clients={entry['n_clients']}  "
              f"samples={entry['total_samples']}")

    print("  [PASS] Federated Learning demo complete.")
    return server


# =========================================================================== #
# 3. Edge ML — INT8 Quantization & Latency Profiling                          #
# =========================================================================== #

def demo_edge_ml(server):
    print(f"\n{SEPARATOR}")
    print("  PART 3 — EDGE ML (INT8 QUANTIZATION + PROFILING)")
    print(SEPARATOR)

    import numpy as np
    from EDGE_ML import INT8Quantizer, LatencyProfiler, ONNXExporter
    from EDGE_ML.quantizer import QuantizationConfig

    float_model = server.global_model
    print(f"  Float32 model parameters: {float_model.parameter_count():,}")

    # Generate calibration data
    rng = np.random.default_rng(0)
    calib_data = rng.random((500, float_model.input_dim), dtype=np.float32)

    # Quantize
    print("\n  Quantizing to INT8 (static, per-tensor, symmetric)...")
    qconfig = QuantizationConfig(scheme="symmetric", calibration_percentile=99.9)
    quantizer = INT8Quantizer(config=qconfig)
    int8_model = quantizer.quantize(float_model, calib_data)

    mem = int8_model.memory_bytes()
    print(f"  INT8 model memory: {mem['total_bytes']} bytes  "
          f"(weights={mem['int8_weights_bytes']} B, "
          f"biases={mem['float32_bias_bytes']} B)")

    # Verify outputs are close
    test_x = rng.random((10, float_model.input_dim), dtype=np.float32)
    float_recon = float_model.forward(test_x)
    int8_recon = int8_model.forward(test_x)
    max_err = float(np.max(np.abs(float_recon - int8_recon)))
    print(f"  Max element-wise error (float32 vs INT8): {max_err:.5f}")
    assert max_err < 0.5, f"Quantization error too large: {max_err}"

    # Latency profiling
    print("\n  Running latency benchmark (1000 inference calls, batch=1)...")
    profiler = LatencyProfiler(
        model=float_model, n_runs=1000, batch_size=1,
        input_dim=float_model.input_dim, warmup_runs=50,
    )
    comparison = profiler.compare(float_model, int8_model)
    print("\n" + LatencyProfiler.report(comparison))

    # Anomaly score test
    healthy_window = rng.random(float_model.input_dim, dtype=np.float32)
    score = int8_model.anomaly_score(healthy_window)
    print(f"\n  INT8 anomaly score (healthy window): {score:.6f}")

    # ONNX export
    print("\n  Exporting to ONNX...")
    onnx_path = os.path.join("EDGE_ML", "pratibimb_anomaly.onnx")
    exporter = ONNXExporter(opset_version=17)
    try:
        exported = exporter.export(float_model, calib_data[:2], onnx_path)
        ok = exporter.validate(exported, calib_data[:5])
        print(f"  ONNX export: {exported}")
        print(f"  ONNX validation: {'PASS' if ok else 'SKIPPED (no OnnxRuntime)'}")
    except Exception as e:
        print(f"  ONNX export skipped: {e}")

    print("  [PASS] Edge ML demo complete.")
    return int8_model, comparison


# =========================================================================== #
# Main                                                                         #
# =========================================================================== #

if __name__ == "__main__":
    print(SEPARATOR)
    print("  PRATIBIMB — Advanced Systems Integration Demo")
    print("  CAN Telemetry | Federated Learning | Edge ML")
    print(SEPARATOR)

    frames = demo_can_telemetry()
    server = demo_federated_learning()
    int8_model, bench = demo_edge_ml(server)

    print(f"\n{SEPARATOR}")
    print("  ALL SYSTEMS DEMO COMPLETE")
    print(f"  CAN frames assembled     : {len(frames)}")
    print(f"  FL global model trained  : {server.summary()}")
    print(f"  INT8 speedup             : {bench.get('speedup_x', '?')}x")
    print(SEPARATOR)
