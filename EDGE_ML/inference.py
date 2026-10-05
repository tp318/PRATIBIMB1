"""
QuantizedMLPAutoencoder — INT8 forward-pass inference engine for edge devices.

Implements the MLP autoencoder forward pass entirely in INT8 arithmetic
(stored as numpy int8, computed in int32 accumulators to avoid overflow)
with activation rescaling at each layer boundary.

This is the model that actually runs on the avionics computer during a sortie.
Its weights are loaded from a QuantizationConfig checkpoint on boot and the
forward pass is called at 100 Hz by the health monitoring pipeline.

Integration with AeroTwin health pipeline:
    The anomaly_score() method returns a scalar reconstruction error directly
    compatible with the Mahalanobis gate threshold used by the statistical
    anomaly detector, so the quantized model is a drop-in replacement.

Memory footprint (default model_dim=9, hidden_dim=32):
    W1: 9*32  = 288 bytes (int8)
    W2: 32*16 = 512 bytes
    W3: 16*32 = 512 bytes
    W4: 32*9  = 288 bytes
    biases:    ~320 bytes (float32 kept at full precision for bias)
    Total:     ~2.4 KB — well within 4 KB L1 cache on Cortex-A53
"""

import numpy as np
from typing import Dict


class QuantizedMLPAutoencoder:
    """
    INT8 quantized MLP autoencoder inference engine.

    Parameters
    ----------
    int8_weights : dict
        Weight tensors quantized to int8 (W1, W2, W3, W4).
    weight_scales : dict
        Per-weight float scale factors.
    activation_scales : dict
        Per-layer activation scale factors from calibration.
    input_dim, hidden_dim : int
        Network dimensions (must match quantized weights).
    """

    def __init__(
        self,
        int8_weights: Dict[str, np.ndarray],
        weight_scales: Dict[str, float],
        activation_scales: Dict[str, float],
        input_dim: int = 9,
        hidden_dim: int = 32,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._w = int8_weights          # int8 weight tensors
        self._ws = weight_scales        # float scale per weight tensor
        self._as = activation_scales    # float scale per activation layer

        # Biases are stored as float32 in the int8_weights dict
        self._b1 = int8_weights.get("b1", np.zeros(hidden_dim, np.float32)).astype(np.float32)
        self._b2 = int8_weights.get("b2", np.zeros(hidden_dim // 2, np.float32)).astype(np.float32)
        self._b3 = int8_weights.get("b3", np.zeros(hidden_dim, np.float32)).astype(np.float32)
        self._b4 = int8_weights.get("b4", np.zeros(input_dim, np.float32)).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        INT8 quantized forward pass.

        Parameters
        ----------
        x : np.ndarray
            Shape (batch, input_dim), float32 in [0, 1].

        Returns
        -------
        np.ndarray
            Reconstructed output, float32.
        """
        # Quantize input
        x_scale = self._as.get("input", 1.0 / 127.0)
        x_q = np.clip(np.round(x / x_scale), -128, 127).astype(np.int8)

        # Layer 1: int8 matmul in int32 accumulator, then rescale + ReLU
        h1 = self._int8_linear(x_q, "W1", self._b1,
                               in_scale=x_scale,
                               out_scale=self._as.get("h1", 1.0 / 127.0))
        h1 = np.maximum(0.0, h1)       # ReLU in float
        h1_q = np.clip(np.round(h1 / self._as.get("h1", 1.0 / 127.0)),
                       -128, 127).astype(np.int8)

        # Layer 2
        h2 = self._int8_linear(h1_q, "W2", self._b2,
                               in_scale=self._as.get("h1", 1.0 / 127.0),
                               out_scale=self._as.get("h2", 1.0 / 127.0))
        h2 = np.maximum(0.0, h2)
        h2_q = np.clip(np.round(h2 / self._as.get("h2", 1.0 / 127.0)),
                       -128, 127).astype(np.int8)

        # Layer 3
        h3 = self._int8_linear(h2_q, "W3", self._b3,
                               in_scale=self._as.get("h2", 1.0 / 127.0),
                               out_scale=self._as.get("h3", 1.0 / 127.0))
        h3 = np.maximum(0.0, h3)
        h3_q = np.clip(np.round(h3 / self._as.get("h3", 1.0 / 127.0)),
                       -128, 127).astype(np.int8)

        # Layer 4 — output in float32 (no output quantization)
        recon = self._int8_linear_float_out(h3_q, "W4", self._b4,
                                            in_scale=self._as.get("h3", 1.0 / 127.0))
        return recon

    def anomaly_score(self, sensor_window: np.ndarray) -> float:
        """
        Compute reconstruction MSE anomaly score for one sensor observation.
        Drop-in replacement for the float32 autoencoder.anomaly_score() method.

        Parameters
        ----------
        sensor_window : np.ndarray
            Shape (input_dim,), normalized sensor values.

        Returns
        -------
        float
            Reconstruction MSE — higher means more anomalous.
        """
        x = sensor_window.reshape(1, -1).astype(np.float32)
        recon = self.forward(x)
        return float(np.mean((sensor_window - recon.squeeze()) ** 2))

    # ------------------------------------------------------------------ #
    # Memory / performance info                                            #
    # ------------------------------------------------------------------ #

    def memory_bytes(self) -> Dict[str, int]:
        int8_bytes = sum(
            v.size for k, v in self._w.items() if k.startswith("W")
        )
        bias_bytes = sum([self._b1.size, self._b2.size,
                          self._b3.size, self._b4.size]) * 4
        return {
            "int8_weights_bytes": int8_bytes,
            "float32_bias_bytes": bias_bytes,
            "total_bytes": int8_bytes + bias_bytes,
        }

    # ------------------------------------------------------------------ #
    # Internal INT8 helpers                                                #
    # ------------------------------------------------------------------ #

    def _int8_linear(
        self, x_q: np.ndarray, w_key: str,
        bias: np.ndarray, in_scale: float, out_scale: float,
    ) -> np.ndarray:
        """
        Quantized linear layer: y = x_q @ W_q (int32) * (in_scale * w_scale)
        + bias, then rescaled to float.
        """
        W_q = self._w[w_key]
        # Compute in int32 to avoid int8 overflow
        acc = x_q.astype(np.int32) @ W_q.astype(np.int32)
        w_scale = self._ws.get(w_key, 1.0 / 127.0)
        return acc.astype(np.float32) * (in_scale * w_scale) + bias

    def _int8_linear_float_out(
        self, x_q: np.ndarray, w_key: str, bias: np.ndarray, in_scale: float,
    ) -> np.ndarray:
        W_q = self._w[w_key]
        acc = x_q.astype(np.int32) @ W_q.astype(np.int32)
        w_scale = self._ws.get(w_key, 1.0 / 127.0)
        return acc.astype(np.float32) * (in_scale * w_scale) + bias
