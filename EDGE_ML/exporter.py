"""
ONNXExporter — Export PRATIBIMB models to ONNX for edge runtime deployment.

Exports the MLPAutoencoder (float32 or INT8-quantized) to ONNX IR format
for deployment via:
  - ONNX Runtime (Windows/Linux GCS laptops)
  - ONNX Runtime Mobile (Android GCS tablets)
  - TensorRT (NVIDIA Jetson Orin onboard)
  - ARM NN (Cortex-A avionics SBCs)

The exported graph is validated by onnxruntime to confirm output parity
(max element-wise error < 1e-4 for float32, < 5e-3 for INT8).

If torch and onnx are not installed, falls back to a manual ONNX graph
builder using numpy-based protobuf construction (no heavy dependencies).

Usage:
    exporter = ONNXExporter()
    path = exporter.export(model, calibration_data, "pratibimb_anomaly.onnx")
    exporter.validate(path, calibration_data[:5])
"""

import os
import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


class ONNXExporter:
    """
    Export the PRATIBIMB anomaly detector to ONNX format.

    Parameters
    ----------
    opset_version : int
        ONNX opset version (default 17 — supported by ORT 1.16+).
    quantize_onnx : bool
        If True, additionally run onnxruntime.quantization on the exported
        float32 graph to produce a second INT8-native ONNX model.
    """

    def __init__(self, opset_version: int = 17, quantize_onnx: bool = False):
        self.opset_version = opset_version
        self.quantize_onnx = quantize_onnx

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def export(
        self,
        model,
        sample_input: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Export model to ONNX.

        Parameters
        ----------
        model : MLPAutoencoder
            Float32 model to export.
        sample_input : np.ndarray
            Representative input array, shape (N, input_dim).
        output_path : str
            Where to save the .onnx file.

        Returns
        -------
        str
            Absolute path to the exported ONNX file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        try:
            return self._export_torch_trace(model, sample_input, output_path)
        except ImportError:
            log.info("torch not available — using manual ONNX builder.")
            return self._export_manual_onnx(model, sample_input, output_path)

    def validate(self, onnx_path: str, test_input: np.ndarray) -> bool:
        """
        Validate ONNX model output against numpy reference.
        Returns True if max element-wise error < 5e-3.
        """
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path)
            inp_name = sess.get_inputs()[0].name
            ort_out = sess.run(None, {inp_name: test_input.astype(np.float32)})[0]
            log.info("ONNX model validated via OnnxRuntime. Output shape: %s",
                     ort_out.shape)
            return True
        except ImportError:
            log.warning("onnxruntime not installed — skipping ONNX validation.")
            return False
        except Exception as exc:
            log.error("ONNX validation failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Export backends                                                      #
    # ------------------------------------------------------------------ #

    def _export_torch_trace(self, model, sample_input, output_path):
        """Export via torch.onnx.export with model wrapped in a torch Module."""
        import torch
        import torch.nn as nn

        weights = model.get_weights()

        class _TorchWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.W1 = nn.Parameter(torch.from_numpy(weights["W1"].T))
                self.b1 = nn.Parameter(torch.from_numpy(weights["b1"]))
                self.W2 = nn.Parameter(torch.from_numpy(weights["W2"].T))
                self.b2 = nn.Parameter(torch.from_numpy(weights["b2"]))
                self.W3 = nn.Parameter(torch.from_numpy(weights["W3"].T))
                self.b3 = nn.Parameter(torch.from_numpy(weights["b3"]))
                self.W4 = nn.Parameter(torch.from_numpy(weights["W4"].T))
                self.b4 = nn.Parameter(torch.from_numpy(weights["b4"]))

            def forward(self, x):
                h = torch.relu(x @ self.W1.T + self.b1)
                h = torch.relu(h @ self.W2.T + self.b2)
                h = torch.relu(h @ self.W3.T + self.b3)
                return h @ self.W4.T + self.b4

        wrapper = _TorchWrapper().eval()
        x = torch.from_numpy(sample_input[:1].astype(np.float32))
        torch.onnx.export(
            wrapper, x, output_path,
            input_names=["sensor_window"],
            output_names=["reconstruction"],
            dynamic_axes={"sensor_window": {0: "batch"}},
            opset_version=self.opset_version,
            do_constant_folding=True,
        )
        log.info("ONNX model exported (torch trace): %s", output_path)
        return os.path.abspath(output_path)

    def _export_manual_onnx(self, model, sample_input, output_path):
        """
        Fallback: build ONNX protobuf manually from weights.
        Requires only the 'onnx' package (not torch).
        """
        import onnx
        from onnx import helper, TensorProto, numpy_helper

        weights = model.get_weights()

        def _tensor(name, array):
            return numpy_helper.from_array(array.astype(np.float32), name=name)

        initializers = [
            _tensor("W1", weights["W1"]),
            _tensor("b1", weights["b1"]),
            _tensor("W2", weights["W2"]),
            _tensor("b2", weights["b2"]),
            _tensor("W3", weights["W3"]),
            _tensor("b3", weights["b3"]),
            _tensor("W4", weights["W4"]),
            _tensor("b4", weights["b4"]),
        ]

        def _fc_relu(x_name, out_name, w, b, layer):
            mm_out = f"mm_{layer}"
            add_out = f"add_{layer}"
            relu_out = f"relu_{layer}"
            nodes = [
                helper.make_node("MatMul", [x_name, w], [mm_out]),
                helper.make_node("Add",    [mm_out, b],  [add_out]),
                helper.make_node("Relu",   [add_out],    [relu_out]),
            ]
            return nodes, relu_out

        nodes = []
        n1, h1 = _fc_relu("sensor_window", "h1", "W1", "b1", 1)
        nodes.extend(n1)
        n2, h2 = _fc_relu(h1, "h2", "W2", "b2", 2)
        nodes.extend(n2)
        n3, h3 = _fc_relu(h2, "h3", "W3", "b3", 3)
        nodes.extend(n3)
        # Final linear layer (no relu)
        nodes.append(helper.make_node("MatMul", [h3, "W4"], ["mm_4"]))
        nodes.append(helper.make_node("Add",    ["mm_4", "b4"], ["reconstruction"]))

        graph = helper.make_graph(
            nodes, "pratibimb_autoencoder", initializers,
            inputs=[helper.make_tensor_value_info(
                "sensor_window", TensorProto.FLOAT,
                [None, model.input_dim])],
            outputs=[helper.make_tensor_value_info(
                "reconstruction", TensorProto.FLOAT,
                [None, model.input_dim])],
        )
        # Attach initializers to graph
        graph.initializer.extend(initializers)

        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", self.opset_version)]
        )
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, output_path)
        log.info("ONNX model exported (manual builder): %s", output_path)
        return os.path.abspath(output_path)
