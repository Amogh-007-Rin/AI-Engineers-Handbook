"""Construct and execute a transparent ONNX affine graph."""

from __future__ import annotations
import numpy as np
import onnx
from onnx import TensorProto, checker, helper
import onnxruntime as ort


def build_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 1])
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [2, 1], [2.0, -1.0])
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [0.5])
    nodes = [helper.make_node("MatMul", ["x", "weight"], ["linear"]), helper.make_node("Add", ["linear", "bias"], ["y"])]
    graph = helper.make_graph(nodes, "affine", [x], [y], [weight, bias])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    checker.check_model(model)
    return model


def predict(model: onnx.ModelProto, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("expected finite float-compatible shape (batch, 2)")
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(["y"], {"x": values})[0]
