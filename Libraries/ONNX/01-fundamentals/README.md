---
title: ONNX graphs export validation and portable inference
slug: onnx-foundations
level: practitioner
stage: deep-learning
estimated_hours: 12
prerequisites:
  - deep-learning-framework-parity
learning_objectives:
  - Inspect ONNX graphs tensors operators shapes and opsets
  - Validate exported behavior against a source implementation
  - Operate portable inference with schemas providers and performance evidence
formats:
  - lesson
  - project
compute: cpu
status: draft
last_verified: 2026-09-03
library: ONNX
supported_versions: 1.x
---

# ONNX graphs, export validation, and portable inference

ONNX represents a typed computation graph: inputs, outputs, initializers, nodes, attributes, shapes, and operator-set versions. It transports inference semantics, not arbitrary training code, preprocessing, labels, decision thresholds, or business contracts. Package those explicitly around the graph.

Export is a translation that can specialize control flow or shapes and replace unsupported operations. Validate with the ONNX checker, shape inference, graph inspection, and differential tests against the source model over normal, boundary, random, and adversarial inputs. Compare outputs with dtype-appropriate tolerances rather than one happy example.

Runtime providers differ in operator support, numerical behavior, precision, startup, memory, and throughput. Record provider/version/hardware; warm up; synchronize where needed; separate session creation from inference; test dynamic axes; and bound input sizes. Treat model files as supply-chain artifacts: verify provenance/checksum and load only trusted graphs in constrained environments.

## Completion criteria

- [ ] Opset, shapes, dtypes, dynamic dimensions, and signatures are recorded.
- [ ] Checker and differential tests cover boundary inputs.
- [ ] Pre/postprocessing and threshold contracts travel with the graph.
- [ ] Provider compatibility, latency, memory, and artifact security are tested.
