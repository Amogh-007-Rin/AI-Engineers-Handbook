# ONNX solution guidance

The reference constructs an explicit opset-18 affine graph, validates it, uses a dynamic batch dimension, constrains finite `(batch,2)` input, selects the CPU provider, and differentially checks matrix behavior. A full export must compare the source model, package pre/postprocessing, record artifact checksum and runtime provider, and separate session startup from inference benchmarks.
