# ONNX differential validation project

Reference CI runtime: CPython 3.12 on CPU. A clean CPython 3.14 run is also
verified with ONNX 1.22 and ONNX Runtime 1.29; older authoring releases may not
provide an equivalent wheel, so resolve the declared range in a fresh environment.

Run `python3 -m unittest -v test_model.py`. Export a trained academy model with dynamic batch axes. Validate graph/opset/shapes, compare source and ONNX Runtime outputs over fixed and generated boundary inputs, package preprocessing/labels/thresholds, and benchmark cold start plus steady-state latency/memory. Test at least two available providers and one unsupported/dynamic-shape failure. Passing requires 80/100 and no unchecked graph or unexplained numerical divergence.
