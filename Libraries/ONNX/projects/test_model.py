import tempfile, unittest
from pathlib import Path
import numpy as np
import onnx
from model import build_model, predict


class ONNXTests(unittest.TestCase):
    def test_differential_inference_and_dynamic_batch(self):
        x = np.array([[1, 3], [-2, 4], [0, 0]], dtype=np.float32)
        expected = x @ np.array([[2], [-1]], dtype=np.float32) + .5
        np.testing.assert_allclose(predict(build_model(), x), expected)

    def test_file_round_trip_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"; onnx.save(build_model(), path)
            restored = onnx.load(path); onnx.checker.check_model(restored)
            np.testing.assert_allclose(predict(restored, [[2, 1]]), [[3.5]])
        with self.assertRaises(ValueError): predict(build_model(), [[1, 2, 3]])


if __name__ == "__main__": unittest.main()
