import tempfile
import unittest

import tensorflow as tf

from model import train


class TensorFlowProjectTests(unittest.TestCase):
    def test_training_and_dynamic_batches(self):
        model, losses = train()
        self.assertLess(losses[-1], losses[0] * 1e-3)
        for batch in (1, 5):
            self.assertEqual(model.serve(tf.zeros([batch, 1]))["predictions"].shape, (batch, 1))

    def test_saved_model_round_trip(self):
        model, _ = train()
        sample = tf.constant([[3.0]], tf.float32)
        expected = model.serve(sample)["predictions"]
        with tempfile.TemporaryDirectory() as directory:
            tf.saved_model.save(model, directory, signatures={"serving_default": model.serve})
            restored = tf.saved_model.load(directory)
            actual = restored.signatures["serving_default"](features=sample)["predictions"]
        tf.debugging.assert_near(actual, expected)


if __name__ == "__main__":
    unittest.main()
