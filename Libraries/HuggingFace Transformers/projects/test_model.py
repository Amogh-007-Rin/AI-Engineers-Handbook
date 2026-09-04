import unittest
from model import config, validate_batch


class TransformersProjectTests(unittest.TestCase):
    def test_offline_config(self):
        cfg = config(); self.assertEqual(cfg.hidden_size, 32); self.assertEqual(cfg.num_labels, 3)

    def test_batch_contract(self):
        self.assertTrue(validate_batch({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}))
        with self.assertRaises(ValueError): validate_batch({"input_ids": [[1]], "attention_mask": [[1, 0]]})
        with self.assertRaises(ValueError): validate_batch({"input_ids": [[1]], "attention_mask": [[2]]})


if __name__ == "__main__": unittest.main()
