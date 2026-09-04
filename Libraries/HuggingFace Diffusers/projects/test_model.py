import unittest
import torch
from model import add_noise


class DiffusersProjectTests(unittest.TestCase):
    def test_shape_and_determinism(self):
        sample, noise = torch.zeros((1, 3, 4, 4)), torch.ones((1, 3, 4, 4))
        first, second = add_noise(sample, noise, 5), add_noise(sample, noise, 5)
        self.assertEqual(first.shape, sample.shape); self.assertTrue(torch.equal(first, second))

    def test_contract_failures(self):
        with self.assertRaises(ValueError): add_noise(torch.zeros((3, 4, 4)), torch.zeros((3, 4, 4)), 1)
        with self.assertRaises(ValueError): add_noise(torch.zeros((1, 3, 4, 4)), torch.zeros((1, 3, 4, 4)), 20)


if __name__ == "__main__": unittest.main()
