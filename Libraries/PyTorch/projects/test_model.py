import unittest
import torch
from model import data, round_trip, train


class PyTorchTests(unittest.TestCase):
    def test_training_and_round_trip(self):
        model = train(); x, y = data()
        with torch.no_grad():
            self.assertLess(torch.mean((model(x) - y) ** 2).item(), 1e-6)
            torch.testing.assert_close(model(x), round_trip(model)(x))

    def test_seeded_training_is_reproducible(self):
        first, second = train(), train()
        for left, right in zip(first.parameters(), second.parameters(), strict=True):
            torch.testing.assert_close(left, right)


if __name__ == "__main__": unittest.main()
