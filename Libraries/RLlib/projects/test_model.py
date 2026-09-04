import unittest
from model import validate_experiment


GOOD = {"algorithm": "PPO", "environment": "CartPole-v1", "environment_runners": 1,
        "cpus_per_runner": 1, "evaluation": {"episodes": 10, "explore": False}, "checkpoint_interval": 5}


class RLlibProjectTests(unittest.TestCase):
    def test_valid_experiment(self): self.assertTrue(validate_experiment(GOOD))

    def test_evaluation_and_resource_gates(self):
        with self.assertRaises(ValueError): validate_experiment({**GOOD, "evaluation": {"episodes": 10, "explore": True}})
        with self.assertRaises(ValueError): validate_experiment({**GOOD, "cpus_per_runner": 0})


if __name__ == "__main__": unittest.main()
