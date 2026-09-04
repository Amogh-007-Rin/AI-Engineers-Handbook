import unittest
from model import validate_experiment


class SB3ProjectTests(unittest.TestCase):
    def test_valid_experiment(self):
        self.assertTrue(validate_experiment({"action_space": "discrete", "algorithm": "DQN", "seeds": [1, 2, 3], "evaluation_episodes": 10}))

    def test_algorithm_and_evaluation_gates(self):
        with self.assertRaises(ValueError): validate_experiment({"action_space": "discrete", "algorithm": "SAC", "seeds": [1, 2, 3], "evaluation_episodes": 10})
        with self.assertRaises(ValueError): validate_experiment({"action_space": "continuous", "algorithm": "PPO", "seeds": [1], "evaluation_episodes": 10})
        with self.assertRaises(ValueError): validate_experiment({"action_space": "continuous", "algorithm": "PPO", "seeds": [1, 2, 3], "evaluation_episodes": 10, "normalized": True})


if __name__ == "__main__": unittest.main()
