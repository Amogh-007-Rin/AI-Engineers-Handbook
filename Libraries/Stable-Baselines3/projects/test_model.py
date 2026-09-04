import importlib.util
import unittest
from model import train_smoke_agent, validate_experiment


class SB3ProjectTests(unittest.TestCase):
    def test_valid_experiment(self):
        self.assertTrue(validate_experiment({"action_space": "discrete", "algorithm": "DQN", "seeds": [1, 2, 3], "evaluation_episodes": 10}))

    def test_algorithm_and_evaluation_gates(self):
        with self.assertRaises(ValueError): validate_experiment({"action_space": "discrete", "algorithm": "SAC", "seeds": [1, 2, 3], "evaluation_episodes": 10})
        with self.assertRaises(ValueError): validate_experiment({"action_space": "continuous", "algorithm": "PPO", "seeds": [1], "evaluation_episodes": 10})
        with self.assertRaises(ValueError): validate_experiment({"action_space": "continuous", "algorithm": "PPO", "seeds": [1, 2, 3], "evaluation_episodes": 10, "normalized": True})

    @unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "academy dependency not installed")
    def test_native_seeded_training_smoke(self):
        self.assertIn(train_smoke_agent(seed=11, timesteps=32), (0, 1))
        with self.assertRaises(ValueError):
            train_smoke_agent(timesteps=8)


if __name__ == "__main__": unittest.main()
