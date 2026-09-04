import unittest

from bandit import EpsilonGreedy, reward_table


class BanditTests(unittest.TestCase):
    def test_incremental_estimate(self) -> None:
        policy = EpsilonGreedy(2, 0.0)
        policy.update(0, 1)
        policy.update(0, 0)
        self.assertEqual(policy.values[0], 0.5)

    def test_seeded_behavior_is_reproducible(self) -> None:
        first = EpsilonGreedy(3, 1.0, seed=4)
        second = EpsilonGreedy(3, 1.0, seed=4)
        self.assertEqual([first.choose() for _ in range(10)], [second.choose() for _ in range(10)])
        self.assertEqual(reward_table([0.1, 0.9], 5, 3), reward_table([0.1, 0.9], 5, 3))

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            EpsilonGreedy(0, 0.1)
        with self.assertRaises(ValueError):
            reward_table([1.2], 1, 0)


if __name__ == "__main__":
    unittest.main()
