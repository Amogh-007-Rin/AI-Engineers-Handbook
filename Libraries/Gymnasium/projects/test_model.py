import unittest
from model import LineWorld


class GymnasiumProjectTests(unittest.TestCase):
    def test_seeded_reset_and_contract(self):
        env = LineWorld()
        self.assertEqual(env.reset(seed=4), env.reset(seed=4))
        output = env.step(1)
        self.assertEqual(len(output), 5)
        self.assertTrue(env.observation_space.contains(output[0]))

    def test_invalid_action_fails(self):
        env = LineWorld(); env.reset()
        with self.assertRaises(ValueError): env.step(4)


if __name__ == "__main__": unittest.main()
