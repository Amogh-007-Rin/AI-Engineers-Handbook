import unittest
from model import validate_transition


class PettingZooProjectTests(unittest.TestCase):
    def test_parallel_transition(self):
        agents = ["a", "b"]; actions = {"a": 0, "b": 1}
        transition = ({"a": 1, "b": 0}, {"a": 1.0, "b": -1.0}, {"a": False, "b": False},
                      {"a": False, "b": False}, {"a": {}, "b": {}})
        self.assertTrue(validate_transition(agents, actions, transition))

    def test_agent_key_mismatch(self):
        transition = ({"a": 1}, {"a": 0}, {"a": False}, {"a": False}, {"a": {}})
        with self.assertRaises(ValueError): validate_transition(["a", "b"], {"a": 0}, transition)


if __name__ == "__main__": unittest.main()
