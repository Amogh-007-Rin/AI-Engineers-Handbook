import unittest
from model import tokens, vocabulary


class NLTKProjectTests(unittest.TestCase):
    def test_adversarial_token_contract(self):
        self.assertEqual(tokens("Don't drop 3.14 or café."), ["don't", "drop", "3.14", "or", "caf"])

    def test_vocabulary_is_deterministic(self):
        self.assertEqual(vocabulary(["Red fox", "red bird"], 2), {"red": 2})

    def test_invalid_inputs_fail(self):
        with self.assertRaises(TypeError): tokens(None)
        with self.assertRaises(ValueError): vocabulary([], 0)


if __name__ == "__main__": unittest.main()
