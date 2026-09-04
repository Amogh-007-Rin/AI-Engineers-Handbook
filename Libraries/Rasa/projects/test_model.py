import unittest
from model import validate_domain


GOOD = {"intents": ["greet", "nlu_fallback"], "responses": {"utter_greet": ["hello"], "utter_fallback": ["please rephrase"]},
        "actions": [{"name": "create_ticket", "side_effects": True, "authorized": True, "idempotent": True}]}


class RasaProjectTests(unittest.TestCase):
    def test_valid_domain(self): self.assertTrue(validate_domain(GOOD))

    def test_fallback_and_action_gates(self):
        with self.assertRaises(ValueError): validate_domain({**GOOD, "intents": ["greet"]})
        with self.assertRaises(ValueError): validate_domain({**GOOD, "actions": [{"name": "pay", "side_effects": True}]})


if __name__ == "__main__": unittest.main()
