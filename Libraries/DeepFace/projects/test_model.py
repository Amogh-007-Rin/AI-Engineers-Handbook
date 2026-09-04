import unittest
from model import validate_governance, verify


class DeepFaceProjectTests(unittest.TestCase):
    def test_threshold_boundary(self):
        self.assertTrue(verify(.4, .4)["verified"]); self.assertFalse(verify(.41, .4)["verified"])
        with self.assertRaises(ValueError): verify(float("nan"), .4)

    def test_governance_gate(self):
        policy = {"consent": True, "purpose": "account recovery", "retention_days": 30,
                  "deletion": "verified request", "encryption": "managed keys", "human_review": True}
        self.assertTrue(validate_governance(policy))
        with self.assertRaises(ValueError): validate_governance({**policy, "consent": False})


if __name__ == "__main__": unittest.main()
