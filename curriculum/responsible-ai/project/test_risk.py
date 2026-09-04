import unittest
from risk import release_allowed, validate_risk_register


RISK = {"harm": "denied access", "affected_group": "applicants", "severity": "high",
        "likelihood": "possible", "mitigation": "human appeal", "owner": "risk lead",
        "monitor": "appeal rate", "rollback": "disable model"}


class RiskTests(unittest.TestCase):
    def test_complete_register(self): self.assertTrue(validate_risk_register([RISK])); self.assertTrue(release_allowed([RISK]))

    def test_release_and_ownership_gates(self):
        with self.assertRaises(ValueError): validate_risk_register([{**RISK, "owner": ""}])
        self.assertFalse(release_allowed([{**RISK, "severity": "critical"}]))


if __name__ == "__main__": unittest.main()
