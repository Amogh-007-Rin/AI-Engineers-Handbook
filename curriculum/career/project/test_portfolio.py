import unittest
from portfolio import REQUIRED, validate_entry


def entry():
    result = {field: f"evidence for {field}" for field in REQUIRED}
    result["claims"] = [{"text": "improved recall", "evidence": "report.json#recall"}]
    return result


class PortfolioTests(unittest.TestCase):
    def test_complete_entry(self): self.assertTrue(validate_entry(entry()))

    def test_missing_and_unsupported_claims(self):
        incomplete = entry(); del incomplete["baseline"]
        with self.assertRaises(ValueError): validate_entry(incomplete)
        unsupported = entry(); unsupported["claims"] = [{"text": "best model"}]
        with self.assertRaises(ValueError): validate_entry(unsupported)


if __name__ == "__main__": unittest.main()
