import unittest
from explorer import profile


class ExplorerTests(unittest.TestCase):
    def test_profile_contract(self):
        result = profile([{"name": "a", "score": "1"}, {"name": "b", "score": "3"}])
        self.assertEqual(result["rows"], 2); self.assertEqual(result["numeric"]["score"]["mean"], 2)
        self.assertNotIn("name", result["numeric"])

    def test_missing_and_schema_failures(self):
        self.assertEqual(profile([{"x": ""}])["missing"], {"x": 1})
        with self.assertRaises(ValueError): profile([])
        with self.assertRaises(ValueError): profile([{"x": 1}, {"y": 2}])
        with self.assertRaises(ValueError): profile([{"x": "nan"}])


if __name__ == "__main__": unittest.main()
