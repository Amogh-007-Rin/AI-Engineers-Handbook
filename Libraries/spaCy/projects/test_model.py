import tempfile
import unittest

from model import annotate, build_pipeline, load, save


class SpacyProjectTests(unittest.TestCase):
    def test_offsets_and_order(self):
        result = annotate(build_pipeline(), ["Python + scikit-learn", "empty"])
        self.assertEqual([row["text"] for row in result], ["Python + scikit-learn", "empty"])
        self.assertEqual(result[0]["entities"], [("Python", "LANGUAGE", 0, 6), ("scikit-learn", "LIBRARY", 9, 21)])

    def test_disk_round_trip(self):
        nlp = build_pipeline()
        expected = annotate(nlp, ["Python"])
        with tempfile.TemporaryDirectory() as directory:
            actual = annotate(load(save(nlp, directory)), ["Python"])
        self.assertEqual(actual, expected)

    def test_single_string_is_rejected(self):
        with self.assertRaises(TypeError): annotate(build_pipeline(), "Python")


if __name__ == "__main__": unittest.main()
