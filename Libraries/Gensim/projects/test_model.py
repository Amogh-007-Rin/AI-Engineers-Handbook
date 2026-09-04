import tempfile
import unittest

from model import load, save, train, transform

DOCS = ["red fox jumps", "blue bird flies", "red bird rests"]


class GensimProjectTests(unittest.TestCase):
    def test_unknown_tokens_do_not_mutate_dictionary(self):
        dictionary, model = train(DOCS)
        size = len(dictionary)
        result = transform(dictionary, model, "unseen red")
        self.assertEqual(len(dictionary), size)
        self.assertEqual([token for token, _ in result], ["red"])

    def test_round_trip_preserves_vector(self):
        dictionary, model = train(DOCS)
        expected = transform(dictionary, model, "red bird")
        with tempfile.TemporaryDirectory() as directory:
            save(dictionary, model, directory)
            restored_dictionary, restored_model = load(directory)
            actual = transform(restored_dictionary, restored_model, "red bird")
        self.assertEqual(actual, expected)

    def test_empty_vocabulary_fails(self):
        with self.assertRaises(ValueError): train(["!!!"])


if __name__ == "__main__": unittest.main()
