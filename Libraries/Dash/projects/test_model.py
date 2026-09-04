import unittest
from model import create_app, summarize


class DashProjectTests(unittest.TestCase):
    def test_pure_callback_logic(self):
        self.assertEqual(summarize([1, 2, 3]), 2)
        with self.assertRaises(ValueError): summarize([])
        with self.assertRaises(TypeError): summarize([1, "x"])

    def test_layout_and_callback_registration(self):
        app = create_app()
        self.assertIsNotNone(app.layout)
        self.assertIn("result.children", app.callback_map)


if __name__ == "__main__": unittest.main()
