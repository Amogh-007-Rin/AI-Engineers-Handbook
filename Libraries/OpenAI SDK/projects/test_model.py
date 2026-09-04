import unittest
from model import validate_request, validate_tool_call


class OpenAISDKProjectTests(unittest.TestCase):
    def test_bounded_request(self):
        self.assertTrue(validate_request({"model": "model-v1", "input": "hi", "timeout_s": 10, "max_cost_usd": .01}))
        with self.assertRaises(ValueError): validate_request({"model": "model-v1", "input": "hi"})

    def test_tool_allowlist_and_arguments(self):
        self.assertTrue(validate_tool_call({"name": "search", "arguments": {"q": "x"}}, {"search"}))
        with self.assertRaises(ValueError): validate_tool_call({"name": "shell", "arguments": {}}, {"search"})
        with self.assertRaises(ValueError): validate_tool_call({"name": "search", "arguments": {"__class__": 1}}, {"search"})


if __name__ == "__main__": unittest.main()
