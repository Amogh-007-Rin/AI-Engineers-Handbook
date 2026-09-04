import importlib.util
import json
import unittest
from model import run_offline_response, validate_request, validate_tool_call


class OpenAISDKProjectTests(unittest.TestCase):
    def test_bounded_request(self):
        self.assertTrue(validate_request({"model": "model-v1", "input": "hi", "timeout_s": 10, "max_cost_usd": .01, "max_output_tokens": 32}))
        with self.assertRaises(ValueError): validate_request({"model": "model-v1", "input": "hi"})
        with self.assertRaises(ValueError): validate_request({"model": "model-v1", "input": "hi", "timeout_s": 10, "max_cost_usd": float("nan"), "max_output_tokens": 32})

    def test_tool_allowlist_and_arguments(self):
        self.assertTrue(validate_tool_call({"name": "search", "arguments": {"q": "x"}}, {"search"}))
        with self.assertRaises(ValueError): validate_tool_call({"name": "shell", "arguments": {}}, {"search"})
        with self.assertRaises(ValueError): validate_tool_call({"name": "search", "arguments": {"__class__": 1}}, {"search"})

    @unittest.skipUnless(importlib.util.find_spec("openai"), "academy dependency not installed")
    def test_real_sdk_offline_responses_round_trip(self):
        observed = {}

        def handler(request):
            observed.update(method=request.method, path=request.url.path,
                            body=json.loads(request.content))
            return __import__("httpx").Response(200, request=request, json={
                "id": "resp_offline", "object": "response", "created_at": 0,
                "status": "completed", "model": "model-v1",
                "output": [{"id": "msg_offline", "type": "message",
                            "status": "completed", "role": "assistant",
                            "content": [{"type": "output_text", "text": "offline-ok",
                                         "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 2,
                          "total_tokens": 3,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}},
            })

        request = {"model": "model-v1", "input": "hello", "timeout_s": 2,
                   "max_cost_usd": .01, "max_output_tokens": 16}
        self.assertEqual(run_offline_response(request, handler), ("offline-ok", 3))
        self.assertEqual((observed["method"], observed["path"]), ("POST", "/v1/responses"))
        self.assertEqual(observed["body"], {"model": "model-v1", "input": "hello",
                                             "max_output_tokens": 16})


if __name__ == "__main__": unittest.main()
