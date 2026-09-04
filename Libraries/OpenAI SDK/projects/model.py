"""Offline OpenAI Responses API and tool-call safety contracts."""

import math


def validate_tool_call(call, allowed):
    if call.get("name") not in allowed:
        raise ValueError("tool is not allowlisted")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict) or any(key.startswith("__") for key in arguments):
        raise ValueError("tool arguments are invalid")
    return True


def validate_request(request):
    timeout = request.get("timeout_s", 0)
    budget = request.get("max_cost_usd", 0)
    output_limit = request.get("max_output_tokens", 0)
    if (
        not request.get("model")
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget <= 0
        or not isinstance(output_limit, int)
        or isinstance(output_limit, bool)
        or output_limit < 1
    ):
        raise ValueError("model, timeout, and cost budget are required")
    if request.get("input") is None:
        raise ValueError("input is required")
    return True


def run_offline_response(request, handler):
    """Exercise the real SDK against an injected, network-free HTTP handler."""
    validate_request(request)
    if not callable(handler):
        raise TypeError("handler must be callable")

    # Imports stay lazy so the dependency-free safety exercises remain usable.
    import httpx
    from openai import OpenAI

    client = OpenAI(
        api_key="offline-test-key",
        base_url="https://offline.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        response = client.responses.create(
            model=request["model"],
            input=request["input"],
            max_output_tokens=request["max_output_tokens"],
            timeout=request["timeout_s"],
        )
        return response.output_text, response.usage.total_tokens
    finally:
        client.close()
