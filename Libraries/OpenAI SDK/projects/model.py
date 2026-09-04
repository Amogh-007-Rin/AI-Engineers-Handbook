"""Offline OpenAI-style response and tool-call contract validators."""


def validate_tool_call(call, allowed):
    if call.get("name") not in allowed:
        raise ValueError("tool is not allowlisted")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict) or any(key.startswith("__") for key in arguments):
        raise ValueError("tool arguments are invalid")
    return True


def validate_request(request):
    if not request.get("model") or request.get("timeout_s", 0) <= 0 or request.get("max_cost_usd", 0) <= 0:
        raise ValueError("model, timeout, and cost budget are required")
    if request.get("input") is None:
        raise ValueError("input is required")
    return True
