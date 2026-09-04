"""Dependency-free Rasa domain/action contract validator."""


def validate_domain(domain):
    intents, responses = set(domain.get("intents", [])), domain.get("responses", {})
    if not intents or not responses:
        raise ValueError("intents and responses are required")
    if "nlu_fallback" not in intents or "utter_fallback" not in responses:
        raise ValueError("explicit fallback intent and response required")
    for action in domain.get("actions", []):
        if action.get("side_effects") and not (action.get("authorized") and action.get("idempotent")):
            raise ValueError("side-effecting actions require authorization and idempotency")
    return True
