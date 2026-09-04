"""Risk-register and release-gate validation for an AI system."""


def validate_risk_register(register):
    if not register:
        raise ValueError("at least one risk required")
    required = {"harm", "affected_group", "severity", "likelihood", "mitigation", "owner", "monitor", "rollback"}
    for index, risk in enumerate(register):
        missing = required - set(risk)
        if missing:
            raise ValueError(f"risk {index} missing: {', '.join(sorted(missing))}")
        if risk["severity"] not in {"low", "medium", "high", "critical"}:
            raise ValueError("invalid severity")
        if not risk["owner"] or not risk["monitor"] or not risk["rollback"]:
            raise ValueError("operational ownership and response required")
    return True


def release_allowed(register):
    validate_risk_register(register)
    return not any(risk["severity"] == "critical" and not risk.get("residual_accepted") for risk in register)
