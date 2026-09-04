"""Capability and budget enforcement independent from model output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    tool: str
    resources: frozenset[str]
    allow_write: bool = False


@dataclass
class Budget:
    remaining_calls: int

    def consume(self) -> None:
        if self.remaining_calls <= 0:
            raise RuntimeError("tool-call budget exhausted")
        self.remaining_calls -= 1


def authorize(capability: Capability, tool: str, resource: str, write: bool, budget: Budget) -> None:
    if tool != capability.tool:
        raise PermissionError("tool is outside capability")
    if resource not in capability.resources:
        raise PermissionError("resource is outside capability")
    if write and not capability.allow_write:
        raise PermissionError("write requires a write capability")
    budget.consume()
