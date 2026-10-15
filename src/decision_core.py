
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class State:
    observation: dict[str, Any]
    legal_actions: tuple[str, ...]


@dataclass(frozen=True)
class Decision:
    action: str
    confidence: float
    reason: str = ""


class Policy(Protocol):
    def decide(self, state: State) -> Decision:
        ...


def execute_decision(policy: Policy, state: State) -> Decision:
    if not state.legal_actions:
        raise ValueError("The environment supplied no legal actions.")

    decision = policy.decide(state)

    if decision.action not in state.legal_actions:
        raise ValueError(
            f"Illegal action: {decision.action!r}. "
            f"Legal actions: {state.legal_actions}"
        )

    if not 0.0 <= decision.confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    return decision

