import random

from src.decision_core import Decision, State


class RandomLegalPolicy:
    """Baseline policy that selects randomly from legal actions."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def decide(self, state: State) -> Decision:
        if not state.legal_actions:
            raise ValueError("The environment supplied no legal actions.")

        action = self.rng.choice(state.legal_actions)

        return Decision(
            action=action,
            confidence=1.0 / len(state.legal_actions),
            reason="Random selection from legal actions",
        )