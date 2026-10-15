import time
from src.result_logger import log_episode
import vizdoom as vzd
import json
from pathlib import Path


from src.decision_core import State, Decision, execute_decision
from src.random_policy import RandomLegalPolicy


BUTTON_MAP = {
    vzd.Button.MOVE_LEFT: "left",
    vzd.Button.MOVE_RIGHT: "right",
    vzd.Button.ATTACK: "attack",
    
}
def load_policy_config() -> dict:
    config_path = Path("configs/policy.json")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)



class BaselinePolicy:
    """Configurable visual heuristic; not a trained combat agent."""

    def __init__(
        self,
        variation_threshold: float = 55,
        darkness_threshold: float = 25,
        attack_fallback: bool = True,
    ):
        self.variation_threshold = variation_threshold
        self.darkness_threshold = darkness_threshold
        self.attack_fallback = attack_fallback

    def decide(self, state: State) -> Decision:

        # Average brightness and variation are simple visual features.
        screen = state.observation.get("screen")
        if screen is None:
            raise ValueError("Policy did not receive screen pixels.")

        _, height, width = screen.shape
        region = screen[
        :,
        height // 4 : 3 * height // 4,
        width // 4 : 3 * width // 4,
        ]

        brightness = float(region.mean())
        variation = float(region.std())

        # Conservative, deterministic rule for an initial experiment.
        if variation > self.variation_threshold and "attack" in state.legal_actions:
            action = "attack"
            reason = "High central visual variation"
        elif brightness < self.darkness_threshold and "left" in state.legal_actions:
            action = "left"
            reason = "Dark central region"
        elif self.attack_fallback and "attack" in state.legal_actions:
            action = "attack"
            reason = "Default visual heuristic"
        else:
            action = state.legal_actions[0]
            reason = "Fallback to first legal action"

        return Decision(action, 0.5, reason)



def run_episode(
    max_steps: int = 200,
    seed: int | None = None,
    policy_name: str = "heuristic",
) -> None:
    game = vzd.DoomGame()
    game.load_config(vzd.scenarios_path + "/basic.cfg")

    if seed is not None:
        game.set_seed(seed)

    game.set_window_visible(False)
    game.set_episode_timeout(max_steps)
    game.init()

    if policy_name == "heuristic":
        config = load_policy_config()
        policy = BaselinePolicy(
            variation_threshold=config["variation_threshold"],
            darkness_threshold=config["darkness_threshold"],
            attack_fallback=config["attack_fallback"],
        )
    elif policy_name == "random":
        policy = RandomLegalPolicy(seed=seed)
    else:
        raise ValueError(
            f"Unknown policy: {policy_name!r}. "
            "Choose 'heuristic' or 'random'."
        )

    total_reward = 0.0
    decision_times = []
    action_cycle_times = []
    steps = 0
    last_observed_tic = None

    try:
        while not game.is_episode_finished() and steps < max_steps:
            observation = game.get_state()
            if observation is None:
                print("Stopping: get_state() returned None")
                break
            last_observed_tic = observation.tic
            available = game.get_available_buttons()
            legal_actions = tuple(BUTTON_MAP[b] for b in available)
            state = State(
                observation={
                    "step": steps,
                    "screen": observation.screen_buffer.copy(),
                    "screen_shape": observation.screen_buffer.shape,
                },
                legal_actions=legal_actions,
            )
            cycle_start = time.perf_counter()
            start = time.perf_counter()
            decision = execute_decision(policy, state)
            decision_times.append((time.perf_counter() - start) * 1000)

            button = next(
                b for b, name in BUTTON_MAP.items() if name == decision.action
            )
            action = [button == available_button for available_button in available]
            reward = game.make_action(action, 1)
            action_cycle_times.append((time.perf_counter() - cycle_start) * 1000)
            total_reward += reward
            steps += 1

        print("Episode steps:", steps)
        print("Last observed tic:", last_observed_tic)
        print("Player dead:", game.is_player_dead())
        print("Episode finished:", game.is_episode_finished())
        print("Total reward:", round(total_reward, 3))
        mean_ms = (
            sum(decision_times) / len(decision_times) if decision_times else 0.0
        )
        mean_action_cycle_ms = (
            sum(action_cycle_times) / len(action_cycle_times)
            if action_cycle_times
            else 0.0
        )
        log_episode(
            environment="ViZDoom basic.cfg",
            policy=(
                "Visual heuristic baseline"
                if policy_name == "heuristic"
                else "Random legal baseline"
            ),
            requested_seed=seed,
            steps=steps,
            last_observed_tic=last_observed_tic,
            total_reward=round(total_reward, 3),
            player_dead=game.is_player_dead(),
            episode_finished=game.is_episode_finished(),
            mean_decision_latency_ms=round(mean_ms, 6),
            mean_action_cycle_ms=round(mean_action_cycle_ms, 6),
            policy_decisions_per_second=(
                round(1000 / mean_ms, 2) if mean_ms else 0.0
            ),
        )
    finally:
        game.close()


if __name__ == "__main__":
    for episode in range(1, 6):
        print(f"\n===== Episode {episode}/5 =====")
        run_episode()