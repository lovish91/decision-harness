import csv
from pathlib import Path
from datetime import datetime, timezone


RESULTS_FILE = Path("results/doom_baseline_v4.csv")

FIELDS = [
    "timestamp_utc",
    "environment",
    "policy",
    "requested_seed",
    "steps",
    "last_observed_tic",
    "total_reward",
    "player_dead",
    "episode_finished",
    "mean_decision_latency_ms",
    "mean_action_cycle_ms",
    "policy_decisions_per_second",
    
]


def log_episode(**result) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_FILE.exists()

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    }

    with RESULTS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print("Results saved to:", RESULTS_FILE.resolve())
