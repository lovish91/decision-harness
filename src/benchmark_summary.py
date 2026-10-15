import csv
import statistics
from pathlib import Path


RESULTS_FILE = Path("results/doom_baseline_v4.csv")


def percentile(values: list[float], p: float) -> float:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def main() -> None:
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        return

    with RESULTS_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("No benchmark records found.")
        return

        print("\nResults by policy:")

    policies = sorted({row["policy"] for row in rows})

    for policy_name in policies:
        policy_rows = [
            row for row in rows
            if row["policy"] == policy_name
        ]
        policy_rewards = [
            float(row["total_reward"]) for row in policy_rows
        ]

        print(f"\n{policy_name}")
        print(f"  Episodes: {len(policy_rows)}")
        print(f"  Mean reward: {statistics.mean(policy_rewards):.3f}")
        print(f"  Min reward: {min(policy_rewards):.3f}")
        print(f"  Max reward: {max(policy_rewards):.3f}")

    rewards = [float(row["total_reward"]) for row in rows]
    policy_ms = [
        float(row["mean_decision_latency_ms"])
        for row in rows
        if row.get("mean_decision_latency_ms")
    ]
    cycle_ms = [
        float(row["mean_action_cycle_ms"])
        for row in rows
        if row.get("mean_action_cycle_ms")
    ]

    completed = [
        row["episode_finished"].strip().lower() == "true"
        for row in rows
    ]
    successes = [
        row["total_reward"] is not None
        and float(row["total_reward"]) > 0
        and row["episode_finished"].strip().lower() == "true"
        for row in rows
    ]

    print(f"Results file: {RESULTS_FILE}")
    print(f"Episodes: {len(rows)}")
    print(f"Mean reward: {statistics.mean(rewards):.3f}")
    print(
        f"Completed episodes: "
        f"{sum(completed)}/{len(completed)} "
        f"({100 * statistics.mean(completed):.1f}%)"
    )
    print(
        f"Positive-reward completed episodes: "
        f"{sum(successes)}/{len(rows)} "
        f"({100 * sum(successes) / len(rows):.1f}%)"
    )

    for label, values in [
        ("Policy latency (ms)", policy_ms),
        ("Action-cycle latency (ms)", cycle_ms),
    ]:
        if values:
            print(f"\n{label}")
            print(f"  Mean:   {statistics.mean(values):.3f}")
            print(f"  Median: {statistics.median(values):.3f}")
            print(f"  P95:    {percentile(values, 95):.3f}")
            print(f"  Min:    {min(values):.3f}")
            print(f"  Max:    {max(values):.3f}")


if __name__ == "__main__":
    main()