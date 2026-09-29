import json
import math
import statistics
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
RESULT_DIRECTORY = PROJECT_DIRECTORY / "output" / "steve_mine_log_benchmark" / "20260915T092936Z_mine_log_100"
SUMMARY_PATH = RESULT_DIRECTORY / "summary.json"
ANALYSIS_PATH = RESULT_DIRECTORY / "analysis.json"
CONFIDENCE_Z = 1.959963984540054


def wilson_interval(successes: int, total: int) -> list[float]:
    proportion = successes / total
    denominator = 1 + CONFIDENCE_Z**2 / total
    center = (proportion + CONFIDENCE_Z**2 / (2 * total)) / denominator
    radius = CONFIDENCE_Z * math.sqrt(proportion * (1 - proportion) / total + CONFIDENCE_Z**2 / (4 * total**2)) / denominator
    return [center - radius, center + radius]


def metric_summary(episodes: list[dict], selector) -> dict:
    selected = [episode for episode in episodes if selector(episode)]
    count = len(selected)
    total = len(episodes)
    return {
        "count": count,
        "rate": count / total,
        "wilson_95": wilson_interval(count, total),
    }


def mean_for(episodes: list[dict], field: str) -> float | None:
    return statistics.mean(float(episode[field]) for episode in episodes) if episodes else None


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    episodes = summary["episodes"]
    if len(episodes) != summary["total_episodes"]:
        raise RuntimeError(f"Expected {summary['total_episodes']} episodes, found {len(episodes)}")
    strict_successes = [episode for episode in episodes if episode["success"]]
    broke_without_pickup = [episode for episode in episodes if episode["broke_log"] and not episode["success"]]
    no_break = [episode for episode in episodes if not episode["broke_log"]]
    analysis = {
        "experiment_name": summary["experiment_name"],
        "prompt": summary["prompt"],
        "episodes": len(episodes),
        "strict_acquired_log": metric_summary(episodes, lambda episode: episode["success"]),
        "broke_any_log": metric_summary(episodes, lambda episode: episode["broke_log"]),
        "broke_but_did_not_acquire": metric_summary(episodes, lambda episode: episode["broke_log"] and not episode["success"]),
        "did_not_break_log": metric_summary(episodes, lambda episode: not episode["broke_log"]),
        "strict_success_steps": {
            "mean": mean_for(strict_successes, "steps"),
            "median": statistics.median(episode["steps"] for episode in strict_successes) if strict_successes else None,
        },
        "mean_horizontal_displacement": {
            "strict_success": mean_for(strict_successes, "horizontal_displacement"),
            "broke_without_pickup": mean_for(broke_without_pickup, "horizontal_displacement"),
            "no_break": mean_for(no_break, "horizontal_displacement"),
        },
        "outcome_partition": {
            "strict_success": len(strict_successes),
            "execution_or_recovery_candidate": len(broke_without_pickup),
            "conditioning_environment_or_inconclusive_candidate": len(no_break),
        },
        "interpretation_limit": "The deterministic outcome partition does not by itself distinguish conditioning failure from an unavailable target or other environmental precondition. Video/state diagnosis is required for that branch.",
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
