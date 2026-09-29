import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SUITE_DIRECTORY = Path(os.environ["MINESTUDIO_MICRO_SUITE_DIRECTORY"])
TASKS = (
    "EXIT_WATER",
    "CLIMB_SHORE",
    "REACQUIRE_STONE",
    "ESCAPE_HOLE",
    "AVOID_DIGGING_TRAP",
    "RECOVER_CAMERA",
    "AVOID_WATER",
)


def read_rows(task: str) -> list[dict]:
    path = SUITE_DIRECTORY / task.lower() / "episodes.jsonl"
    return sorted((json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line), key=lambda row: row["episode_index"])


def correlation(first: list[dict], second: list[dict]) -> float | None:
    first_by_index = {row["episode_index"]: float(row["success"]) for row in first}
    second_by_index = {row["episode_index"]: float(row["success"]) for row in second}
    indices = sorted(set(first_by_index) & set(second_by_index))
    left = np.asarray([first_by_index[index] for index in indices], dtype=np.float64)
    right = np.asarray([second_by_index[index] for index in indices], dtype=np.float64)
    if len(indices) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def task_metrics(task: str, rows: list[dict], gemini_rows: list[dict]) -> dict:
    summary = json.loads((SUITE_DIRECTORY / task.lower() / "summary.json").read_text(encoding="utf-8"))
    exact = summary["overall"]
    labels = Counter(label for row in gemini_rows if row["task"] == task for label in row["labels"])
    return {
        "episodes": exact["episodes"],
        "successes": exact["successes"],
        "success_rate": exact["success_rate"],
        "success_wilson_95": exact["success_wilson_95"],
        "mean_completion_steps": exact["mean_completion_steps"],
        "median_completion_steps": exact["median_completion_steps"],
        "mean_episode_steps": exact["mean_episode_steps"],
        "environment_failure_modes": exact["failure_reasons"],
        "gemini_failure_modes": dict(sorted(labels.items())),
        "water_entry_rate": exact["water_entry_rate"],
        "trap_entry_rate": exact["trap_entry_rate"],
        "mean_router_transitions": exact["mean_router_transitions"],
        "failures_criticized": sum(1 for row in gemini_rows if row["task"] == task),
    }


def main() -> None:
    rows_by_task = {task: read_rows(task) for task in TASKS}
    critique_path = SUITE_DIRECTORY / "gemini_failure_critiques.jsonl"
    gemini_rows = [json.loads(line) for line in critique_path.read_text(encoding="utf-8").splitlines() if line] if critique_path.exists() else []
    metrics = {task: task_metrics(task, rows_by_task[task], gemini_rows) for task in TASKS}
    ranking = sorted(TASKS, key=lambda task: (metrics[task]["success_rate"], metrics[task]["mean_completion_steps"] if metrics[task]["mean_completion_steps"] is not None else math.inf))
    correlations = []
    for first_index, first_task in enumerate(TASKS):
        for second_task in TASKS[first_index + 1:]:
            value = correlation(rows_by_task[first_task], rows_by_task[second_task])
            if value is not None:
                correlations.append({"first": first_task, "second": second_task, "phi": value, "absolute_phi": abs(value)})
    correlations.sort(key=lambda row: row["absolute_phi"], reverse=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_directory": str(SUITE_DIRECTORY),
        "checkpoint": "Upper LoRA r32 step 6400",
        "router": "old",
        "pickup": "old",
        "training": False,
        "live_gemini_planning": False,
        "episodes": sum(len(rows) for rows in rows_by_task.values()),
        "tasks": metrics,
        "weakest_to_strongest": ranking,
        "cross_task_success_correlations": correlations,
        "correlation_note": "Phi correlations pair the same episode indices across controlled tasks. They are exploratory because task geometry differs even when seed and biome indices match.",
        "gemini_critic_complete": len(gemini_rows) == sum(1 for rows in rows_by_task.values() for row in rows if not row["success"]),
        "gemini_critiques": len(gemini_rows),
    }
    output_path = SUITE_DIRECTORY / "system1_recovery_micro_report.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
