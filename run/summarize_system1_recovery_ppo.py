import json
import math
import os
import random
from pathlib import Path

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = Path(os.environ["MINESTUDIO_PPO_OUTPUT"])
BASELINE_MICRO_DIRECTORY = PROJECT_DIRECTORY / "output" / "system1_recovery_micro" / "20260920T075136Z"
BASELINE_NORMAL_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_acquisition" / "screen_normal_upper_lora_r32_step6400_v1"
BASELINE_HAZARD_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_acquisition" / "screen_hazard_upper_lora_r32_step6400_v1"
CANDIDATE_MICRO_DIRECTORY = RUN_DIRECTORY / "final_evaluation" / "micro_candidate"
CANDIDATE_NORMAL_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_acquisition" / f"ppo_{RUN_DIRECTORY.name}_normal_candidate"
CANDIDATE_HAZARD_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_acquisition" / f"ppo_{RUN_DIRECTORY.name}_hazard_candidate"
TASKS = (
    "EXIT_WATER",
    "CLIMB_SHORE",
    "REACQUIRE_STONE",
    "ESCAPE_HOLE",
    "AVOID_DIGGING_TRAP",
    "RECOVER_CAMERA",
    "AVOID_WATER",
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def exact_mcnemar(baseline_only: int, candidate_only: int) -> float:
    total = baseline_only + candidate_only
    if total == 0:
        return 1.0
    lower = min(baseline_only, candidate_only)
    probability = sum(math.comb(total, index) for index in range(lower + 1)) / (2**total)
    return min(1.0, 2.0 * probability)


def paired_comparison(baseline: list[dict], candidate: list[dict], success_key: str = "success") -> dict:
    baseline_by_id = {row["result_id"]: row for row in baseline}
    candidate_by_id = {row["result_id"]: row for row in candidate}
    identifiers = sorted(set(baseline_by_id) & set(candidate_by_id))
    pairs = [(bool(baseline_by_id[identifier][success_key]), bool(candidate_by_id[identifier][success_key])) for identifier in identifiers]
    baseline_successes = sum(first for first, second in pairs)
    candidate_successes = sum(second for first, second in pairs)
    baseline_only = sum(first and not second for first, second in pairs)
    candidate_only = sum(second and not first for first, second in pairs)
    differences = np.asarray([float(second) - float(first) for first, second in pairs], dtype=np.float64)
    generator = np.random.default_rng(20261001)
    bootstrap = np.asarray([generator.choice(differences, size=len(differences), replace=True).mean() for _ in range(10000)]) if len(differences) else np.asarray([0.0])
    return {
        "episodes": len(pairs),
        "baseline_successes": baseline_successes,
        "candidate_successes": candidate_successes,
        "baseline_success_rate": baseline_successes / len(pairs) if pairs else 0.0,
        "candidate_success_rate": candidate_successes / len(pairs) if pairs else 0.0,
        "difference": float(differences.mean()) if len(differences) else 0.0,
        "paired_bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "mcnemar": {
            "baseline_only": baseline_only,
            "candidate_only": candidate_only,
            "discordant": baseline_only + candidate_only,
            "exact_p_value": exact_mcnemar(baseline_only, candidate_only),
        },
    }


def combine_stone_rows(directory: Path) -> list[dict]:
    return rows(directory / "episodes.jsonl")


def qualify_result_ids(values: list[dict], split: str) -> list[dict]:
    return [{**value, "result_id": f"{split}:{value['result_id']}"} for value in values]


def main() -> None:
    training_summary = json.loads((RUN_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    micro = {}
    for task in TASKS:
        baseline = rows(BASELINE_MICRO_DIRECTORY / task.lower() / "episodes.jsonl")
        candidate = rows(CANDIDATE_MICRO_DIRECTORY / task.lower() / "episodes.jsonl")
        micro[task] = paired_comparison(baseline, candidate)
    baseline_normal = combine_stone_rows(BASELINE_NORMAL_DIRECTORY)
    baseline_hazard = combine_stone_rows(BASELINE_HAZARD_DIRECTORY)
    candidate_normal = combine_stone_rows(CANDIDATE_NORMAL_DIRECTORY)
    candidate_hazard = combine_stone_rows(CANDIDATE_HAZARD_DIRECTORY)
    stone_normal = paired_comparison(baseline_normal, candidate_normal)
    stone_hazard = paired_comparison(baseline_hazard, candidate_hazard)
    stone_combined = paired_comparison(
        qualify_result_ids(baseline_normal, "normal") + qualify_result_ids(baseline_hazard, "hazard"),
        qualify_result_ids(candidate_normal, "normal") + qualify_result_ids(candidate_hazard, "hazard"),
    )
    critic_path = CANDIDATE_MICRO_DIRECTORY / "gemini_critic_summary.json"
    critic = json.loads(critic_path.read_text(encoding="utf-8"))
    baseline_normal_rate = stone_normal["baseline_success_rate"]
    candidate_normal_rate = stone_normal["candidate_success_rate"]
    improved_weak_tasks = [task for task in ("RECOVER_CAMERA", "REACQUIRE_STONE", "AVOID_DIGGING_TRAP") if micro[task]["difference"] > 0]
    promotion = stone_combined["candidate_success_rate"] > 0.66 and len(improved_weak_tasks) >= 2 and candidate_normal_rate >= baseline_normal_rate - 0.02
    report = {
        "schema_version": 1,
        "run_id": RUN_DIRECTORY.name,
        "reference": "Upper LoRA r32 step 6400 + old router + old pickup",
        "candidate_checkpoint": training_summary["best"]["checkpoint_directory"],
        "training": training_summary,
        "micro_benchmarks": micro,
        "stone": {
            "normal": stone_normal,
            "hazard": stone_hazard,
            "combined": stone_combined,
        },
        "gemini_residual_failures": {
            "critiques": critic["critiques"],
            "label_counts": critic["label_counts"],
            "by_task": critic["by_task"],
        },
        "promotion": {
            "promoted": promotion,
            "stone_above_66_percent": stone_combined["candidate_success_rate"] > 0.66,
            "improved_priority_tasks": improved_weak_tasks,
            "normal_regression_within_2_points": candidate_normal_rate >= baseline_normal_rate - 0.02,
        },
    }
    output_path = RUN_DIRECTORY / "final_report.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
