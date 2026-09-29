import json
import math
import os
import random
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SCREEN_DIRECTORY = Path(os.environ["MINESTUDIO_SCREEN_DIRECTORY"])
POLICIES = (
    "vanilla",
    "upper_lora_r8_step800",
    "upper_lora_r8_step1600",
    "upper_lora_r8_step3200",
    "upper_lora_r8_step4800",
    "upper_lora_r8_step6400",
    "upper_lora_r32_step6400",
    "recurrent_upper_lora_step6400",
)
SPLITS = ("normal", "hazard")
BOOTSTRAP_SAMPLES = 20000


def experiment_directory(split: str, policy: str) -> Path:
    return PROJECT_DIRECTORY / "output" / "stone_acquisition" / f"screen_{split}_{policy}_v1"


def read_rows(split: str, policy: str) -> list[dict]:
    path = experiment_directory(split, policy) / "episodes.jsonl"
    return sorted((json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line), key=lambda row: row["episode_index"])


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"baseline_only": left_only, "candidate_only": right_only, "discordant": discordant, "exact_p_value": p_value}


def paired_interval(baseline: list[bool], candidate: list[bool], seed: int) -> list[float]:
    differences = [float(candidate[index]) - float(baseline[index]) for index in range(len(baseline))]
    generator = random.Random(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        samples.append(sum(differences[generator.randrange(len(differences))] for _ in differences) / len(differences))
    samples.sort()
    return [samples[round(0.025 * (len(samples) - 1))], samples[round(0.975 * (len(samples) - 1))]]


def main() -> None:
    all_rows = {(split, policy): read_rows(split, policy) for split in SPLITS for policy in POLICIES}
    manifest_ids = {split: [row["result_id"] for row in all_rows[(split, "vanilla")]] for split in SPLITS}
    for split in SPLITS:
        for policy in POLICIES:
            if [row["result_id"] for row in all_rows[(split, policy)]] != manifest_ids[split]:
                raise RuntimeError(f"Unpaired episode set for {split}/{policy}")
    baseline = [row["success"] for split in SPLITS for row in all_rows[(split, "vanilla")]]
    policy_rows = []
    for policy_index, policy in enumerate(POLICIES):
        normal = all_rows[("normal", policy)]
        hazard = all_rows[("hazard", policy)]
        combined = normal + hazard
        candidate = [row["success"] for row in combined]
        water = [row for row in combined if row["entered_water"]]
        policy_rows.append({
            "policy": policy,
            "normal_success_rate": sum(row["success"] for row in normal) / len(normal),
            "hazard_success_rate": sum(row["success"] for row in hazard) / len(hazard),
            "combined_success_rate": sum(candidate) / len(candidate),
            "water_entry_rate": len(water) / len(combined),
            "success_given_entered_water": sum(row["success"] for row in water) / len(water) if water else None,
            "successful_water_exit_rate": sum(row["water_exit_success"] for row in water) / len(water) if water else None,
            "stone_reacquisition_after_water_rate": sum(row["stone_reacquired_after_water"] for row in water) / len(water) if water else None,
            "timeout_after_water_rate": sum(row["timeout_after_water"] for row in water) / len(water) if water else None,
            "death_rate": sum(row["died"] for row in combined) / len(combined),
            "mean_steps": sum(row["steps"] for row in combined) / len(combined),
            "versus_vanilla": {
                "success_rate_difference": sum(candidate) / len(candidate) - sum(baseline) / len(baseline),
                "paired_bootstrap_95_ci": paired_interval(baseline, candidate, 20260922 + policy_index),
                "mcnemar": exact_mcnemar(baseline, candidate),
            },
        })
    learned_ranking = sorted((row for row in policy_rows if row["policy"] != "vanilla"), key=lambda row: (row["combined_success_rate"], row["hazard_success_rate"], row["normal_success_rate"]), reverse=True)
    report = {
        "screen_directory": str(SCREEN_DIRECTORY),
        "episodes_per_policy": {"normal": 40, "hazard": 60, "combined": 100},
        "paired": True,
        "system2_protocol": "frozen event-driven Gemini task bank; no live planner calls",
        "policies": policy_rows,
        "learned_ranking": [row["policy"] for row in learned_ranking],
        "selected_top_two": [row["policy"] for row in learned_ranking[:2]],
    }
    output_path = SCREEN_DIRECTORY / "behavioral_screen_report.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
