import json
import math
import os
import random
from collections import Counter
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = Path(os.environ["MINESTUDIO_SYSTEM0_ISOLATION_DIRECTORY"])
SPLITS = ("normal", "hazard")
CONFIGURATIONS = (
    "old_router_old_pickup",
    "old_router_hardened_pickup",
    "new_router_old_pickup",
    "new_router_hardened_pickup",
)
BOOTSTRAP_SAMPLES = 20000


def experiment_directory(split: str, configuration: str) -> Path:
    names = {
        "old_router_old_pickup": f"screen_{split}_upper_lora_r32_step6400_v1",
        "old_router_hardened_pickup": f"system0_isolation_{split}_old_router_hardened_pickup_v1",
        "new_router_old_pickup": f"system0_isolation_{split}_new_router_old_pickup_v1",
        "new_router_hardened_pickup": f"system0_ablation_{split}_repaired_v1",
    }
    return PROJECT_DIRECTORY / "output" / "stone_acquisition" / names[configuration]


def read_rows(split: str, configuration: str) -> list[dict]:
    path = experiment_directory(split, configuration) / "episodes.jsonl"
    return sorted((json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line), key=lambda row: row["episode_index"])


def exact_mcnemar(champion: list[bool], candidate: list[bool]) -> dict:
    champion_only = sum(left and not right for left, right in zip(champion, candidate))
    candidate_only = sum(right and not left for left, right in zip(champion, candidate))
    discordant = champion_only + candidate_only
    tail = sum(math.comb(discordant, index) for index in range(min(champion_only, candidate_only) + 1)) / (2 ** discordant) if discordant else 0.5
    return {
        "champion_only": champion_only,
        "candidate_only": candidate_only,
        "discordant": discordant,
        "exact_p_value": min(1.0, 2.0 * tail),
    }


def paired_interval(champion: list[bool], candidate: list[bool]) -> list[float]:
    differences = [float(candidate[index]) - float(champion[index]) for index in range(len(champion))]
    generator = random.Random(20260926)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        samples.append(sum(differences[generator.randrange(len(differences))] for _ in differences) / len(differences))
    samples.sort()
    return [samples[round(0.025 * (len(samples) - 1))], samples[round(0.975 * (len(samples) - 1))]]


def metrics(rows: list[dict]) -> dict:
    failures = [row for row in rows if not row["success"]]
    water = [row for row in rows if row["entered_water"]]
    collect_records = [record for row in rows for record in row.get("option_records", []) if record.get("option") == "COLLECT_DROP"]
    return {
        "episodes": len(rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "water_entry_rate": len(water) / len(rows),
        "success_given_entered_water": sum(row["success"] for row in water) / len(water) if water else None,
        "water_exit_success_rate": sum(row["water_exit_success"] for row in water) / len(water) if water else None,
        "stone_reacquisition_after_water_rate": sum(row["stone_reacquired_after_water"] for row in water) / len(water) if water else None,
        "timeout_after_water_rate": sum(row["timeout_after_water"] for row in water) / len(water) if water else None,
        "strict_pickup_tail_failures": sum("STONE_BROKEN_NOT_COLLECTED" in row["labels"] and row["stone_mined_delta"] >= 3 and row["cobblestone_delta"] < 3 for row in failures),
        "pickup_tail_failures": sum("STONE_BROKEN_NOT_COLLECTED" in row["labels"] for row in failures),
        "collect_attempts": len(collect_records),
        "collect_failures": sum(not record.get("success", False) for record in collect_records),
        "collect_needs_system1": sum(record.get("status") == "NEEDS_SYSTEM1" for record in collect_records),
        "mean_router_transitions": sum(len(row["task_transitions"]) for row in rows) / len(rows),
        "mean_router_transitions_success": sum(len(row["task_transitions"]) for row in rows if row["success"]) / max(1, sum(row["success"] for row in rows)),
        "mean_router_transitions_failure": sum(len(row["task_transitions"]) for row in failures) / max(1, len(failures)),
        "primary_failure_categories": dict(Counter(row["primary_label"] for row in failures)),
        "overlapping_failure_categories": dict(Counter(label for row in failures for label in row["labels"])),
    }


def comparison(champion_rows: list[dict], candidate_rows: list[dict]) -> dict:
    champion = [row["success"] for row in champion_rows]
    candidate = [row["success"] for row in candidate_rows]
    return {
        "success_rate_difference": sum(candidate) / len(candidate) - sum(champion) / len(champion),
        "paired_bootstrap_95_ci": paired_interval(champion, candidate),
        "mcnemar": exact_mcnemar(champion, candidate),
    }


def main() -> None:
    rows = {(split, configuration): read_rows(split, configuration) for split in SPLITS for configuration in CONFIGURATIONS}
    for split in SPLITS:
        expected = [(row["result_id"], row["seed"]) for row in rows[(split, "old_router_old_pickup")]]
        for configuration in CONFIGURATIONS[1:]:
            observed = [(row["result_id"], row["seed"]) for row in rows[(split, configuration)]]
            if observed != expected:
                raise RuntimeError(f"Unpaired episode set for {split}/{configuration}")
    report = {
        "paired": True,
        "checkpoint": "upper_lora_r32_step6400",
        "configurations": {
            "old_router_old_pickup": {"router": "old", "pickup": "old", "role": "champion"},
            "old_router_hardened_pickup": {"router": "old", "pickup": "hardened", "role": "pickup isolation"},
            "new_router_old_pickup": {"router": "new", "pickup": "old", "role": "router isolation"},
            "new_router_hardened_pickup": {"router": "new", "pickup": "hardened", "role": "combined repaired"},
        },
        "splits": {},
        "combined": {},
    }
    for split in SPLITS:
        champion_rows = rows[(split, "old_router_old_pickup")]
        report["splits"][split] = {
            "metrics": {configuration: metrics(rows[(split, configuration)]) for configuration in CONFIGURATIONS},
            "comparisons_to_champion": {
                configuration: comparison(champion_rows, rows[(split, configuration)])
                for configuration in CONFIGURATIONS[1:]
            },
        }
    combined_rows = {
        configuration: [row for split in SPLITS for row in rows[(split, configuration)]]
        for configuration in CONFIGURATIONS
    }
    champion_rows = combined_rows["old_router_old_pickup"]
    report["combined"] = {
        "metrics": {configuration: metrics(combined_rows[configuration]) for configuration in CONFIGURATIONS},
        "comparisons_to_champion": {
            configuration: comparison(champion_rows, combined_rows[configuration])
            for configuration in CONFIGURATIONS[1:]
        },
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "system0_isolation_report.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
