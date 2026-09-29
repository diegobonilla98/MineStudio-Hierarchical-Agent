import json
import math
import os
import random
from collections import Counter
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = Path(os.environ["MINESTUDIO_SYSTEM0_ABLATION_DIRECTORY"])
SPLITS = ("normal", "hazard")
BOOTSTRAP_SAMPLES = 20000


def experiment_directory(split: str, variant: str) -> Path:
    if variant == "old_system0_router":
        return PROJECT_DIRECTORY / "output" / "stone_acquisition" / f"screen_{split}_upper_lora_r32_step6400_v1"
    return PROJECT_DIRECTORY / "output" / "stone_acquisition" / f"system0_ablation_{split}_repaired_v1"


def read_rows(split: str, variant: str) -> list[dict]:
    path = experiment_directory(split, variant) / "episodes.jsonl"
    return sorted((json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line), key=lambda row: row["episode_index"])


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    discordant = left_only + right_only
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)) / (2 ** discordant) if discordant else 0.5
    return {
        "old_only": left_only,
        "repaired_only": right_only,
        "discordant": discordant,
        "exact_p_value": min(1.0, 2.0 * tail),
    }


def paired_interval(baseline: list[bool], candidate: list[bool]) -> list[float]:
    differences = [float(candidate[index]) - float(baseline[index]) for index in range(len(baseline))]
    generator = random.Random(20260925)
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
        "pickup_tail_failures": sum("STONE_BROKEN_NOT_COLLECTED" in row["labels"] for row in failures),
        "definite_pickup_tail_failures": sum("STONE_BROKEN_NOT_COLLECTED" in row["labels"] and row["stone_mined_delta"] >= 3 and row["cobblestone_delta"] < 3 for row in failures),
        "mean_router_transitions": sum(len(row["task_transitions"]) for row in rows) / len(rows),
        "mean_router_transitions_success": sum(len(row["task_transitions"]) for row in rows if row["success"]) / max(1, sum(row["success"] for row in rows)),
        "mean_router_transitions_failure": sum(len(row["task_transitions"]) for row in failures) / max(1, len(failures)),
        "collect_attempts": len(collect_records),
        "collect_failures": sum(not record.get("success", False) for record in collect_records),
        "collect_needs_system1": sum(record.get("status") == "NEEDS_SYSTEM1" for record in collect_records),
        "primary_failure_categories": dict(Counter(row["primary_label"] for row in failures)),
        "overlapping_failure_categories": dict(Counter(label for row in failures for label in row["labels"])),
    }


def main() -> None:
    rows = {(split, variant): read_rows(split, variant) for split in SPLITS for variant in ("old_system0_router", "repaired_system0_router")}
    for split in SPLITS:
        old_ids = [(row["result_id"], row["seed"]) for row in rows[(split, "old_system0_router")]]
        repaired_ids = [(row["result_id"], row["seed"]) for row in rows[(split, "repaired_system0_router")]]
        if old_ids != repaired_ids:
            raise RuntimeError(f"Unpaired episode set for {split}")
    report = {"paired": True, "checkpoint": "upper_lora_r32_step6400", "splits": {}, "combined": {}}
    for split in SPLITS:
        old_success = [row["success"] for row in rows[(split, "old_system0_router")]]
        repaired_success = [row["success"] for row in rows[(split, "repaired_system0_router")]]
        report["splits"][split] = {
            "old_system0_router": metrics(rows[(split, "old_system0_router")]),
            "repaired_system0_router": metrics(rows[(split, "repaired_system0_router")]),
            "success_rate_difference": sum(repaired_success) / len(repaired_success) - sum(old_success) / len(old_success),
            "paired_bootstrap_95_ci": paired_interval(old_success, repaired_success),
            "mcnemar": exact_mcnemar(old_success, repaired_success),
        }
    old = [row for split in SPLITS for row in rows[(split, "old_system0_router")]]
    repaired = [row for split in SPLITS for row in rows[(split, "repaired_system0_router")]]
    old_success = [row["success"] for row in old]
    repaired_success = [row["success"] for row in repaired]
    report["combined"] = {
        "old_system0_router": metrics(old),
        "repaired_system0_router": metrics(repaired),
        "success_rate_difference": sum(repaired_success) / len(repaired_success) - sum(old_success) / len(old_success),
        "paired_bootstrap_95_ci": paired_interval(old_success, repaired_success),
        "mcnemar": exact_mcnemar(old_success, repaired_success),
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / "system0_router_ablation_report.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
