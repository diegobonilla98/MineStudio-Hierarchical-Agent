import json
import math
import os
from pathlib import Path

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_DIRECTORY / "output" / "stone_acquisition"
SUITE_DIRECTORY = Path(os.environ["MINESTUDIO_CONFIRMATION_SUITE_DIRECTORY"])
POLICIES = ("vanilla", "lora400", "lora800")
SPLITS = ("general", "stress")
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 2026091702


def load_rows(split: str, policy: str) -> dict[str, dict]:
    path = OUTPUT_ROOT / f"stage1b_{split}_{policy}_v1" / "episodes.jsonl"
    return {row["result_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def load_summary(split: str, policy: str) -> dict:
    return json.loads((OUTPUT_ROOT / f"stage1b_{split}_{policy}_v1" / "summary.json").read_text(encoding="utf-8"))


def paired_comparison(control: dict[str, dict], candidate: dict[str, dict]) -> dict:
    identifiers = sorted(set(control) & set(candidate))
    differences = np.asarray([int(candidate[key]["success"]) - int(control[key]["success"]) for key in identifiers], dtype=np.float64)
    control_only = int(np.sum(differences == -1))
    candidate_only = int(np.sum(differences == 1))
    discordant = control_only + candidate_only
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(min(control_only, candidate_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    batch_size = 1000
    for start in range(0, BOOTSTRAP_SAMPLES, batch_size):
        end = min(BOOTSTRAP_SAMPLES, start + batch_size)
        indices = generator.integers(0, len(differences), size=(end - start, len(differences)))
        bootstrap[start:end] = differences[indices].mean(axis=1)
    confidence_interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "paired_episodes": len(identifiers),
        "control_success_candidate_failure": control_only,
        "control_failure_candidate_success": candidate_only,
        "difference": float(differences.mean()),
        "paired_bootstrap_95": confidence_interval,
        "mcnemar_exact_two_sided_p": p_value,
    }


def compact(summary: dict) -> dict:
    metrics = summary["overall"]
    return {
        "episodes": metrics["episodes"],
        "success_rate": metrics["success_rate"],
        "success_wilson_95": metrics["success_wilson_95"],
        "mean_steps_to_success": metrics["mean_steps_to_success"],
        "water_entry_rate": metrics["water_entry_rate"],
        "entered_water_episodes": metrics["entered_water_episodes"],
        "success_given_entered_water": metrics["success_given_entered_water"],
        "successful_water_exit_rate": metrics["successful_water_exit_rate"],
        "stone_reacquisition_after_water_rate": metrics["stone_reacquisition_after_water_rate"],
        "timeout_after_water_rate": metrics["timeout_after_water_rate"],
        "stuck_rate": metrics["stuck_rate"],
        "trapped_in_hole_rate": metrics["trapped_in_hole_rate"],
        "stone_break_rate": metrics["stone_break_rate"],
        "cobblestone_acquisition_rate": metrics["cobblestone_acquisition_rate"],
        "failure_distribution": metrics["primary_failure_distribution"],
    }


def grouped_rates(summary: dict, group: str) -> dict:
    return {name: compact({"overall": metrics}) for name, metrics in summary.get(group, {}).items()}


def main() -> None:
    summaries = {split: {policy: load_summary(split, policy) for policy in POLICIES} for split in SPLITS}
    rows = {split: {policy: load_rows(split, policy) for policy in POLICIES} for split in SPLITS}
    report = {"splits": {}, "comparisons": {}, "integrity": {}}
    for split in SPLITS:
        report["splits"][split] = {}
        for policy in POLICIES:
            summary = summaries[split][policy]
            report["splits"][split][policy] = {
                **compact(summary),
                "by_scenario": grouped_rates(summary, "by_scenario"),
                "by_biome": grouped_rates(summary, "by_biome"),
                "by_stress_stratum": grouped_rates(summary, "by_stress_stratum"),
            }
            report["integrity"][f"{split}/{policy}"] = {
                "episode_rows": len(rows[split][policy]),
                "unique_result_ids": len(set(rows[split][policy])),
                "manifest_kind": summary["manifest_kind"],
                "base_seed": summary["base_seed"],
                "benchmark_source_sha256": summary["source_sha256"]["benchmark_stone_acquisition.py"],
            }
        report["comparisons"][split] = {
            "vanilla_to_lora400": paired_comparison(rows[split]["vanilla"], rows[split]["lora400"]),
            "vanilla_to_lora800": paired_comparison(rows[split]["vanilla"], rows[split]["lora800"]),
            "lora400_to_lora800": paired_comparison(rows[split]["lora400"], rows[split]["lora800"]),
        }
    path = SUITE_DIRECTORY / "confirmation_report.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
