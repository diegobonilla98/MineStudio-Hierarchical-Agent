import json
import math
import os
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_acquisition"
SUITE_DIRECTORY = Path(os.environ.get("MINESTUDIO_EVAL_SUITE_DIRECTORY", OUTPUT_DIRECTORY / "evaluation_suite_v1"))
VARIANTS = ("action_head", "action_head_recurrent", "upper_lora")
RUNS = {
    "frozen": {
        "vanilla": OUTPUT_DIRECTORY / "stage0_baseline_v1",
        **{variant: OUTPUT_DIRECTORY / f"stage1_frozen_{variant}_v1" for variant in VARIANTS},
    },
    "heldout": {
        "vanilla": OUTPUT_DIRECTORY / "stage1_heldout_vanilla_v1",
        **{variant: OUTPUT_DIRECTORY / f"stage1_heldout_{variant}_v1" for variant in VARIANTS},
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_results(directory: Path) -> dict[str, dict]:
    path = directory / "episodes.jsonl"
    return {row["result_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def exact_mcnemar(control: dict[str, dict], candidate: dict[str, dict]) -> dict:
    ids = sorted(set(control) & set(candidate))
    control_only = sum(control[key]["success"] and not candidate[key]["success"] for key in ids)
    candidate_only = sum(candidate[key]["success"] and not control[key]["success"] for key in ids)
    discordant = control_only + candidate_only
    if discordant == 0:
        probability = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(control_only, candidate_only) + 1)) / (2**discordant)
        probability = min(1.0, 2.0 * tail)
    return {
        "paired_episodes": len(ids),
        "control_success_candidate_failure": control_only,
        "control_failure_candidate_success": candidate_only,
        "discordant_pairs": discordant,
        "exact_two_sided_p": probability,
    }


def non_water_success(results: dict[str, dict]) -> float:
    selected = [row for row in results.values() if row["scenario"] != "water_near_stone"]
    return sum(row["success"] for row in selected) / len(selected)


def compact(summary: dict, results: dict[str, dict]) -> dict:
    overall = summary["overall"]
    water = summary["by_scenario"]["water_near_stone"]
    swamp = summary["by_biome"]["swamp"]
    return {
        "episodes": overall["episodes"],
        "success_rate": overall["success_rate"],
        "success_wilson_95": overall["success_wilson_95"],
        "water_near_stone_success_rate": water["success_rate"],
        "swamp_success_rate": swamp["success_rate"],
        "non_water_success_rate": non_water_success(results),
        "mean_steps_to_success": overall["mean_steps_to_success"],
        "water_entry_rate": overall["water_entry_rate"],
        "stuck_rate": overall["stuck_rate"],
        "trapped_in_hole_rate": overall["trapped_in_hole_rate"],
        "stone_break_rate": overall["stone_break_rate"],
        "cobblestone_acquisition_rate": overall["cobblestone_acquisition_rate"],
        "failure_distribution": overall["primary_failure_distribution"],
    }


def main() -> None:
    report = {"splits": {}, "gates": {}, "selection": {}}
    cached_results = {}
    for split, paths in RUNS.items():
        summaries = {name: read_json(path / "summary.json") for name, path in paths.items()}
        results = {name: episode_results(path) for name, path in paths.items()}
        cached_results[split] = results
        control = compact(summaries["vanilla"], results["vanilla"])
        split_report = {"vanilla": control}
        for variant in VARIANTS:
            candidate = compact(summaries[variant], results[variant])
            candidate["delta_success_rate"] = candidate["success_rate"] - control["success_rate"]
            candidate["delta_water_near_stone"] = candidate["water_near_stone_success_rate"] - control["water_near_stone_success_rate"]
            candidate["delta_swamp"] = candidate["swamp_success_rate"] - control["swamp_success_rate"]
            candidate["delta_non_water"] = candidate["non_water_success_rate"] - control["non_water_success_rate"]
            candidate["paired_test"] = exact_mcnemar(results["vanilla"], results[variant])
            split_report[variant] = candidate
        report["splits"][split] = split_report
    for variant in VARIANTS:
        heldout = report["splits"]["heldout"][variant]
        gate = {
            "overall_at_least_0_88": heldout["success_rate"] >= 0.88,
            "water_near_stone_at_least_0_70": heldout["water_near_stone_success_rate"] >= 0.70,
            "swamp_at_least_0_80": heldout["swamp_success_rate"] >= 0.80,
            "non_water_regression_at_most_0_02": heldout["delta_non_water"] >= -0.02,
        }
        gate["passed"] = all(gate.values())
        report["gates"][variant] = gate
    passing = [variant for variant in VARIANTS if report["gates"][variant]["passed"]]
    ranked = sorted(VARIANTS, key=lambda variant: (
        report["splits"]["heldout"][variant]["success_rate"],
        report["splits"]["heldout"][variant]["water_near_stone_success_rate"],
        report["splits"]["frozen"][variant]["success_rate"],
    ), reverse=True)
    report["selection"] = {
        "status": "gate_passed" if passing else "no_variant_passed_all_gates",
        "selected": next((variant for variant in ranked if variant in passing), None),
        "ranking": ranked,
    }
    SUITE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = SUITE_DIRECTORY / "evaluation_report.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
