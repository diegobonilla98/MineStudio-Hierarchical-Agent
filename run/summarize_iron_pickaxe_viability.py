import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SUITE_DIRECTORY = Path(os.environ.get("MINESTUDIO_IRON_SUITE", PROJECT_DIRECTORY / "output" / "iron_pickaxe_viability" / "development"))
ARMS = ("full", "no_specialist", "no_system2")
MILESTONES = (
    "logs_obtained",
    "crafting_capability_established",
    "wooden_pickaxe",
    "required_cobblestone",
    "stone_pickaxe",
    "furnace_available",
    "raw_iron_3",
    "iron_ingots_3",
    "iron_pickaxe",
)


def load_results(arm: str) -> dict[int, dict]:
    rows = {}
    path = SUITE_DIRECTORY / arm / "results.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["status"] == "complete":
            rows[int(row["seed"])] = row
    return rows


def exact_mcnemar(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    lower = min(discordant_a, discordant_b)
    probability = sum(math.comb(total, index) for index in range(lower + 1)) / (2 ** total)
    return min(1.0, 2.0 * probability)


def paired_comparison(left: dict[int, dict], right: dict[int, dict], metric) -> dict:
    seeds = sorted(set(left) & set(right))
    differences = np.array([float(metric(left[seed])) - float(metric(right[seed])) for seed in seeds], dtype=np.float64)
    generator = np.random.default_rng(20260921)
    bootstrap = np.array([
        float(generator.choice(differences, size=len(differences), replace=True).mean())
        for _ in range(20000)
    ]) if len(differences) else np.array([])
    return {
        "paired_episodes": len(seeds),
        "left_rate": float(np.mean([metric(left[seed]) for seed in seeds])) if seeds else None,
        "right_rate": float(np.mean([metric(right[seed]) for seed in seeds])) if seeds else None,
        "difference": float(differences.mean()) if len(differences) else None,
        "paired_bootstrap_95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))] if len(bootstrap) else None,
    }


def final_comparison(left: dict[int, dict], right: dict[int, dict]) -> dict:
    report = paired_comparison(left, right, lambda row: row["success"])
    seeds = sorted(set(left) & set(right))
    left_only = sum(int(left[seed]["success"] and not right[seed]["success"]) for seed in seeds)
    right_only = sum(int(right[seed]["success"] and not left[seed]["success"]) for seed in seeds)
    report["discordant_left_only"] = left_only
    report["discordant_right_only"] = right_only
    report["mcnemar_exact_p"] = exact_mcnemar(left_only, right_only)
    return report


def arm_report(rows: dict[int, dict]) -> dict:
    values = list(rows.values())
    milestone_report = {}
    for milestone in MILESTONES:
        achieved = [row for row in values if row["milestones"][milestone]["step"] is not None]
        milestone_report[milestone] = {
            "count": len(achieved),
            "rate": len(achieved) / len(values),
            "median_steps": float(np.median([row["milestones"][milestone]["step"] for row in achieved])) if achieved else None,
            "median_seconds": float(np.median([row["milestones"][milestone]["seconds"] for row in achieved])) if achieved else None,
        }
    owners = Counter()
    causes = Counter()
    for row in values:
        critique = row.get("failure_critique")
        if critique:
            owners[critique["primary_owner"]] += 1
            causes[critique["primary_cause"]] += 1
    return {
        "episodes": len(values),
        "successes": sum(int(row["success"]) for row in values),
        "success_rate": float(np.mean([row["success"] for row in values])),
        "milestones": milestone_report,
        "mean_steps": float(np.mean([row["total_steps"] for row in values])),
        "median_steps": float(np.median([row["total_steps"] for row in values])),
        "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in values])),
        "mean_system0_calls": float(np.mean([row["system0_calls"] for row in values])),
        "mean_system1_calls": float(np.mean([row["system1_calls"] for row in values])),
        "mean_router_transitions": float(np.mean([row["router_transitions"] for row in values])),
        "mean_planner_replans": float(np.mean([row["planner_replans"] for row in values])),
        "deaths": sum(int(row["death"]) for row in values),
        "failure_ownership": dict(owners),
        "failure_causes": dict(causes),
    }


def representative(rows: dict[int, dict]) -> dict:
    ordered = list(rows.values())
    successes = [row for row in ordered if row["success"]]
    failures = [row for row in ordered if not row["success"]]
    milestone_score = lambda row: sum(int(row["milestones"][name]["step"] is not None) for name in MILESTONES)
    success = min(successes, key=lambda row: row["total_steps"]) if successes else None
    failure = max(failures, key=lambda row: (milestone_score(row), -row["total_steps"])) if failures else None
    return {
        "success": {"seed": success["seed"], "steps": success["total_steps"], "episode_path": f"episodes/seed_{success['seed']}"} if success else None,
        "failure": {"seed": failure["seed"], "steps": failure["total_steps"], "last_milestone_count": milestone_score(failure), "episode_path": f"episodes/seed_{failure['seed']}"} if failure else None,
    }


def main() -> None:
    results = {arm: load_results(arm) for arm in ARMS}
    manifest = json.loads((SUITE_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    expected = {int(entry["seed"]) for entry in manifest["episodes"]}
    for arm in ARMS:
        if set(results[arm]) != expected:
            raise RuntimeError(f"{arm} does not have the exact frozen seed set: {len(results[arm])}/{len(expected)}")
    report = {
        "objective": manifest["objective"],
        "manifest_sha256": manifest["manifest_sha256"],
        "paired": True,
        "arms": {arm: arm_report(results[arm]) for arm in ARMS},
        "paired_final_success": {
            "full_vs_no_specialist": final_comparison(results["full"], results["no_specialist"]),
            "full_vs_no_system2": final_comparison(results["full"], results["no_system2"]),
        },
        "paired_milestone_differences": {
            comparison: {
                milestone: paired_comparison(results["full"], results[right], lambda row, name=milestone: row["milestones"][name]["step"] is not None)
                for milestone in MILESTONES
            }
            for comparison, right in (("full_vs_no_specialist", "no_specialist"), ("full_vs_no_system2", "no_system2"))
        },
        "representative_trajectories": {arm: representative(results[arm]) for arm in ARMS},
    }
    (SUITE_DIRECTORY / "final_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["IRON PICKAXE VIABILITY EXPERIMENT", ""]
    for arm in ARMS:
        item = report["arms"][arm]
        lines.append(f"{arm}: {item['successes']}/{item['episodes']} final success ({item['success_rate']:.1%})")
        lines.append("  milestones: " + ", ".join(f"{name}={item['milestones'][name]['count']}/{item['episodes']}" for name in MILESTONES))
        lines.append(f"  mean steps={item['mean_steps']:.1f}, S0 calls={item['mean_system0_calls']:.1f}, S1 calls={item['mean_system1_calls']:.1f}, transitions={item['mean_router_transitions']:.1f}")
        lines.append(f"  failure ownership={item['failure_ownership']}")
    lines.append("")
    for name, comparison in report["paired_final_success"].items():
        lines.append(f"{name}: difference={comparison['difference']:.1%}, 95% CI={comparison['paired_bootstrap_95_ci']}, McNemar p={comparison['mcnemar_exact_p']:.6f}")
    (SUITE_DIRECTORY / "final_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
