import json
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
ROOT = PROJECT_DIRECTORY / "output" / "stone_acquisition"
SMOKES = {
    "smoke_system0_isolation_old_router_hardened_pickup": ("old", "hardened"),
    "smoke_system0_isolation_new_router_old_pickup": ("new", "old"),
}


def main() -> None:
    output = {"smokes": {}, "paired_manifests": {}}
    for name, expected in SMOKES.items():
        directory = ROOT / name
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
        observed = {(row["router_implementation"], row["pickup_implementation"]) for row in rows}
        if summary["router_implementation"] != expected[0] or summary["pickup_implementation"] != expected[1] or observed != {expected}:
            raise RuntimeError(f"Selector mismatch for {name}: summary={summary['router_implementation'], summary['pickup_implementation']} rows={observed}")
        output["smokes"][name] = {"episodes": len(rows), "success_rate": summary["overall"]["success_rate"], "router": expected[0], "pickup": expected[1]}
    for split in ("normal", "hazard"):
        champion = ROOT / f"screen_{split}_upper_lora_r32_step6400_v1" / "manifest.jsonl"
        combined = ROOT / f"system0_ablation_{split}_repaired_v1" / "manifest.jsonl"
        champion_bytes = champion.read_bytes()
        combined_bytes = combined.read_bytes()
        if champion_bytes != combined_bytes:
            raise RuntimeError(f"Stored champion and combined manifests differ for {split}")
        output["paired_manifests"][split] = {"identical": True, "episodes": len(champion.read_text(encoding="utf-8").splitlines())}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
