import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


OUTPUT_DIRECTORY = Path(os.environ["MINESTUDIO_PPO_OUTPUT"])


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    checkpoint = torch.load(OUTPUT_DIRECTORY / "last_training_state.pt", map_location="cpu", weights_only=False)
    records = [json.loads(line) for line in (OUTPUT_DIRECTORY / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    development = [record for record in records if record["split"] == "development"]
    training = [record for record in records if record["split"] == "train"]
    best_record = max(development, key=lambda record: record["behavioral_score"])
    best = checkpoint["best"]
    if best["iteration"] != best_record["iteration"]:
        raise RuntimeError(f"Checkpoint best iteration {best['iteration']} differs from metrics best {best_record['iteration']}")
    checkpoint_directory = Path(best["checkpoint_directory"])
    if not (checkpoint_directory / "model.safetensors").is_file():
        raise RuntimeError(f"Best model is missing: {checkpoint_directory}")
    summary = {
        "status": "completed",
        "run_id": OUTPUT_DIRECTORY.name,
        "completed_training_iterations": max(record["iteration"] for record in training),
        "last_persisted_iteration": checkpoint["iteration"],
        "stop_reason": "behavioral_plateau_after_iteration_10_peak",
        "baseline_evaluation": checkpoint["baseline_evaluation"],
        "best": best,
        "evaluated_iterations": [record["iteration"] for record in development],
        "discarded_partial_evaluation_iteration": 20,
        "training_seconds": sum(float(record["iteration_seconds"]) for record in training),
        "reference_checkpoint": "/home/boni/projects/MineStudio/output/stone_recovery_bc/architecture_sweeps/20260917T213751Z/upper_lora_rank32/best_model",
        "training_curve": str(OUTPUT_DIRECTORY / "metrics.jsonl"),
    }
    atomic_json(OUTPUT_DIRECTORY / "summary.json", summary)
    atomic_json(OUTPUT_DIRECTORY / "run_state.json", {
        "status": "completed",
        "phase": "training",
        "iteration": max(record["iteration"] for record in training),
        "stop_reason": summary["stop_reason"],
        "best_iteration": best["iteration"],
        "best_checkpoint_directory": best["checkpoint_directory"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
