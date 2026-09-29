import json
import os
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
TRAINING_DIRECTORY = Path(os.environ.get("MINESTUDIO_BC_OUTPUT_DIRECTORY", PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "latest_run"))
VARIANTS = ("action_head", "action_head_recurrent", "upper_lora")


def main() -> None:
    candidates = []
    for variant in VARIANTS:
        summary_path = TRAINING_DIRECTORY / variant / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Missing training summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["best_step"] <= 0:
            continue
        checkpoint_directory = Path(summary["checkpoint_directory"])
        if not checkpoint_directory.is_dir():
            raise RuntimeError(f"Missing checkpoint: {checkpoint_directory}")
        candidates.append({
            "variant": variant,
            "validation_loss": summary["final_validation"]["loss"],
            "targeted_validation_loss": summary["final_validation"]["targeted_loss"],
            "regularization_validation_loss": summary["final_validation"]["regularization_loss"],
            "best_step": summary["best_step"],
            "checkpoint_directory": str(checkpoint_directory),
        })
    if not candidates:
        raise RuntimeError("No supervised variant improved over the pretrained checkpoint")
    selected = min(candidates, key=lambda item: (item["validation_loss"], item["targeted_validation_loss"]))
    result = {"selected": selected, "candidates": sorted(candidates, key=lambda item: item["validation_loss"])}
    output_path = TRAINING_DIRECTORY / "selection.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
