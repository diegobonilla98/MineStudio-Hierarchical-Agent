import json
import os
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SWEEP_DIRECTORY = Path(os.environ["MINESTUDIO_BC_OUTPUT_DIRECTORY"])
VARIANTS = (
    "upper_lora",
    "upper_lora_rank32",
    "recurrent_upper_lora",
    "upper_blocks_full",
    "policy_no_visual_full",
)


def main() -> None:
    rows = []
    for variant in VARIANTS:
        summary_path = SWEEP_DIRECTORY / variant / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "variant": variant,
            "trainable_parameters": summary["trainable_parameters"],
            "completed_steps": summary["completed_steps"],
            "best_step": summary["best_step"],
            "initial_validation_loss": summary["initial_validation"]["loss"],
            "best_validation_loss": summary["best_validation_loss"],
            "targeted_validation_loss": summary["final_validation"]["targeted_loss"],
            "regularization_validation_loss": summary["final_validation"]["regularization_loss"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "checkpoint_directory": summary["checkpoint_directory"],
        })
    ranking = sorted(rows, key=lambda row: row["best_validation_loss"])
    report = {
        "sweep_directory": str(SWEEP_DIRECTORY),
        "selection_metric": "best_validation_loss",
        "winner": ranking[0]["variant"],
        "ranking": ranking,
    }
    output_path = SWEEP_DIRECTORY / "architecture_report.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
