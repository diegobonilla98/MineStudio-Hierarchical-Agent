import json
from datetime import datetime, timezone

import numpy as np
import benchmark_stone_acquisition as benchmark


SMOKE_EPISODES = 1
SMOKE_WORKERS = 1
SMOKE_MAX_STEPS = 60


def validate_output(output_directory) -> dict:
    summary = json.loads((output_directory / "summary.json").read_text(encoding="utf-8"))
    episode_lines = (output_directory / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    manifest_lines = (output_directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["episodes_complete"] == SMOKE_EPISODES
    assert summary["episodes_target"] == SMOKE_EPISODES
    assert len(episode_lines) == SMOKE_EPISODES
    assert len(manifest_lines) == SMOKE_EPISODES
    result = json.loads(episode_lines[0])
    trajectory_path = output_directory / result["trajectory_npz"]
    assert trajectory_path.is_file()
    with np.load(trajectory_path) as trajectory:
        assert trajectory["rgb"].shape[0] == result["steps"]
        assert trajectory["rgb"].shape[1:] == (128, 128, 3)
    return {
        "output": str(output_directory),
        "episodes": summary["episodes_complete"],
        "steps": result["steps"],
        "success": result["success"],
        "labels": result["labels"],
        "trajectory": str(trajectory_path),
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_directory = benchmark.OUTPUT_ROOT / f"smoke_main_{timestamp}"
    benchmark.EPISODES = SMOKE_EPISODES
    benchmark.NUM_WORKERS = SMOKE_WORKERS
    benchmark.MAX_STEPS = SMOKE_MAX_STEPS
    benchmark.EXPERIMENT_NAME = output_directory.name
    benchmark.EXPERIMENT_DIRECTORY = output_directory
    benchmark.main()
    first = validate_output(output_directory)
    benchmark.main()
    second = validate_output(output_directory)
    assert first == second
    print(json.dumps({"first_run": first, "resume_run": second}, indent=2), flush=True)


if __name__ == "__main__":
    main()
