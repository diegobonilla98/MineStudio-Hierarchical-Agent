import json
import pickle
from pathlib import Path

import av
import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = PROJECT_DIRECTORY / "output" / "coach_dataset"


def load_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    session_directories = sorted(path for path in DATASET_DIRECTORY.iterdir() if path.is_dir())
    if not session_directories:
        raise RuntimeError(f"No recorded sessions found in {DATASET_DIRECTORY}")
    session_directory = session_directories[-1]
    metadata = json.loads((session_directory / "session.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise RuntimeError(f"Latest session is not finalized: {metadata['status']}")

    action_path = session_directory / metadata["files"]["actions"]
    minestudio_action_path = session_directory / metadata["files"]["minestudio_actions"]
    info_path = session_directory / metadata["files"]["infos"]
    steps_path = session_directory / metadata["files"]["steps"]
    tasks_path = session_directory / metadata["files"]["tasks"]
    video_path = session_directory / metadata["files"]["video"]
    if video_path.stem != minestudio_action_path.stem or video_path.stem != metadata["session_id"]:
        raise RuntimeError("Video and MineStudio action filenames do not identify the same unique episode")
    actions = json.loads(action_path.read_text(encoding="utf-8"))
    with minestudio_action_path.open("rb") as action_pickle_file:
        action_arrays = pickle.load(action_pickle_file)
    infos = json.loads(info_path.read_text(encoding="utf-8"))
    steps = load_json_lines(steps_path)
    tasks = load_json_lines(tasks_path)
    total_frames = metadata["total_frames"]

    counts = {"metadata": total_frames, "actions": len(actions), "infos": len(infos), "steps": len(steps)}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Frame-aligned record counts differ: {counts}")
    if any(len(values) != total_frames for values in action_arrays.values()):
        raise RuntimeError("MineStudio action arrays do not match the video length")
    if action_arrays["camera"].shape != (total_frames, 2):
        raise RuntimeError(f"Invalid camera action shape: {action_arrays['camera'].shape}")
    if metadata["total_tasks"] != len(tasks):
        raise RuntimeError(f"Task count differs: metadata={metadata['total_tasks']}, tasks={len(tasks)}")
    if total_frames == 0:
        raise RuntimeError("The session contains no recorded frames")
    if not tasks:
        raise RuntimeError("The session contains no task annotations")
    if tasks[0]["start_frame"] != 0 or tasks[-1]["end_frame"] != total_frames:
        raise RuntimeError("Task annotations do not cover the complete trajectory")
    for previous_task, next_task in zip(tasks, tasks[1:]):
        if previous_task["end_frame"] != next_task["start_frame"]:
            raise RuntimeError("Task annotations contain a gap or overlap")
    for frame_id, step in enumerate(steps):
        if step["frame_id"] != frame_id or step["action"] != actions[frame_id] or infos[frame_id]["frame_id"] != frame_id:
            raise RuntimeError(f"Step alignment failed at frame {frame_id}")
    global_goal = metadata.get("global_goal")
    if global_goal is not None:
        if any(step.get("global_goal_id") != global_goal["goal_id"] for step in steps):
            raise RuntimeError("Per-step global goal IDs do not match session metadata")
        if abs(steps[-1]["global_progress_after"] - metadata["global_progress"]) > 1e-6:
            raise RuntimeError("Final global progress does not match session metadata")
    camera = action_arrays["camera"]
    if not np.isfinite(camera).all():
        raise RuntimeError("Camera actions contain non-finite values")

    video_container = av.open(video_path)
    decoded_frames = sum(1 for _ in video_container.decode(video=0))
    video_container.close()
    if decoded_frames != total_frames:
        raise RuntimeError(f"Video frame count differs: video={decoded_frames}, metadata={total_frames}")

    successful_tasks = sum(task["succeeded"] for task in tasks)
    timestamps = np.asarray([step["timestamp_seconds"] for step in steps], dtype=np.float64)
    timestamp_deltas = np.diff(timestamps)
    gameplay_deltas = timestamp_deltas[(timestamp_deltas > 0) & (timestamp_deltas < 0.2)]
    effective_fps = 1.0 / np.median(gameplay_deltas) if len(gameplay_deltas) else 0.0
    pause_count = int(np.count_nonzero(timestamp_deltas >= 0.2))
    camera_magnitude = np.linalg.norm(camera, axis=1)
    active_buttons = {
        name: int(np.count_nonzero(values))
        for name, values in action_arrays.items()
        if name != "camera" and np.count_nonzero(values)
    }
    print(f"VALID: {session_directory}")
    print(f"Frames/actions: {total_frames}")
    print(f"Tasks: {len(tasks)} ({successful_tasks} succeeded)")
    print(f"Close reason: {metadata['close_reason']}")
    if global_goal is not None:
        print(f"Global goal: {global_goal['instruction']} ({metadata['global_progress']:.1f}/{global_goal['verifier']['threshold']:g})")
    print(f"Observed gameplay rate: {effective_fps:.2f} steps/s ({pause_count} coach pauses excluded)")
    print(f"Camera magnitude p95/p99/max: {np.percentile(camera_magnitude, 95):.3f}/{np.percentile(camera_magnitude, 99):.3f}/{camera_magnitude.max():.3f}")
    print(f"Active button frames: {active_buttons}")


if __name__ == "__main__":
    main()
