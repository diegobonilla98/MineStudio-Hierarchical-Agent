import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_acquisition" / "stage0_baseline_v1"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "dataset_v1"
VALIDATION_FRACTION = 0.20
MIN_WINDOW_STEPS = 24
STABLE_DRY_STEPS = 20
WATER_ENTRY_BEFORE = 60
WATER_ENTRY_AFTER = 200
SHORE_EXIT_BEFORE = 120
SHORE_EXIT_AFTER = 100
STONE_REACQUIRE_BEFORE = 120
STONE_REACQUIRE_AFTER = 40
HOLE_BEFORE = 80
HOLE_AFTER = 120
REGULARIZATION_LENGTH = 128
REGULARIZATION_WINDOWS_PER_EPISODE = 3
STEVE_PHASE_PREFIX = "STEVE_1/"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assign_splits(rows: list[dict]) -> dict[str, str]:
    groups = defaultdict(list)
    for row in rows:
        if not row["success"]:
            continue
        group = (bool(row["entered_water"]), row["biome"])
        groups[group].append(row["result_id"])
    assignments = {}
    for result_ids in groups.values():
        ordered = sorted(result_ids, key=stable_key)
        validation_count = max(1, round(len(ordered) * VALIDATION_FRACTION))
        validation_ids = set(ordered[:validation_count])
        for result_id in ordered:
            assignments[result_id] = "validation" if result_id in validation_ids else "train"
    return assignments


def transition_indices(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.diff(values) > 0) + 1


def water_entries(water: np.ndarray) -> list[int]:
    entries = []
    if len(water) and water[0] == 1:
        entries.append(0)
    if len(water) > 1:
        entries.extend((np.flatnonzero((water[1:] == 1) & (water[:-1] != 1)) + 1).tolist())
    return entries


def stable_water_exits(water: np.ndarray) -> list[int]:
    exits = []
    if len(water) < 2:
        return exits
    candidates = np.flatnonzero((water[1:] == 0) & (water[:-1] == 1)) + 1
    for index in candidates:
        end = min(len(water), index + STABLE_DRY_STEPS)
        if end - index == STABLE_DRY_STEPS and np.all(water[index:end] == 0):
            exits.append(int(index))
    return exits


def hole_recoveries(position: np.ndarray, baseline_y: float) -> list[tuple[int, int]]:
    if len(position) == 0:
        return []
    y_values = position[:, 1]
    low_indices = np.flatnonzero(y_values <= baseline_y - 1.5)
    recoveries = []
    consumed_until = -1
    for low_index in low_indices:
        if low_index <= consumed_until:
            continue
        recovered = np.flatnonzero(y_values[low_index + 1:] >= baseline_y - 0.5)
        if len(recovered) == 0:
            continue
        recovery_index = int(low_index + 1 + recovered[0])
        if recovery_index - low_index >= 8:
            recoveries.append((int(low_index), recovery_index))
            consumed_until = recovery_index
    return recoveries


def steve_spans(phases: list[str], start: int, end: int) -> list[tuple[int, int]]:
    spans = []
    span_start = None
    for index in range(max(0, start), min(len(phases), end)):
        is_steve = phases[index].startswith(STEVE_PHASE_PREFIX)
        if is_steve and span_start is None:
            span_start = index
        if not is_steve and span_start is not None:
            if index - span_start >= MIN_WINDOW_STEPS:
                spans.append((span_start, index))
            span_start = None
    clipped_end = min(len(phases), end)
    if span_start is not None and clipped_end - span_start >= MIN_WINDOW_STEPS:
        spans.append((span_start, clipped_end))
    return spans


def add_candidate(candidates: list[dict], row: dict, phases: list[str], start: int, end: int, tags: list[str], source_group: str, split: str) -> None:
    tag_set = set(tags)
    if "hole_recovery" in tag_set:
        task_category = "hole_recovery"
    elif "stone_reacquisition" in tag_set or "resume_mining" in tag_set:
        task_category = "stone_reacquisition"
    elif "shore_exit" in tag_set:
        task_category = "shore_exit"
    elif "water_entry" in tag_set or "water_recovery" in tag_set:
        task_category = "water_recovery"
    else:
        task_category = "stone_acquisition"
    for span_start, span_end in steve_spans(phases, start, end):
        candidates.append({
            "window_id": f"{row['result_id']}:{span_start:04d}:{span_end:04d}:{'+'.join(sorted(tags))}",
            "result_id": row["result_id"],
            "episode_index": row["episode_index"],
            "trajectory_npz": row["trajectory_npz"],
            "trajectory_metadata": row["trajectory_metadata"],
            "start": span_start,
            "end": span_end,
            "steps": span_end - span_start,
            "tags": sorted(set(tags)),
            "source_group": source_group,
            "split": split,
            "scenario": row["scenario"],
            "biome": row["biome"],
            "entered_water": bool(row["entered_water"]),
            "task_category": task_category,
        })


def regularization_centers(stone_progress: np.ndarray, length: int) -> list[int]:
    centers = [length // 4]
    centers.extend(int(index) for index in stone_progress[:REGULARIZATION_WINDOWS_PER_EPISODE - 1])
    return centers[:REGULARIZATION_WINDOWS_PER_EPISODE]


def extract_episode(row: dict, split: str) -> list[dict]:
    npz_path = SOURCE_DIRECTORY / row["trajectory_npz"]
    metadata_path = SOURCE_DIRECTORY / row["trajectory_metadata"]
    with np.load(npz_path, allow_pickle=False) as arrays:
        water = arrays["water"].copy()
        stone_mined = arrays["stone_mined"].copy()
        position = arrays["position"].copy()
    with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
        metadata = json.load(handle)
    phases = metadata["phases"]
    candidates = []
    entries = water_entries(water)
    exits = stable_water_exits(water)
    stone_progress = transition_indices(stone_mined)
    if row["entered_water"]:
        for entry in entries:
            add_candidate(candidates, row, phases, entry - WATER_ENTRY_BEFORE, entry + WATER_ENTRY_AFTER, ["water_entry", "water_recovery"], "targeted_recovery", split)
        for exit_index in exits:
            add_candidate(candidates, row, phases, exit_index - SHORE_EXIT_BEFORE, exit_index + SHORE_EXIT_AFTER, ["shore_exit", "water_recovery"], "targeted_recovery", split)
            later_progress = stone_progress[stone_progress > exit_index]
            if len(later_progress):
                progress_index = int(later_progress[0])
                add_candidate(candidates, row, phases, progress_index - STONE_REACQUIRE_BEFORE, progress_index + STONE_REACQUIRE_AFTER, ["stone_reacquisition", "resume_mining"], "targeted_recovery", split)
    baseline_y = float(metadata["baseline_state"]["position"]["y"])
    for low_index, recovered_index in hole_recoveries(position, baseline_y):
        add_candidate(candidates, row, phases, low_index - HOLE_BEFORE, recovered_index + HOLE_AFTER, ["hole_recovery"], "targeted_recovery", split)
    if not row["entered_water"]:
        for center in regularization_centers(stone_progress, len(phases)):
            start = max(0, center - REGULARIZATION_LENGTH // 2)
            add_candidate(candidates, row, phases, start, start + REGULARIZATION_LENGTH, ["ordinary_stone"], "stone_regularization", split)
    deduplicated = {}
    for candidate in candidates:
        deduplicated[candidate["window_id"]] = candidate
    return list(deduplicated.values())


def validate_source(rows: list[dict], summary: dict) -> None:
    if summary["episodes_complete"] != summary["episodes_target"]:
        raise RuntimeError("Stage 0 baseline is incomplete")
    if len(rows) != summary["episodes_complete"]:
        raise RuntimeError("Episode result count does not match summary")
    result_ids = [row["result_id"] for row in rows]
    episode_indices = [row["episode_index"] for row in rows]
    if len(set(result_ids)) != len(rows) or len(set(episode_indices)) != len(rows):
        raise RuntimeError("Stage 0 baseline contains duplicate episode identifiers")


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary_path = SOURCE_DIRECTORY / "summary.json"
    episodes_path = SOURCE_DIRECTORY / "episodes.jsonl"
    manifest_path = SOURCE_DIRECTORY / "manifest.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_jsonl(episodes_path)
    validate_source(rows, summary)
    assignments = assign_splits(rows)
    successful_rows = [row for row in rows if row["success"]]
    windows = []
    for complete, row in enumerate(successful_rows, start=1):
        windows.extend(extract_episode(row, assignments[row["result_id"]]))
        if complete % 50 == 0 or complete == len(successful_rows):
            print(json.dumps({"status": "extracting", "complete": complete, "target": len(successful_rows), "windows": len(windows)}), flush=True)
    windows.sort(key=lambda item: (item["split"], item["source_group"], item["episode_index"], item["start"], item["window_id"]))
    (OUTPUT_DIRECTORY / "windows.jsonl").write_text("".join(json.dumps(item, separators=(",", ":")) + "\n" for item in windows), encoding="utf-8")
    write_json(OUTPUT_DIRECTORY / "episode_splits.json", assignments)
    frozen_manifest_path = OUTPUT_DIRECTORY / "frozen_manifest.jsonl"
    shutil.copyfile(manifest_path, frozen_manifest_path)
    freeze = {
        "source_directory": str(SOURCE_DIRECTORY),
        "episodes": len(rows),
        "successful_episodes": len(successful_rows),
        "summary_sha256": file_sha256(summary_path),
        "episodes_sha256": file_sha256(episodes_path),
        "manifest_sha256": file_sha256(manifest_path),
        "frozen_manifest": str(frozen_manifest_path),
        "source_sha256": summary["source_sha256"],
    }
    if file_sha256(frozen_manifest_path) != freeze["manifest_sha256"]:
        raise RuntimeError("Frozen manifest copy does not match the Stage 0 manifest")
    write_json(OUTPUT_DIRECTORY / "frozen_baseline.json", freeze)
    counts = Counter((window["split"], window["source_group"]) for window in windows)
    tag_counts = Counter(tag for window in windows for tag in window["tags"])
    episode_counts = Counter((split, bool(next(row for row in successful_rows if row["result_id"] == result_id)["entered_water"])) for result_id, split in assignments.items())
    dataset_summary = {
        "source": freeze,
        "windows": len(windows),
        "window_steps": sum(window["steps"] for window in windows),
        "counts": {f"{split}/{group}": count for (split, group), count in sorted(counts.items())},
        "tag_counts": dict(sorted(tag_counts.items())),
        "successful_episode_splits": {f"{split}/water={water}": count for (split, water), count in sorted(episode_counts.items())},
        "transition_contract": "row t stores observation/state before action t",
        "action_target": "agent_buttons and agent_camera from STEVE_1 phases only",
        "deterministic_option_frames_included": False,
        "task_conditioning": "Gemini task category attached to every training window",
    }
    write_json(OUTPUT_DIRECTORY / "summary.json", dataset_summary)
    print(json.dumps({"status": "complete", "output": str(OUTPUT_DIRECTORY), **dataset_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
