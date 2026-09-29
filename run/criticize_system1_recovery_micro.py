import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SUITE_DIRECTORY = Path(os.environ["MINESTUDIO_MICRO_SUITE_DIRECTORY"])
MODEL_NAME = "gemini-3.8-flash"
TASKS = (
    "EXIT_WATER",
    "CLIMB_SHORE",
    "REACQUIRE_STONE",
    "ESCAPE_HOLE",
    "AVOID_DIGGING_TRAP",
    "RECOVER_CAMERA",
    "AVOID_WATER",
)
LABELS = (
    "wrong_exit_direction",
    "failed_shore_climb",
    "water_oscillation",
    "failed_stone_reacquisition",
    "target_lost",
    "mined_into_trap",
    "failed_hole_escape",
    "bad_camera_orientation",
    "unnecessary_water_entry",
    "stuck_no_progress",
    "successful_partial_recovery",
    "other",
)
MAX_FRAMES = 10
SYSTEM_PROMPT = """You are a Minecraft visuomotor trajectory critic evaluating a frozen System 1 controller on one controlled recovery micro-benchmark. Diagnose execution behavior from chronological frames and exact MineStudio state. The exact task, verifier, outcome, and environment-derived diagnostics are authoritative. Do not judge planning, System 0, or strategy. Do not invent unseen events. Select only labels directly supported by the evidence. Feedback will determine a narrow PPO curriculum, not a numerical reward."""


class Critique(BaseModel):
    labels: list[Literal["wrong_exit_direction", "failed_shore_climb", "water_oscillation", "failed_stone_reacquisition", "target_lost", "mined_into_trap", "failed_hole_escape", "bad_camera_orientation", "unnecessary_water_entry", "stuck_no_progress", "successful_partial_recovery", "other"]]
    summary: str = Field(min_length=5, max_length=500)
    evidence: list[str] = Field(max_length=5)
    recommended_curriculum: str = Field(min_length=5, max_length=300)


def frame_parts(path: Path) -> tuple[list[types.Part], list[int]]:
    with np.load(path, allow_pickle=False) as arrays:
        frames = arrays["rgb"]
        indices = np.linspace(0, len(frames) - 1, min(MAX_FRAMES, len(frames)), dtype=int).tolist()
        parts = []
        for index in indices:
            success, encoded = cv2.imencode(".jpg", cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not success:
                raise RuntimeError(f"Could not encode frame {index} from {path}")
            parts.append(types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"))
    return parts, indices


def failure_rows() -> list[tuple[str, Path, dict]]:
    rows = []
    for task in TASKS:
        task_directory = SUITE_DIRECTORY / task.lower()
        episodes_path = task_directory / "episodes.jsonl"
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row["success"]:
                rows.append((task, task_directory, row))
    return rows


def write_summary(output_path: Path, existing: dict, target: int, status: str, error: str | None = None) -> None:
    by_task = {}
    for task in TASKS:
        task_rows = [row for row in existing.values() if row["task"] == task]
        by_task[task] = {
            "critiques": len(task_rows),
            "label_counts": dict(sorted(Counter(label for row in task_rows for label in row["labels"]).items())),
        }
    summary = {
        "status": status,
        "model": MODEL_NAME,
        "critiques": len(existing),
        "target": target,
        "remaining": max(0, target - len(existing)),
        "label_counts": dict(sorted(Counter(label for row in existing.values() for label in row["labels"]).items())),
        "by_task": by_task,
        "output": str(output_path),
    }
    if error is not None:
        summary["error"] = error
    summary_path = SUITE_DIRECTORY / "gemini_critic_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary), flush=True)


def main() -> None:
    load_dotenv(PROJECT_DIRECTORY / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    client = genai.Client(api_key=api_key)
    output_path = SUITE_DIRECTORY / "gemini_failure_critiques.jsonl"
    existing = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["critique_id"]] = row
    rows = failure_rows()
    target = len(rows)
    for task, task_directory, row in rows:
        critique_id = f"{task}:{row['result_id']}"
        if critique_id in existing:
            continue
        parts, indices = frame_parts(task_directory / row["trajectory_npz"])
        request = {
            "task": task,
            "task_instruction": row["policy_prompt"],
            "result_id": row["result_id"],
            "verifier_failure_reason": row["failure_reason"],
            "steps": row["steps"],
            "diagnostics": row["diagnostics"],
            "router_transitions": row["router_transitions"],
            "sampled_frame_indices": indices,
            "frame_order": "oldest_to_newest",
            "allowed_labels": LABELS,
        }
        contents = [types.Part.from_text(text=json.dumps(request, separators=(",", ":"))), *parts]
        critique = None
        for retry in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=Critique,
                        max_output_tokens=2048,
                        temperature=0.1,
                        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                critique = Critique.model_validate_json(response.text)
                break
            except errors.ClientError as error:
                status_code = getattr(error, "code", None)
                if status_code == 402 or "402" in str(error) and "prepayment credits are depleted" in str(error):
                    write_summary(output_path, existing, target, "blocked_billing", str(error))
                    return
                if retry == 2:
                    raise
                time.sleep(2**retry)
            except ValidationError:
                if retry == 2:
                    raise
                time.sleep(2**retry)
            except Exception:
                if retry == 2:
                    raise
                time.sleep(2**retry)
        output = {"critique_id": critique_id, "model": MODEL_NAME, "task": task, "result_id": row["result_id"], **critique.model_dump()}
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(output, separators=(",", ":")) + "\n")
        existing[critique_id] = output
        write_summary(output_path, existing, target, "running")
    write_summary(output_path, existing, target, "completed")


if __name__ == "__main__":
    main()
