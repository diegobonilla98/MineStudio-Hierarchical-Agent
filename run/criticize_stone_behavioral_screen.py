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
from pydantic import BaseModel, Field


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SCREEN_DIRECTORY = Path(os.environ["MINESTUDIO_SCREEN_DIRECTORY"])
MODEL_NAME = "gemini-3.8-flash"
DEFAULT_POLICIES = (
    "vanilla",
    "upper_lora_r8_step800",
    "upper_lora_r8_step1600",
    "upper_lora_r8_step3200",
    "upper_lora_r8_step4800",
    "upper_lora_r8_step6400",
    "upper_lora_r32_step6400",
    "recurrent_upper_lora_step6400",
)
POLICIES = tuple(value for value in os.environ.get("MINESTUDIO_CRITIC_POLICIES", ",".join(DEFAULT_POLICIES)).split(",") if value)
SPLITS = tuple(value for value in os.environ.get("MINESTUDIO_CRITIC_SPLITS", "normal,hazard").split(",") if value)
SUCCESS_SAMPLES_PER_GROUP = int(os.environ.get("MINESTUDIO_CRITIC_SUCCESS_SAMPLES", "5"))
EXPERIMENT_PREFIX = os.environ.get("MINESTUDIO_CRITIC_EXPERIMENT_PREFIX", "screen")
MAX_FRAMES = 12
MAX_TASKS = int(os.environ.get("MINESTUDIO_CRITIC_MAX_TASKS", "0"))
LABELS = (
    "unnecessary_water_entry",
    "wrong_shoreline_selected",
    "failed_shore_climb",
    "water_oscillation",
    "target_lost_after_recovery",
    "failed_stone_reacquisition",
    "bad_camera_orientation",
    "mined_into_trap",
    "successful_recovery",
    "other",
)
SYSTEM_PROMPT = """You are a Minecraft visuomotor trajectory critic. Diagnose execution behavior from chronological frames, fixed local-objective transitions, and exact MineStudio outcome state. Exact state and verifier labels are authoritative. Do not judge planning quality and do not invent unseen events. Select only labels supported by evidence. Your feedback will define a later RL curriculum, not a numerical reward."""


class Critique(BaseModel):
    labels: list[Literal["unnecessary_water_entry", "wrong_shoreline_selected", "failed_shore_climb", "water_oscillation", "target_lost_after_recovery", "failed_stone_reacquisition", "bad_camera_orientation", "mined_into_trap", "successful_recovery", "other"]]
    summary: str = Field(min_length=5, max_length=500)
    evidence: list[str] = Field(max_length=5)
    recommended_curriculum: str = Field(min_length=5, max_length=300)


def write_summary(output_path: Path, existing: dict, target: int, status: str, error: str | None = None) -> dict:
    counts = Counter(label for row in existing.values() for label in row["labels"])
    summary = {
        "status": status,
        "model": MODEL_NAME,
        "critiques": len(existing),
        "target": target,
        "remaining": max(0, target - len(existing)),
        "label_counts": dict(sorted(counts.items())),
        "output": str(output_path),
    }
    if error is not None:
        summary["error"] = error
    summary_path = SCREEN_DIRECTORY / "gemini_trajectory_critic_summary.json"
    temporary_path = summary_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(summary_path)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def experiment_directory(split: str, policy: str) -> Path:
    return PROJECT_DIRECTORY / "output" / "stone_acquisition" / f"{EXPERIMENT_PREFIX}_{split}_{policy}_v1"


def selected_rows(split: str, policy: str) -> list[dict]:
    rows = [json.loads(line) for line in (experiment_directory(split, policy) / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    failures = [row for row in rows if not row["success"]]
    successes = sorted((row for row in rows if row["success"]), key=lambda row: row["result_id"])[:SUCCESS_SAMPLES_PER_GROUP]
    return failures + successes


def frame_parts(path: Path) -> tuple[list[types.Part], list[int]]:
    with np.load(path, allow_pickle=False) as arrays:
        frames = arrays["rgb"]
        indices = np.linspace(0, len(frames) - 1, min(MAX_FRAMES, len(frames)), dtype=int).tolist()
        parts = []
        for index in indices:
            success, encoded = cv2.imencode(".jpg", cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                raise RuntimeError(f"Could not encode frame {index} from {path}")
            parts.append(types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"))
    return parts, indices


def main() -> None:
    load_dotenv(PROJECT_DIRECTORY / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    client = genai.Client(api_key=api_key)
    output_path = SCREEN_DIRECTORY / "gemini_trajectory_critiques.jsonl"
    existing = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                existing[row["critique_id"]] = row
    tasks = []
    for split in SPLITS:
        for policy in POLICIES:
            for row in selected_rows(split, policy):
                critique_id = f"{split}:{policy}:{row['result_id']}"
                if critique_id not in existing:
                    tasks.append((critique_id, split, policy, row))
    target = len(existing) + len(tasks)
    if MAX_TASKS > 0:
        tasks = tasks[:MAX_TASKS]
        target = len(existing) + len(tasks)
    for task_index, (critique_id, split, policy, row) in enumerate(tasks, start=1):
        directory = experiment_directory(split, policy)
        parts, indices = frame_parts(directory / row["trajectory_npz"])
        request = {
            "policy": policy,
            "split": split,
            "result_id": row["result_id"],
            "success": row["success"],
            "primary_label": row["primary_label"],
            "exact_labels": row["labels"],
            "entered_water": row["entered_water"],
            "water_exit_success": row["water_exit_success"],
            "stone_reacquired_after_water": row["stone_reacquired_after_water"],
            "died": row["died"],
            "stuck": row["stuck"],
            "trapped_in_hole": row["trapped_in_hole"],
            "cobblestone_delta": row["cobblestone_delta"],
            "stone_mined_delta": row["stone_mined_delta"],
            "task_transitions": row["task_transitions"],
            "sampled_frame_indices": indices,
            "frame_order": "oldest_to_newest",
            "allowed_labels": LABELS,
        }
        contents = [types.Part.from_text(text=json.dumps(request, separators=(",", ":"))), *parts]
        response = None
        for retry in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=Critique,
                        max_output_tokens=1024,
                        temperature=0.1,
                        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                break
            except errors.ClientError as error:
                status_code = getattr(error, "code", None)
                if status_code == 402 or "402" in str(error) and "prepayment credits are depleted" in str(error):
                    write_summary(output_path, existing, target, "blocked_billing", str(error))
                    return
                if retry == 2:
                    raise
                time.sleep(2 ** retry)
            except Exception:
                if retry == 2:
                    raise
                time.sleep(2 ** retry)
        critique = Critique.model_validate_json(response.text)
        output = {"critique_id": critique_id, "model": MODEL_NAME, "split": split, "policy": policy, "result_id": row["result_id"], "success": row["success"], **critique.model_dump()}
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
        existing[critique_id] = output
        print(json.dumps({"status": "critic_progress", "complete": task_index, "target": len(tasks), "critique_id": critique_id, "labels": critique.labels}), flush=True)
    write_summary(output_path, existing, target, "completed")


if __name__ == "__main__":
    main()
