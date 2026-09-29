import faulthandler
import json
import math
import os
import pickle
import random
import re
import signal
import socket
import struct
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal

import av
import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from minestudio.simulator import MinecraftSim


PROJECT_DIRECTORY = Path(__file__).resolve().parents[3]
MODEL_NAME = "gemini-3.8-flash"
FRAME_SAMPLE_INTERVAL = 20
MAX_CONTEXT_FRAMES = 12
MAX_BUFFERED_CONTEXT_FRAMES = 48
MAX_TASK_HISTORY = 24
POST_SUCCESS_SETTLE_STEPS = 4
JPEG_QUALITY = 80
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18765
TARGET_FPS = 25
DATASET_DIRECTORY = PROJECT_DIRECTORY / "output" / "coach_dataset"
TRAINING_ACTION_NAMES = [
    "attack",
    "back",
    "forward",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
    "use",
    "drop",
    "inventory",
    *[f"hotbar.{number}" for number in range(1, 10)],
]
ITEM_GROUP_SUFFIXES = {
    "log": ("_log", "_wood", "_stem", "_hyphae"),
    "planks": ("_planks",),
    "leaves": ("_leaves",),
    "sapling": ("_sapling", "_fungus"),
    "wool": ("_wool",),
}

SYSTEM_PROMPT = """
You are the curriculum coach and task compiler for a human playing Minecraft 1.16 in MineStudio.

Your job is to achieve the fixed global_goal supplied by the engine while issuing exactly one concrete executable subtask at a time. Never replace, reinterpret, or wander away from that global goal. Review the player's attempt, update a concise 2-to-5-step prerequisite plan toward it, and issue the next subtask. The strategic_goal field must restate the supplied global goal. Use the supplied chronological gameplay frames, completed-task history, prior plan, global progress, and machine-readable state together. The current_state inventory is the complete current inventory: omitted items have count zero. Current state and deterministic verifier results are authoritative. Older frames can show items or situations that no longer exist.

Choose tasks that are feasible from the visible situation and current state. Work toward meaningful survival milestones such as basic tools, food, shelter, iron, and exploration. Before requesting resources, budget the quantity needed for the next several planned steps and ask for that useful batch in one task. Do not repeatedly request tiny amounts of the same material. Do not craft a shovel, hoe, axe, duplicate tool, or other optional item unless the current plan has an immediate specific use for it. Do not ask the player to pick up an item merely because an older frame shows a drop; after a successful collection verifier, assume it was collected unless current machine state proves otherwise. Do not request materials already sufficiently present in current inventory.

Every executable subtask must be one meaningful, immediately actionable skill-sized unit, normally taking 20 to 120 seconds. A subtask is not a single key press or an arbitrary movement increment. Never issue tasks such as "walk 8 blocks", "turn around", or repeated short travel steps. Never issue an unbounded objective such as "find iron" or "find a cave". When a needed resource is not currently supported by visual or machine state, issue one 45-to-90-second scouting phase using explore_duration, naming what useful terrain or resource signs the player should look for. Reassess only after that scouting phase. Gathering tasks should request the useful batch needed by the forward plan when the material is locally available. If the previous task timed out, explain the likely obstacle briefly and choose a different useful action or scouting direction. If it succeeded, give specific concise feedback and advance by one meaningful step. Never claim that an object is present unless the latest frames or current state support that claim. When visual uncertainty can be resolved from recent history, set context_request.more_frames=true and request up to 48 frames; the engine will call you again with expanded visual context. Do not request more context for inventory or task history because those are already complete.

Useful recipe costs include: 4 planks per log, 4 sticks per 2 planks, 4 planks per crafting table, 3 planks plus 2 sticks per wooden pickaxe, 3 cobblestone plus 2 sticks per stone pickaxe, 8 cobblestone per furnace, and 2 cobblestone plus 1 stick per stone sword. Conserve materials for planned milestones.

Compile every task into exactly one supported verifier:
- explore_duration: source must be player_pos, target must be any, threshold is seconds spent scouting. Use 45 to 90 seconds. Completion also requires meaningful horizontal movement, so instruct the player to scout the nearby terrain rather than wait. This is the only supported verifier for local exploration or resource searching.
- inventory_delta: source must be inventory, target is a lowercase Minecraft item identifier or a supported generic group, threshold is the net quantity gained after assignment. Use this when the instruction requires possessing or collecting an item.
- stat_delta: source must be one of mine_block, craft_item, pickup, or kill_entity; target is a lowercase Minecraft identifier or a supported generic group, threshold is the cumulative event-count increase after assignment. mine_block proves breaking only, not collecting; pickup proves pickup only. Never put both breaking and collecting in an instruction unless one supported verifier alone proves the stated result.

Supported generic material groups are log, planks, leaves, sapling, and wool. Always prefer a generic group when any species, wood type, or color satisfies the goal. Say "collect logs" rather than "collect oak logs" or "collect spruce logs". Use an exact identifier only when that exact subtype is intrinsically required.

The natural-language instruction must state exactly what its verifier proves. Do not use visual judgment, coordinates, vague milestones, compound goals, or unsupported verifier types as completion criteria. The instruction must not mention internal counters, schemas, hidden state, or verifier implementation. Keep feedback under two sentences, the instruction under one sentence, and the hint under one sentence. For inventory_delta and stat_delta subtasks use thresholds from 1 to 16. For explore_duration use 45 to 90 seconds and set max_seconds at least 15 seconds above its threshold. Use max_seconds from 20 to 120.
""".strip()


class VerifierSpec(BaseModel):
    kind: Literal["travel_distance", "explore_duration", "inventory_delta", "stat_delta"]
    source: Literal["player_pos", "inventory", "mine_block", "craft_item", "pickup", "kill_entity"]
    target: str
    threshold: float = Field(ge=1, le=90)


class GlobalGoalSpec(BaseModel):
    goal_id: str
    instruction: str
    verifier: VerifierSpec


class CoachTask(BaseModel):
    instruction: str
    hint: str
    verifier: VerifierSpec
    max_seconds: int = Field(ge=20, le=120)


class ContextRequest(BaseModel):
    more_frames: bool = False
    frame_count: int = Field(default=0, ge=0, le=48)
    reason: str = ""


class CoachResponse(BaseModel):
    feedback: str
    strategic_goal: str
    plan: list[str] = Field(min_length=2, max_length=5)
    context_request: ContextRequest
    task: CoachTask


class ViewerClosed(Exception):
    pass


GLOBAL_GOALS = [
    GlobalGoalSpec(goal_id="collect_logs_8", instruction="Collect 8 logs of any type.", verifier=VerifierSpec(kind="inventory_delta", source="inventory", target="log", threshold=8)),
    GlobalGoalSpec(goal_id="craft_crafting_table", instruction="Craft a crafting table.", verifier=VerifierSpec(kind="stat_delta", source="craft_item", target="crafting_table", threshold=1)),
    GlobalGoalSpec(goal_id="craft_wooden_pickaxe", instruction="Craft a wooden pickaxe.", verifier=VerifierSpec(kind="stat_delta", source="craft_item", target="wooden_pickaxe", threshold=1)),
    GlobalGoalSpec(goal_id="collect_cobblestone_12", instruction="Collect 12 cobblestone.", verifier=VerifierSpec(kind="inventory_delta", source="inventory", target="cobblestone", threshold=12)),
    GlobalGoalSpec(goal_id="craft_stone_pickaxe", instruction="Craft a stone pickaxe.", verifier=VerifierSpec(kind="stat_delta", source="craft_item", target="stone_pickaxe", threshold=1)),
    GlobalGoalSpec(goal_id="craft_stone_sword", instruction="Craft a stone sword.", verifier=VerifierSpec(kind="stat_delta", source="craft_item", target="stone_sword", threshold=1)),
    GlobalGoalSpec(goal_id="craft_stone_shovel", instruction="Craft a stone shovel.", verifier=VerifierSpec(kind="stat_delta", source="craft_item", target="stone_shovel", threshold=1)),
    GlobalGoalSpec(goal_id="craft_furnace", instruction="Craft a furnace.", verifier=VerifierSpec(kind="stat_delta", source="craft_item", target="furnace", threshold=1)),
    GlobalGoalSpec(goal_id="collect_coal_8", instruction="Collect 8 coal.", verifier=VerifierSpec(kind="inventory_delta", source="inventory", target="coal", threshold=8)),
    GlobalGoalSpec(goal_id="obtain_iron_ingots_3", instruction="Obtain 3 iron ingots.", verifier=VerifierSpec(kind="inventory_delta", source="inventory", target="iron_ingot", threshold=3)),
    GlobalGoalSpec(goal_id="explore_16_blocks", instruction="Explore at least 16 horizontal blocks from spawn.", verifier=VerifierSpec(kind="travel_distance", source="player_pos", target="any", threshold=16)),
]


class DemonstrationRecorder:
    def __init__(self, first_frame: np.ndarray, world_seed: int, global_goal: GlobalGoalSpec) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        self.session_directory = DATASET_DIRECTORY / self.session_id
        self.session_directory.mkdir(parents=True, exist_ok=False)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_monotonic = time.monotonic()
        self.world_seed = world_seed
        self.global_goal = global_goal
        self.global_progress = 0.0
        self.global_succeeded = False
        self.frame_index = 0
        self.task_count = 0
        self.active_task = None
        self.closed = False
        self.video_path = self.session_directory / f"{self.session_id}.mp4"
        self.action_path = self.session_directory / f"{self.session_id}_action.json"
        self.minestudio_action_path = self.session_directory / f"{self.session_id}.pkl"
        self.info_path = self.session_directory / f"{self.session_id}_info.json"
        self.steps_path = self.session_directory / "steps.jsonl"
        self.tasks_path = self.session_directory / "tasks.jsonl"
        self.video_container = av.open(self.video_path, mode="w", format="mp4")
        self.video_stream = self.video_container.add_stream("h264", rate=TARGET_FPS)
        self.video_stream.width = int(first_frame.shape[1])
        self.video_stream.height = int(first_frame.shape[0])
        self.video_stream.pix_fmt = "yuv420p"
        self.action_file = self.action_path.open("w", encoding="utf-8")
        self.info_file = self.info_path.open("w", encoding="utf-8")
        self.steps_file = self.steps_path.open("w", encoding="utf-8")
        self.tasks_file = self.tasks_path.open("w", encoding="utf-8")
        self.action_file.write("[")
        self.info_file.write("[")
        self.write_session_metadata("recording")

    def json_value(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): self.json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.json_value(item) for item in value]
        return value

    def write_session_metadata(self, status: str, close_reason: str = "") -> None:
        metadata = {
            "format": "minestudio_gemini_coach",
            "format_version": 1,
            "session_id": self.session_id,
            "status": status,
            "close_reason": close_reason,
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat() if status == "complete" else None,
            "model": MODEL_NAME,
            "world_seed": self.world_seed,
            "global_goal": self.global_goal.model_dump(),
            "global_progress": self.global_progress,
            "global_succeeded": self.global_succeeded,
            "fps": TARGET_FPS,
            "resolution": [self.video_stream.width, self.video_stream.height],
            "frame_action_alignment": "frame_t_is_observation_before_action_t",
            "total_frames": self.frame_index,
            "total_tasks": self.task_count,
            "files": {
                "video": self.video_path.name,
                "actions": self.action_path.name,
                "minestudio_actions": self.minestudio_action_path.name,
                "infos": self.info_path.name,
                "steps": self.steps_path.name,
                "tasks": self.tasks_path.name,
            },
        }
        temporary_path = self.session_directory / "session.json.tmp"
        temporary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary_path, self.session_directory / "session.json")

    def start_task(self, response: CoachResponse, baseline: Dict) -> None:
        task_id = f"task_{self.task_count:04d}"
        self.active_task = {
            "task_id": task_id,
            "start_frame": self.frame_index,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "started_monotonic": time.monotonic(),
            "feedback": response.feedback,
            "global_goal_id": self.global_goal.goal_id,
            "strategic_goal": response.strategic_goal,
            "plan": response.plan,
            "instruction": response.task.instruction,
            "hint": response.task.hint,
            "verifier": response.task.verifier.model_dump(),
            "max_seconds": response.task.max_seconds,
            "baseline_state": baseline,
        }

    def record_step(
        self,
        frame: np.ndarray,
        action: Dict,
        state_before: Dict,
        state_after: Dict,
        progress_before: float,
        progress_after: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        global_progress_before: float,
        global_progress_after: float,
    ) -> None:
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self.video_stream.encode(video_frame):
            self.video_container.mux(packet)
        training_action = {name: action.get(name, 0) for name in TRAINING_ACTION_NAMES}
        training_action["camera"] = action["camera"]
        serial_action = self.json_value(training_action)
        info_record = {
            "frame_id": self.frame_index,
            "timestamp_seconds": time.monotonic() - self.started_monotonic,
            "task_id": self.active_task["task_id"],
            "state_before": state_before,
            "state_after": state_after,
            "progress_before": progress_before,
            "progress_after": progress_after,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "global_goal_id": self.global_goal.goal_id,
            "global_progress_before": float(global_progress_before),
            "global_progress_after": float(global_progress_after),
            "global_threshold": self.global_goal.verifier.threshold,
        }
        separator = "" if self.frame_index == 0 else ","
        self.action_file.write(separator + json.dumps(serial_action, separators=(",", ":")))
        self.info_file.write(separator + json.dumps(info_record, separators=(",", ":")))
        step_record = {"action": serial_action, **info_record}
        self.steps_file.write(json.dumps(step_record, separators=(",", ":")) + "\n")
        self.global_progress = float(global_progress_after)
        self.global_succeeded = self.global_progress >= self.global_goal.verifier.threshold
        self.frame_index += 1
        if self.frame_index % TARGET_FPS == 0:
            self.action_file.flush()
            self.info_file.flush()
            self.steps_file.flush()
            self.write_session_metadata("recording")

    def finish_task(self, outcome: str, final_state: Dict, progress: float) -> None:
        if self.active_task is None:
            return
        task_record = dict(self.active_task)
        started_monotonic = task_record.pop("started_monotonic")
        task_record.update(
            {
                "end_frame": self.frame_index,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "outcome": outcome,
                "succeeded": outcome == "succeeded",
                "final_progress": float(progress),
                "final_state": final_state,
            }
        )
        self.tasks_file.write(json.dumps(task_record, separators=(",", ":")) + "\n")
        self.tasks_file.flush()
        self.task_count += 1
        self.active_task = None
        self.write_session_metadata("recording")

    def close(self, close_reason: str, final_state: Dict, progress: float, global_progress: float, global_succeeded: bool) -> None:
        if self.closed:
            return
        if self.active_task is not None:
            self.finish_task(close_reason, final_state, progress)
        self.global_progress = float(global_progress)
        self.global_succeeded = bool(global_succeeded)
        for packet in self.video_stream.encode():
            self.video_container.mux(packet)
        self.video_container.close()
        self.action_file.write("]")
        self.info_file.write("]")
        self.action_file.close()
        self.info_file.close()
        actions = json.loads(self.action_path.read_text(encoding="utf-8"))
        action_arrays = {
            name: np.asarray([action[name] for action in actions], dtype=np.float32 if name == "camera" else np.uint8)
            for name in [*TRAINING_ACTION_NAMES, "camera"]
        }
        with self.minestudio_action_path.open("wb") as action_pickle_file:
            pickle.dump(action_arrays, action_pickle_file)
        self.steps_file.close()
        self.tasks_file.close()
        self.closed = True
        self.write_session_metadata("complete", close_reason)
        print(f"Dataset saved to {self.session_directory}")


def normalize_identifier(value: str) -> str:
    return value.lower().replace("minecraft:", "").replace(" ", "_")


def select_global_goal(world_seed: int) -> GlobalGoalSpec:
    return random.Random(world_seed).choice(GLOBAL_GOALS).model_copy(deep=True)


def grouped_value(values: Dict[str, float], target: str) -> float:
    normalized_target = normalize_identifier(target)
    suffixes = ITEM_GROUP_SUFFIXES.get(normalized_target)
    if suffixes is None:
        return values.get(normalized_target, 0.0)
    return sum(value for name, value in values.items() if name.endswith(suffixes))


def inventory_counts(info: Dict) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    for stack in info.get("inventory", {}).values():
        item_name = normalize_identifier(str(stack.get("type", "air")))
        quantity = float(stack.get("quantity", 0))
        if item_name not in {"air", "none"} and quantity > 0:
            counts[item_name] = counts.get(item_name, 0.0) + quantity
    return counts


def numeric_mapping(info: Dict, source: str) -> Dict[str, float]:
    values = info.get(source, {})
    return {
        normalize_identifier(str(name)): float(np.asarray(value))
        for name, value in values.items()
        if float(np.asarray(value)) != 0.0
    }


def state_snapshot(info: Dict) -> Dict:
    player_pos = info.get("player_pos", {})
    return {
        "inventory": inventory_counts(info),
        "mine_block": numeric_mapping(info, "mine_block"),
        "craft_item": numeric_mapping(info, "craft_item"),
        "pickup": numeric_mapping(info, "pickup"),
        "kill_entity": numeric_mapping(info, "kill_entity"),
        "player_pos": {
            "x": float(player_pos.get("x", 0.0)),
            "y": float(player_pos.get("y", 0.0)),
            "z": float(player_pos.get("z", 0.0)),
        },
        "health": float(info.get("health", 0.0)),
        "food_level": float(info.get("food_level", 0.0)),
    }


def validate_verifier(verifier: VerifierSpec) -> None:
    expected_sources = {
        "travel_distance": {"player_pos"},
        "explore_duration": {"player_pos"},
        "inventory_delta": {"inventory"},
        "stat_delta": {"mine_block", "craft_item", "pickup", "kill_entity"},
    }
    if verifier.source not in expected_sources[verifier.kind]:
        raise ValueError(f"Invalid source {verifier.source} for {verifier.kind}")
    if verifier.kind not in {"travel_distance", "explore_duration"} and normalize_identifier(verifier.target) in {"", "any", "none"}:
        raise ValueError(f"Invalid target for {verifier.kind}")


def validate_task(task: CoachTask) -> None:
    validate_verifier(task.verifier)
    if task.verifier.kind == "travel_distance":
        raise ValueError("Short travel-distance subtasks are not allowed; use a meaningful exploration phase")
    if task.verifier.kind == "explore_duration":
        if task.verifier.threshold < 45:
            raise ValueError("Exploration phases must last at least 45 seconds")
        if task.max_seconds < task.verifier.threshold + 15:
            raise ValueError("Exploration max_seconds must leave time for movement beyond its duration threshold")
    elif task.verifier.threshold > 16:
        raise ValueError("Inventory and event subtasks cannot exceed a useful batch of 16")
    instruction = task.instruction.lower()
    forbidden_phrases = ("turn around", "return to the surface", "reach the surface")
    numbered_walk = re.search(r"\b(?:walk|travel|move)\s+(?:about\s+)?\d+\s+blocks?\b", instruction)
    if numbered_walk or any(phrase in instruction for phrase in forbidden_phrases):
        raise ValueError(f"Subtask is an unsupported micro-navigation goal: {task.instruction}")


def verifier_progress(
    verifier: VerifierSpec,
    baseline: Dict,
    current: Dict,
    elapsed_seconds: float = 0.0,
    cumulative_horizontal_distance: float = 0.0,
) -> tuple[float, bool]:
    if verifier.kind == "travel_distance":
        delta_x = current["player_pos"]["x"] - baseline["player_pos"]["x"]
        delta_z = current["player_pos"]["z"] - baseline["player_pos"]["z"]
        progress = math.hypot(delta_x, delta_z)
    elif verifier.kind == "explore_duration":
        minimum_distance = max(8.0, verifier.threshold / 5.0)
        time_fraction = elapsed_seconds / verifier.threshold
        movement_fraction = cumulative_horizontal_distance / minimum_distance
        progress = min(time_fraction, movement_fraction, 1.0) * verifier.threshold
        return progress, elapsed_seconds >= verifier.threshold and cumulative_horizontal_distance >= minimum_distance
    else:
        target = normalize_identifier(verifier.target)
        progress = grouped_value(current[verifier.source], target) - grouped_value(baseline[verifier.source], target)
    return progress, progress >= verifier.threshold


def encode_frame(image: np.ndarray) -> bytes:
    blue_green_red = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    encode_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    success, encoded = cv2.imencode(".jpg", blue_green_red, encode_parameters)
    if not success:
        raise RuntimeError("Could not encode a Minecraft frame")
    return encoded.tobytes()


def request_task(
    client: genai.Client,
    outcome: str,
    state: Dict,
    frames: deque[bytes],
    completed_tasks: deque[Dict],
    prior_plan: list[str],
    global_goal: GlobalGoalSpec,
    global_progress: float,
) -> CoachResponse:
    available_frames = list(frames)
    selected_frames = available_frames[-MAX_CONTEXT_FRAMES:]
    context_delivery = "default_recent_context"
    for request_round in range(2):
        request_text = json.dumps(
            {
                "event": outcome,
                "current_state": state,
                "completed_tasks": list(completed_tasks),
                "prior_plan": prior_plan,
                "global_goal": global_goal.model_dump(),
                "global_progress": global_progress,
                "frame_order": "oldest_to_newest",
                "visual_context": {
                    "delivery": context_delivery,
                    "frames_sent": len(selected_frames),
                    "frames_available": len(available_frames),
                },
                "request": "Review the attempt, update the forward plan, and issue exactly one next task.",
            },
            separators=(",", ":"),
        )
        contents = [types.Part.from_text(text=request_text)]
        contents.extend(types.Part.from_bytes(data=frame, mime_type="image/jpeg") for frame in selected_frames)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CoachResponse,
                max_output_tokens=2048,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        coach_response = CoachResponse.model_validate_json(response.text)
        validate_task(coach_response.task)
        requested_count = coach_response.context_request.frame_count or MAX_BUFFERED_CONTEXT_FRAMES
        expanded_count = min(max(requested_count, MAX_CONTEXT_FRAMES + 1), MAX_BUFFERED_CONTEXT_FRAMES, len(available_frames))
        can_expand = expanded_count > len(selected_frames)
        if not coach_response.context_request.more_frames or not can_expand or request_round == 1:
            return coach_response
        selected_frames = available_frames[-expanded_count:]
        context_delivery = f"expanded_context_for: {coach_response.context_request.reason}"
    raise RuntimeError("Gemini context request loop ended without a task")


def request_task_resilient(
    client: genai.Client,
    outcome: str,
    state: Dict,
    frames: deque[bytes],
    completed_tasks: deque[Dict],
    prior_plan: list[str],
    global_goal: GlobalGoalSpec,
    global_progress: float,
) -> CoachResponse:
    try:
        return request_task(client, outcome, state, frames, completed_tasks, prior_plan, global_goal, global_progress)
    except (httpx.HTTPError, errors.APIError, ValidationError, ValueError) as error:
        print(f"Gemini unavailable ({type(error).__name__}); continuing with a local fallback task.")
        return CoachResponse(
            feedback="Gemini is temporarily unavailable, so recording will continue with a safe local task.",
            strategic_goal=global_goal.instruction,
            plan=["Continue making safe progress", "Resume the goal plan when Gemini reconnects"],
            context_request=ContextRequest(),
            task=CoachTask(
                instruction="Scout the nearby terrain for 60 seconds while continuing toward the global goal.",
                hint="Move through safe visible terrain and look for resources or landmarks relevant to the goal.",
                verifier=VerifierSpec(kind="explore_duration", source="player_pos", target="any", threshold=60),
                max_seconds=90,
            ),
        )


def task_overlay(response: CoachResponse, progress: float) -> str:
    threshold = response.task.verifier.threshold
    return f"TASK: {response.task.instruction} | {progress:.1f}/{threshold:g} | HINT: {response.task.hint}"


def receive_exact(connection: socket.socket, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = connection.recv(byte_count - len(chunks))
        if not chunk:
            raise ViewerClosed()
        chunks.extend(chunk)
    return bytes(chunks)


def receive_action(connection: socket.socket) -> Dict:
    payload_size = struct.unpack("!I", receive_exact(connection, 4))[0]
    payload = json.loads(receive_exact(connection, payload_size))
    if payload.get("shutdown"):
        raise ViewerClosed()
    return payload["action"]


def send_frame(connection: socket.socket, image: np.ndarray, payload: Dict) -> Dict:
    header = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    jpeg = encode_frame(image)
    connection.sendall(struct.pack("!I", len(header)) + header + struct.pack("!I", len(jpeg)) + jpeg)
    return receive_action(connection)


def environment_action(sim: MinecraftSim, viewer_action: Dict) -> Dict:
    action = sim.noop_action()
    for name, value in viewer_action.items():
        if name not in action:
            continue
        if name == "camera":
            action[name] = np.asarray(value, dtype=np.float32)
        else:
            action[name] = int(value)
    return action


def viewer_payload(
    response: CoachResponse,
    progress: float,
    state: Dict,
    status: str,
    recorder: DemonstrationRecorder,
    global_goal: GlobalGoalSpec,
    global_progress: float,
    session_complete: bool = False,
) -> Dict:
    return {
        "status": status,
        "recording": True,
        "session_id": recorder.session_id,
        "recorded_frames": recorder.frame_index,
        "feedback": response.feedback,
        "strategic_goal": response.strategic_goal,
        "plan": response.plan,
        "global_goal": global_goal.instruction,
        "global_progress": global_progress,
        "global_threshold": global_goal.verifier.threshold,
        "session_complete": session_complete,
        "instruction": response.task.instruction,
        "hint": response.task.hint,
        "progress": progress,
        "threshold": response.task.verifier.threshold,
        "health": state["health"],
        "food_level": state["food_level"],
        "player_pos": state["player_pos"],
    }


def main() -> None:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    load_dotenv(PROJECT_DIRECTORY / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY is missing from {PROJECT_DIRECTORY / '.env'}")

    client = genai.Client(api_key=api_key)
    world_seed = int.from_bytes(os.urandom(4), byteorder="big")
    global_goal = select_global_goal(world_seed)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((SERVER_HOST, SERVER_PORT))
    listener.listen(1)
    print(f"Waiting for the Windows viewer on port {SERVER_PORT}...")
    connection, viewer_address = listener.accept()
    print(f"Windows viewer connected from {viewer_address[0]}")
    sim = MinecraftSim(
        obs_size=(640, 360),
        action_type="env",
        seed=world_seed,
        callbacks=[],
    )
    recorder = None
    current_state = {}
    progress = 0.0
    global_progress = 0.0
    global_succeeded = False
    close_reason = "session_closed"

    try:
        obs, info = sim.reset()
        frames: deque[bytes] = deque(maxlen=MAX_BUFFERED_CONTEXT_FRAMES)
        completed_tasks: deque[Dict] = deque(maxlen=MAX_TASK_HISTORY)
        frames.append(encode_frame(info["pov"]))
        current_state = state_snapshot(info)
        global_baseline = current_state
        response = request_task_resilient(
            client,
            "session_start",
            current_state,
            frames,
            completed_tasks,
            [],
            global_goal,
            global_progress,
        )
        baseline = current_state
        recorder = DemonstrationRecorder(info["pov"], world_seed, global_goal)
        recorder.start_task(response, baseline)
        task_started = time.monotonic()
        cumulative_horizontal_distance = 0.0
        step_number = 0
        progress = 0.0
        print(f"Global goal: {global_goal.instruction}")
        print(f"\nGemini: {response.feedback}\nTask: {response.task.instruction}\nHint: {response.task.hint}\n")
        terminated = False
        truncated = False

        while not (terminated or truncated):
            frame_started = time.monotonic()
            payload = viewer_payload(response, progress, current_state, "playing", recorder, global_goal, global_progress)
            frame_before = info["pov"]
            state_before = current_state
            progress_before = progress
            global_progress_before = global_progress
            viewer_action = send_frame(connection, info["pov"], payload)
            action = environment_action(sim, viewer_action)
            obs, reward, terminated, truncated, info = sim.step(action)
            step_number += 1
            if step_number % FRAME_SAMPLE_INTERVAL == 0:
                frames.append(encode_frame(info["pov"]))
            current_state = state_snapshot(info)
            step_distance = math.hypot(
                current_state["player_pos"]["x"] - state_before["player_pos"]["x"],
                current_state["player_pos"]["z"] - state_before["player_pos"]["z"],
            )
            cumulative_horizontal_distance += min(step_distance, 2.0)
            task_elapsed = time.monotonic() - task_started
            progress, succeeded = verifier_progress(
                response.task.verifier,
                baseline,
                current_state,
                task_elapsed,
                cumulative_horizontal_distance,
            )
            global_progress, global_succeeded = verifier_progress(global_goal.verifier, global_baseline, current_state)
            timed_out = task_elapsed >= response.task.max_seconds
            recorder.record_step(
                frame_before,
                action,
                state_before,
                current_state,
                progress_before,
                progress,
                reward,
                terminated,
                truncated,
                global_progress_before,
                global_progress,
            )

            if global_succeeded or succeeded or timed_out:
                if global_succeeded or succeeded:
                    for _ in range(POST_SUCCESS_SETTLE_STEPS):
                        if terminated or truncated:
                            break
                        settle_frame = info["pov"]
                        settle_state = current_state
                        settle_progress = progress
                        settle_global_progress = global_progress
                        settle_action = sim.noop_action()
                        obs, reward, terminated, truncated, info = sim.step(settle_action)
                        current_state = state_snapshot(info)
                        progress, _ = verifier_progress(
                            response.task.verifier,
                            baseline,
                            current_state,
                            time.monotonic() - task_started,
                            cumulative_horizontal_distance,
                        )
                        global_progress, global_succeeded = verifier_progress(global_goal.verifier, global_baseline, current_state)
                        recorder.record_step(
                            settle_frame,
                            settle_action,
                            settle_state,
                            current_state,
                            settle_progress,
                            progress,
                            reward,
                            terminated,
                            truncated,
                            settle_global_progress,
                            global_progress,
                        )
                task_outcome = "succeeded" if succeeded else "global_goal_completed" if global_succeeded else "timed_out"
                recorder.finish_task(task_outcome, current_state, progress)
                completed_tasks.append(
                    {
                        "instruction": response.task.instruction,
                        "verifier": response.task.verifier.model_dump(),
                        "outcome": task_outcome,
                        "final_progress": float(progress),
                        "inventory_after": current_state["inventory"],
                    }
                )
                if global_succeeded:
                    completion_payload = viewer_payload(
                        response,
                        progress,
                        current_state,
                        "Global goal completed. Saving session...",
                        recorder,
                        global_goal,
                        global_progress,
                        session_complete=True,
                    )
                    send_frame(connection, info["pov"], completion_payload)
                    close_reason = "global_goal_succeeded"
                    break
                if terminated or truncated:
                    break
                frames.append(encode_frame(info["pov"]))
                outcome = "previous_task_succeeded" if succeeded else "previous_task_timed_out"
                outcome += f": {response.task.instruction}; final_progress={progress:.2f}/{response.task.verifier.threshold:g}"
                reviewing_payload = viewer_payload(
                    response,
                    progress,
                    current_state,
                    "Gemini is reviewing your attempt...",
                    recorder,
                    global_goal,
                    global_progress,
                )
                send_frame(connection, info["pov"], reviewing_payload)
                response = request_task_resilient(
                    client,
                    outcome,
                    current_state,
                    frames,
                    completed_tasks,
                    response.plan,
                    global_goal,
                    global_progress,
                )
                baseline = current_state
                recorder.start_task(response, baseline)
                task_started = time.monotonic()
                cumulative_horizontal_distance = 0.0
                progress = 0.0
                print(f"\nGemini: {response.feedback}\nTask: {response.task.instruction}\nHint: {response.task.hint}\n")
            remaining_time = 1.0 / TARGET_FPS - (time.monotonic() - frame_started)
            time.sleep(max(0.0, remaining_time))
        if not global_succeeded:
            close_reason = "environment_terminated" if terminated else "environment_truncated"
    except KeyboardInterrupt:
        close_reason = "keyboard_interrupt"
        print("\nMineStudio Gemini Coach stopped.")
    except ViewerClosed:
        close_reason = "viewer_closed"
        print("\nMineStudio Gemini Coach stopped.")
    except (ConnectionError, BrokenPipeError):
        close_reason = "connection_lost"
        print("\nMineStudio Gemini Coach stopped.")
    except Exception as error:
        close_reason = f"error_{type(error).__name__}"
        raise
    finally:
        if recorder is not None:
            recorder.close(close_reason, current_state, progress, global_progress, global_succeeded)
        sim.close()
        connection.close()
        listener.close()


if __name__ == "__main__":
    main()
