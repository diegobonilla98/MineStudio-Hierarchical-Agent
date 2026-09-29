import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import av
import cv2
import httpx
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from closed_loop_options import ClosedLoopOptions, grouped_inventory_value
from minestudio.models import SteveOnePolicy
from minestudio.simulator import MinecraftSim
from pydantic import BaseModel, Field, ValidationError


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = PROJECT_DIRECTORY / "checkpoints" / "steve_one_official"
OUTPUT_ROOT = PROJECT_DIRECTORY / "output" / "steve_hierarchy"
EXPERIMENT_SUFFIX = "stone_pickaxe_options_gated"
MODEL_NAME = "gemini-3.8-flash"
GLOBAL_OBJECTIVE = "Craft a stone pickaxe."
WORLD_SEED = 20260915
CONDITION_SCALE = 6.0
DETERMINISTIC_TEXT_PRIOR = False
DETERMINISTIC_ACTIONS = False
MAX_TOTAL_STEPS = 6000
MAX_ATTEMPTS = 16
FRAME_SAMPLE_INTERVAL = 40
MAX_CONTEXT_FRAMES = 12
JPEG_QUALITY = 80
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
VIDEO_FPS = 20
SKILL_LIBRARY = {
    "collect_log": {
        "executor_prompt": "chop down the tree, gather wood, pick up wood, chop it down, break tree",
        "verifier_kind": "inventory_delta",
        "source": "inventory",
        "target": "log",
        "threshold": 1,
        "default_steps": 1000,
        "option": "MINE_AND_COLLECT",
    },
    "pickup_log": {
        "executor_prompt": "pick up the wood",
        "verifier_kind": "inventory_delta",
        "source": "inventory",
        "target": "log",
        "threshold": 1,
        "default_steps": 400,
        "option": "COLLECT_DROP",
    },
    "explore": {
        "executor_prompt": "go explore",
        "verifier_kind": "travel_distance",
        "source": "player_pos",
        "target": "any",
        "threshold": 8,
        "default_steps": 600,
        "option": "STEVE_1",
    },
    "craft_planks": {
        "executor_prompt": "craft wooden planks",
        "verifier_kind": "stat_delta",
        "source": "craft_item",
        "target": "planks",
        "threshold": 4,
        "default_steps": 600,
        "option": "CRAFT_RECIPE",
    },
    "craft_sticks": {
        "executor_prompt": "craft sticks",
        "verifier_kind": "stat_delta",
        "source": "craft_item",
        "target": "stick",
        "threshold": 2,
        "default_steps": 600,
        "option": "CRAFT_RECIPE",
    },
    "craft_crafting_table": {
        "executor_prompt": "craft a crafting table",
        "verifier_kind": "stat_delta",
        "source": "craft_item",
        "target": "crafting_table",
        "threshold": 1,
        "default_steps": 600,
        "option": "CRAFT_RECIPE",
    },
    "place_crafting_table": {
        "executor_prompt": "place the crafting table",
        "verifier_kind": "stat_delta",
        "source": "place_block",
        "target": "crafting_table",
        "threshold": 1,
        "default_steps": 600,
        "option": "PLACE_BLOCK",
    },
    "craft_wooden_pickaxe": {
        "executor_prompt": "craft a wooden pickaxe",
        "verifier_kind": "stat_delta",
        "source": "craft_item",
        "target": "wooden_pickaxe",
        "threshold": 1,
        "default_steps": 800,
        "option": "CRAFT_RECIPE",
    },
    "mine_cobblestone": {
        "executor_prompt": "mine stone and collect cobblestone",
        "verifier_kind": "inventory_delta",
        "source": "inventory",
        "target": "cobblestone",
        "threshold": 3,
        "default_steps": 1000,
        "option": "MINE_AND_COLLECT",
    },
    "craft_stone_pickaxe": {
        "executor_prompt": "craft a stone pickaxe",
        "verifier_kind": "stat_delta",
        "source": "craft_item",
        "target": "stone_pickaxe",
        "threshold": 1,
        "default_steps": 1000,
        "option": "CRAFT_RECIPE",
    },
}
ITEM_GROUP_SUFFIXES = {
    "log": ("_log", "_wood", "_stem", "_hyphae", "hyphae"),
    "planks": ("_planks",),
    "leaves": ("_leaves",),
    "sapling": ("_sapling", "_fungus"),
    "wool": ("_wool",),
}


class PlannerDecision(BaseModel):
    global_objective: Literal["Craft a stone pickaxe."]
    local_objective: str = Field(min_length=3, max_length=160)
    executor_skill: Literal[
        "collect_log",
        "pickup_log",
        "explore",
        "craft_planks",
        "craft_sticks",
        "craft_crafting_table",
        "place_crafting_table",
        "craft_wooden_pickaxe",
        "mine_cobblestone",
        "craft_stone_pickaxe",
    ]
    failure_category: Literal[
        "not_applicable",
        "execution_or_recovery",
        "conditioning_or_skill_understanding",
        "environment_or_precondition",
        "inconclusive",
    ]
    evidence: str = Field(min_length=3, max_length=300)
    forward_plan: list[str] = Field(min_length=1, max_length=6)
    max_steps: int = Field(ge=200, le=1200)


def planner_system_prompt() -> str:
    library = {
        skill_id: {
            "executor_prompt": value["executor_prompt"],
            "execution_option": value["option"],
            "verifier": {
                "kind": value["verifier_kind"],
                "source": value["source"],
                "target": value["target"],
                "threshold": value["threshold"],
            },
            "default_steps": value["default_steps"],
        }
        for skill_id, value in SKILL_LIBRARY.items()
    }
    return f"""
You are the low-frequency multimodal planner for a Minecraft 1.16 agent. The fixed global objective is exactly: {GLOBAL_OBJECTIVE}

The executor uses released STEVE-1 for fuzzy visual world behavior and verifier-driven closed-loop options for deterministic tails and GUI mechanics. MINE_AND_COLLECT uses STEVE-1 to approach and break a target, then performs local pickup recovery until inventory changes. CRAFT_RECIPE visually controls the real Minecraft crafting GUI. PLACE_BLOCK equips the requested item and places it with state verification. You never issue keyboard or mouse actions. Select exactly one feasible local objective at a time from the provided skill library. The engine, not you, supplies the exact executor prompt or option and verifies success from Minecraft state.

Use the current inventory, cumulative events, attempt history, and chronological frames. Machine state and deterministic verifier outcomes are authoritative. Do not claim that an object is visible unless the recent frames support it. Do not select a crafting step before its material and interface preconditions. Minecraft recipe facts: one log can make four planks; two planks can make four sticks; a crafting table costs four planks; a wooden pickaxe costs three planks and two sticks; a stone pickaxe costs three cobblestone and two sticks; pickaxes require a placed crafting table; stone requires a pickaxe.

Preserve the global objective. local_objective explains the immediate result needed. executor_skill selects a fine-grained prompt from the library. forward_plan is a concise prerequisite chain beginning with the selected local objective. On session_start use failure_category not_applicable. After a timeout, classify the failed attempt from exact events, action counts, movement, and frames: execution_or_recovery means the prompt was aligned but motor execution or follow-through failed; conditioning_or_skill_understanding means behavior was unrelated or the selected skill/prompt was wrong; environment_or_precondition means the needed target/material/interface was unavailable; inconclusive means evidence is insufficient. A failure classification is diagnostic, never a success signal. Change skill after a timeout when repeating it would not address the evidence.

Skill library:
{json.dumps(library, indent=2)}
""".strip()


def normalize_identifier(value: str) -> str:
    return value.lower().replace("minecraft:", "").replace(" ", "_")


def grouped_value(values: dict[str, float], target: str) -> float:
    normalized_target = normalize_identifier(target)
    suffixes = ITEM_GROUP_SUFFIXES.get(normalized_target)
    if suffixes is None:
        return values.get(normalized_target, 0.0)
    return sum(value for name, value in values.items() if name.endswith(suffixes))


def numeric_mapping(info: dict, source: str) -> dict[str, float]:
    values = info.get(source, {})
    return {
        normalize_identifier(str(name)): float(np.asarray(value))
        for name, value in values.items()
        if float(np.asarray(value)) != 0.0
    }


def inventory_counts(info: dict) -> dict[str, float]:
    counts: dict[str, float] = {}
    for stack in info.get("inventory", {}).values():
        item_name = normalize_identifier(str(stack.get("type", "air")))
        quantity = float(stack.get("quantity", 0))
        if item_name not in {"air", "none"} and quantity > 0:
            counts[item_name] = counts.get(item_name, 0.0) + quantity
    return counts


def state_snapshot(info: dict) -> dict:
    player_pos = info.get("player_pos", {})
    return {
        "inventory": inventory_counts(info),
        "mine_block": numeric_mapping(info, "mine_block"),
        "craft_item": numeric_mapping(info, "craft_item"),
        "pickup": numeric_mapping(info, "pickup"),
        "place_block": numeric_mapping(info, "place_block"),
        "kill_entity": numeric_mapping(info, "kill_entity"),
        "player_pos": {
            "x": float(player_pos.get("x", 0.0)),
            "y": float(player_pos.get("y", 0.0)),
            "z": float(player_pos.get("z", 0.0)),
        },
        "health": float(info.get("health", 0.0)),
        "food_level": float(info.get("food_level", 0.0)),
    }


def verifier_progress(skill: dict, baseline: dict, current: dict) -> tuple[float, bool]:
    if skill["verifier_kind"] == "travel_distance":
        delta_x = current["player_pos"]["x"] - baseline["player_pos"]["x"]
        delta_z = current["player_pos"]["z"] - baseline["player_pos"]["z"]
        value = math.hypot(delta_x, delta_z)
    else:
        value = grouped_value(current[skill["source"]], skill["target"]) - grouped_value(baseline[skill["source"]], skill["target"])
    return value, value >= skill["threshold"]


def global_success(global_baseline: dict, current: dict) -> tuple[float, bool]:
    crafted = grouped_value(current["craft_item"], "stone_pickaxe") - grouped_value(global_baseline["craft_item"], "stone_pickaxe")
    inventory = grouped_value(current["inventory"], "stone_pickaxe")
    value = max(crafted, inventory)
    return value, value >= 1


def encode_frame(image: np.ndarray) -> bytes:
    blue_green_red = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", blue_green_red, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not success:
        raise RuntimeError("Could not encode frame")
    return encoded.tobytes()


def fallback_decision(state: dict, outcome: str) -> PlannerDecision:
    inventory = state["inventory"]
    crafted = state["craft_item"]
    placed = state["place_block"]
    if grouped_value(state["mine_block"], "log") >= 1 and grouped_value(inventory, "log") < 1:
        skill_id = "pickup_log"
        local_objective = "Pick up a dropped log."
    elif grouped_value(inventory, "log") < 1 and grouped_value(crafted, "planks") < 4:
        skill_id = "collect_log"
        local_objective = "Collect one log."
    elif grouped_value(inventory, "planks") < 4 and grouped_value(crafted, "crafting_table") < 1:
        skill_id = "craft_planks"
        local_objective = "Craft wooden planks from the collected log."
    elif grouped_value(crafted, "crafting_table") < 1 and grouped_value(inventory, "crafting_table") < 1:
        skill_id = "craft_crafting_table"
        local_objective = "Craft a crafting table."
    elif grouped_value(inventory, "stick") < 2 and grouped_value(crafted, "stick") < 2:
        skill_id = "craft_sticks"
        local_objective = "Craft at least two sticks."
    elif grouped_value(placed, "crafting_table") < 1:
        skill_id = "place_crafting_table"
        local_objective = "Place the crafting table."
    elif grouped_value(crafted, "wooden_pickaxe") < 1 and grouped_value(inventory, "wooden_pickaxe") < 1:
        skill_id = "craft_wooden_pickaxe"
        local_objective = "Craft a wooden pickaxe."
    elif grouped_value(inventory, "cobblestone") < 3:
        skill_id = "mine_cobblestone"
        local_objective = "Collect three cobblestone."
    else:
        skill_id = "craft_stone_pickaxe"
        local_objective = "Craft a stone pickaxe."
    return PlannerDecision(
        global_objective=GLOBAL_OBJECTIVE,
        local_objective=local_objective,
        executor_skill=skill_id,
        failure_category="not_applicable" if outcome == "session_start" else "inconclusive",
        evidence="Local fallback selected from exact inventory and event state.",
        forward_plan=[local_objective, GLOBAL_OBJECTIVE],
        max_steps=SKILL_LIBRARY[skill_id]["default_steps"],
    )


def request_decision(client: genai.Client, outcome: str, state: dict, frames: deque[bytes], attempts: list[dict]) -> tuple[PlannerDecision, str | None]:
    request = {
        "event": outcome,
        "global_objective": GLOBAL_OBJECTIVE,
        "current_state": state,
        "attempt_history": attempts[-8:],
        "frame_order": "oldest_to_newest",
        "frames_sent": len(frames),
        "request": "Diagnose the prior attempt if present, preserve the global goal, and select the next local objective and executor skill.",
    }
    contents = [types.Part.from_text(text=json.dumps(request, separators=(",", ":")))]
    contents.extend(types.Part.from_bytes(data=frame, mime_type="image/jpeg") for frame in frames)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=planner_system_prompt(),
                response_mime_type="application/json",
                response_schema=PlannerDecision,
                max_output_tokens=2048,
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        decision = PlannerDecision.model_validate_json(response.text)
        return decision, None
    except (httpx.HTTPError, errors.APIError, ValidationError, ValueError) as error:
        return fallback_decision(state, outcome), f"{type(error).__name__}: {error}"


def create_video(path: Path) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream]:
    container = av.open(path, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=VIDEO_FPS)
    stream.width = RENDER_SIZE[0]
    stream.height = RENDER_SIZE[1]
    stream.pix_fmt = "yuv420p"
    return container, stream


def write_video_frame(container: av.container.OutputContainer, stream: av.video.stream.VideoStream, frame: np.ndarray) -> None:
    video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def overlay_frame(frame: np.ndarray, decision: PlannerDecision, skill: dict, attempt_index: int, total_step: int, value: float) -> np.ndarray:
    image = np.ascontiguousarray(frame.copy())
    lines = [
        f"GLOBAL: {GLOBAL_OBJECTIVE}",
        f"LOCAL: {decision.local_objective}",
        f"{skill['option']}: {skill['executor_prompt']}",
        f"attempt {attempt_index + 1}/{MAX_ATTEMPTS}  total step {total_step}/{MAX_TOTAL_STEPS}  progress {value:.1f}/{skill['threshold']}",
    ]
    cv2.rectangle(image, (0, 0), (image.shape[1], 108), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        cv2.putText(image, line[:105], (10, 22 + 25 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    load_dotenv(PROJECT_DIRECTORY / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY is missing from {PROJECT_DIRECTORY / '.env'}")
    if not CHECKPOINT_DIRECTORY.is_dir():
        raise FileNotFoundError(CHECKPOINT_DIRECTORY)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(WORLD_SEED)
    np.random.seed(WORLD_SEED % (2**32))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_directory = OUTPUT_ROOT / f"{timestamp}_{EXPERIMENT_SUFFIX}"
    experiment_directory.mkdir(parents=True, exist_ok=False)
    video_path = experiment_directory / "hierarchy.mp4"
    attempts_path = experiment_directory / "attempts.jsonl"
    summary_path = experiment_directory / "summary.json"
    (experiment_directory / "planner_system_prompt.txt").write_text(planner_system_prompt() + "\n", encoding="utf-8")
    (experiment_directory / "skill_library.json").write_text(json.dumps(SKILL_LIBRARY, indent=2) + "\n", encoding="utf-8")
    client = genai.Client(api_key=api_key)
    model = SteveOnePolicy.from_pretrained(CHECKPOINT_DIRECTORY).to("cuda").eval()
    torch.manual_seed(WORLD_SEED)
    simulator = MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=WORLD_SEED,
        preferred_spawn_biome="forest",
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[],
    )
    container, stream = create_video(video_path)
    started_at = datetime.now(timezone.utc).isoformat()
    started_time = time.monotonic()
    attempts = []
    recurrent_state = None
    total_steps = 0
    outcome = "session_start"
    planner_errors = []
    try:
        observation, info = simulator.reset()
        current_state = state_snapshot(info)
        global_baseline = current_state
        frames: deque[bytes] = deque([encode_frame(np.asarray(info["pov"]))], maxlen=MAX_CONTEXT_FRAMES)
        for attempt_index in range(MAX_ATTEMPTS):
            decision, planner_error = request_decision(client, outcome, current_state, frames, attempts)
            if planner_error:
                planner_errors.append(planner_error)
            skill = SKILL_LIBRARY[decision.executor_skill]
            max_steps = min(decision.max_steps, skill["default_steps"], MAX_TOTAL_STEPS - total_steps)
            if max_steps <= 0:
                break
            baseline = current_state
            attempt_total_start = total_steps
            condition = model.prepare_condition(
                {"cond_scale": CONDITION_SCALE, "text": skill["executor_prompt"]},
                deterministic=DETERMINISTIC_TEXT_PRIOR,
            )
            action_counts = {"attack": 0, "forward": 0, "jump": 0, "inventory": 0, "use": 0, "camera": 0}
            agent_button_counts = {}
            agent_camera_counts = {}
            blocked_world_controls = {}
            local_value = 0.0
            local_succeeded = False
            terminated = False
            truncated = False
            attempt_started = time.monotonic()
            attempt_steps = 0
            option_records = []
            option_verified_success = False
            print(json.dumps({
                "attempt": attempt_index + 1,
                "global": GLOBAL_OBJECTIVE,
                "local": decision.local_objective,
                "executor_skill": decision.executor_skill,
                "executor_prompt": skill["executor_prompt"],
                "execution_option": skill["option"],
                "max_steps": max_steps,
                "failure_category": decision.failure_category,
                "evidence": decision.evidence,
            }), flush=True)

            def option_step(action: dict, phase: str):
                nonlocal observation, info, current_state, total_steps, terminated, truncated
                rendered = overlay_frame(np.asarray(info["pov"]), decision, skill, attempt_index, total_steps, local_value)
                cv2.putText(rendered, phase[:95], (10, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 150), 1, cv2.LINE_AA)
                write_video_frame(container, stream, rendered)
                observation, reward, terminated, truncated, info = simulator.step(action)
                total_steps += 1
                current_state = state_snapshot(info)
                if total_steps % FRAME_SAMPLE_INTERVAL == 0:
                    frames.append(encode_frame(np.asarray(info["pov"])))
                return observation, reward, terminated, truncated, info

            controller = ClosedLoopOptions(simulator, option_step)
            execution_option = skill["option"]
            if execution_option == "CRAFT_RECIPE":
                recipe_name = {
                    "craft_planks": "planks",
                    "craft_sticks": "stick",
                    "craft_crafting_table": "crafting_table",
                    "craft_wooden_pickaxe": "wooden_pickaxe",
                    "craft_stone_pickaxe": "stone_pickaxe",
                }[decision.executor_skill]
                option_result, info = controller.craft_recipe(recipe_name, info)
                option_records.append(option_result)
                recurrent_state = None
            elif execution_option == "PLACE_BLOCK":
                option_result, info = controller.place_block("crafting_table", info)
                option_records.append(option_result)
                option_verified_success = bool(option_result["success"])
                recurrent_state = None
            elif execution_option == "COLLECT_DROP":
                baseline_quantity = grouped_inventory_value(info, "log")
                option_result, info = controller.collect_drop("log", baseline_quantity, info, max_steps=min(120, max_steps))
                option_records.append(option_result)
                recurrent_state = None
            else:
                if decision.executor_skill == "mine_cobblestone":
                    option_result, info = controller.equip_item("wooden_pickaxe", info)
                    option_records.append(option_result)
                    recurrent_state = None
                mine_trigger_target = "log" if decision.executor_skill == "collect_log" else "stone"
                last_mined_value = grouped_value(current_state["mine_block"], mine_trigger_target)
                learned_step_budget = min(max_steps, MAX_TOTAL_STEPS - total_steps)
                for attempt_step in range(1, learned_step_budget + 1):
                    if total_steps - attempt_total_start >= max_steps:
                        break
                    rendered = overlay_frame(np.asarray(info["pov"]), decision, skill, attempt_index, total_steps, local_value)
                    write_video_frame(container, stream, rendered)
                    image = torch.from_numpy(observation["image"]).unsqueeze(0).unsqueeze(0).to("cuda")
                    model_input = {"image": image, "condition": condition}
                    batched_action, recurrent_state = model.get_action(
                        model_input,
                        recurrent_state,
                        deterministic=DETERMINISTIC_ACTIONS,
                        input_shape="BT*",
                    )
                    action = {name: value[0][0] for name, value in batched_action.items()}
                    action, environment_action, blocked_controls = controller.gate_world_control(action)
                    for control in blocked_controls:
                        blocked_world_controls[control] = blocked_world_controls.get(control, 0) + 1
                    button_code = str(int(np.asarray(action["buttons"]).item()))
                    camera_code = str(int(np.asarray(action["camera"]).item()))
                    agent_button_counts[button_code] = agent_button_counts.get(button_code, 0) + 1
                    agent_camera_counts[camera_code] = agent_camera_counts.get(camera_code, 0) + 1
                    observation, reward, terminated, truncated, info = simulator.step(action)
                    total_steps += 1
                    for name in action_counts:
                        if name == "camera":
                            action_counts[name] += int(np.any(np.asarray(environment_action.get(name, [0.0, 0.0]))) != 0)
                        else:
                            action_counts[name] += int(np.asarray(environment_action.get(name, 0)).item() != 0)
                    current_state = state_snapshot(info)
                    local_value, local_succeeded = verifier_progress(skill, baseline, current_state)
                    global_value, completed = global_success(global_baseline, current_state)
                    if total_steps % FRAME_SAMPLE_INTERVAL == 0:
                        frames.append(encode_frame(np.asarray(info["pov"])))
                    current_mined_value = grouped_value(current_state["mine_block"], mine_trigger_target)
                    if execution_option == "MINE_AND_COLLECT" and current_mined_value > last_mined_value and not local_succeeded:
                        pickup_baseline = grouped_inventory_value(info, skill["target"])
                        remaining_steps = min(120, max_steps - (total_steps - attempt_total_start), MAX_TOTAL_STEPS - total_steps)
                        if remaining_steps > 0:
                            option_result, info = controller.collect_drop(skill["target"], pickup_baseline, info, max_steps=remaining_steps)
                            option_records.append(option_result)
                            recurrent_state = None
                            current_state = state_snapshot(info)
                            local_value, local_succeeded = verifier_progress(skill, baseline, current_state)
                            global_value, completed = global_success(global_baseline, current_state)
                        last_mined_value = current_mined_value
                    if local_succeeded or completed or terminated or truncated or total_steps >= MAX_TOTAL_STEPS or total_steps - attempt_total_start >= max_steps:
                        break

            current_state = state_snapshot(info)
            local_value, local_succeeded = verifier_progress(skill, baseline, current_state)
            if option_verified_success:
                local_value = max(local_value, float(skill["threshold"]))
                local_succeeded = True
            global_value, completed = global_success(global_baseline, current_state)
            attempt_steps = total_steps - attempt_total_start
            frames.append(encode_frame(np.asarray(info["pov"])))
            global_value, completed = global_success(global_baseline, current_state)
            attempt_result = {
                "attempt_index": attempt_index,
                "decision": decision.model_dump(),
                "executor_prompt": skill["executor_prompt"],
                "execution_option": skill["option"],
                "option_records": option_records,
                "verifier": {
                    "kind": skill["verifier_kind"],
                    "source": skill["source"],
                    "target": skill["target"],
                    "threshold": skill["threshold"],
                },
                "outcome": "global_success" if completed else "local_success" if local_succeeded else "environment_ended" if terminated or truncated else "timeout",
                "local_progress": local_value,
                "local_succeeded": local_succeeded,
                "global_progress": global_value,
                "global_succeeded": completed,
                "steps": attempt_steps,
                "total_steps": total_steps,
                "elapsed_seconds": time.monotonic() - attempt_started,
                "action_counts": action_counts,
                "agent_button_counts": agent_button_counts,
                "agent_camera_counts": agent_camera_counts,
                "blocked_world_controls": blocked_world_controls,
                "condition_embedding": {
                    "mean": float(condition["mineclip_embeds"].mean().item()),
                    "std": float(condition["mineclip_embeds"].std().item()),
                    "norm": float(condition["mineclip_embeds"].norm().item()),
                },
                "baseline_state": baseline,
                "final_state": current_state,
                "planner_error": planner_error,
            }
            attempts.append(attempt_result)
            with attempts_path.open("a", encoding="utf-8") as attempts_file:
                attempts_file.write(json.dumps(attempt_result, separators=(",", ":")) + "\n")
            print(json.dumps({
                "attempt": attempt_index + 1,
                "outcome": attempt_result["outcome"],
                "local_progress": local_value,
                "global_progress": global_value,
                "steps": attempt_steps,
                "total_steps": total_steps,
            }), flush=True)
            outcome = attempt_result["outcome"]
            if completed or terminated or truncated or total_steps >= MAX_TOTAL_STEPS:
                break
    finally:
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        simulator.close()
    global_value, completed = global_success(global_baseline, current_state)
    final_frame_path = experiment_directory / "final.jpg"
    final_frame = cv2.cvtColor(np.asarray(info["pov"]), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(final_frame_path), final_frame):
        raise RuntimeError(f"Could not write {final_frame_path}")
    summary = {
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started_time,
        "checkpoint": str(CHECKPOINT_DIRECTORY),
        "planner_model": MODEL_NAME,
        "global_objective": GLOBAL_OBJECTIVE,
        "global_verifier": {"source": "craft_item_or_inventory", "target": "stone_pickaxe", "threshold": 1},
        "global_progress": global_value,
        "global_succeeded": completed,
        "world_seed": WORLD_SEED,
        "condition_scale": CONDITION_SCALE,
        "deterministic_text_prior": DETERMINISTIC_TEXT_PRIOR,
        "deterministic_actions": DETERMINISTIC_ACTIONS,
        "recurrent_state": "preserved within STEVE-1 skills and reset after deterministic option transitions",
        "architecture": "STEVE-1 fuzzy world control plus verifier-driven pixel-GUI and recovery options",
        "total_steps": total_steps,
        "attempts": attempts,
        "planner_errors": planner_errors,
        "video": video_path.name,
        "final_frame": final_frame_path.name,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
