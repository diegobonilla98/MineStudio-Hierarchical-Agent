import json
import math
import os
import random
import time
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import httpx
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from minestudio.models import SteveOnePolicy
from minestudio.simulator import MinecraftSim
from pydantic import BaseModel, Field, ValidationError, field_validator

from closed_loop_options import ClosedLoopOptions, grouped_inventory_value, item_matches, normalize_identifier


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SUITE_DIRECTORY = Path(os.environ.get("MINESTUDIO_IRON_SUITE", PROJECT_DIRECTORY / "output" / "iron_pickaxe_viability" / "development"))
MANIFEST_PATH = SUITE_DIRECTORY / "manifest.json"
ARM = os.environ.get("MINESTUDIO_IRON_ARM", "full").lower()
EPISODES = int(os.environ.get("MINESTUDIO_IRON_EPISODES", "24"))
MAX_TOTAL_STEPS = int(os.environ.get("MINESTUDIO_IRON_MAX_STEPS", "12000"))
MAX_DECISIONS = int(os.environ.get("MINESTUDIO_IRON_MAX_DECISIONS", "32"))
MODEL_NAME = os.environ.get("MINESTUDIO_IRON_GEMINI_MODEL", "gemini-3.8-flash")
VANILLA_CHECKPOINT = Path(os.environ.get("MINESTUDIO_IRON_VANILLA_CHECKPOINT", PROJECT_DIRECTORY / "checkpoints" / "steve_one_official"))
SPECIALIST_CHECKPOINT = Path(os.environ.get("MINESTUDIO_IRON_SPECIALIST_CHECKPOINT", PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "architecture_sweeps" / "20260917T213751Z" / "upper_lora_rank32" / "best_model"))
GLOBAL_OBJECTIVE = "Build an iron pickaxe from a fresh random survival world."
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
CONDITION_SCALE = 6.0
FRAME_SAMPLE_INTERVAL = 100
MAX_CONTEXT_FRAMES = 12
JPEG_QUALITY = 78
ARM_NAMES = {"full": "FULL SYSTEM", "no_specialist": "NO SPECIALIST", "no_system2": "NO SYSTEM 2"}
SYSTEM1_ACTIONS = {"collect_logs", "collect_cobblestone", "collect_iron_ore", "collect_coal", "explore", "recover"}
SYSTEM0_ACTIONS = {
    "craft_planks",
    "craft_sticks",
    "craft_crafting_table",
    "craft_wooden_pickaxe",
    "craft_stone_pickaxe",
    "craft_furnace",
    "craft_iron_pickaxe",
    "equip_wooden_pickaxe",
    "equip_stone_pickaxe",
    "smelt_iron",
}
ACTION_PROMPTS = {
    "collect_logs": "chop down trees and collect wood logs",
    "collect_cobblestone": "mine stone and collect cobblestone",
    "collect_iron_ore": "find exposed iron ore, mine it with a stone pickaxe, and collect the iron ore",
    "collect_coal": "find exposed coal ore, mine it, and collect coal",
    "explore": "explore the world and search for useful exposed resources or cave openings",
    "recover": "escape the local terrain trap and return to stable navigable ground",
}
ACTION_TARGETS = {
    "collect_logs": ("log", "log"),
    "collect_cobblestone": ("stone", "cobblestone"),
    "collect_iron_ore": ("iron_ore", "iron_ore"),
    "collect_coal": ("coal_ore", "coal"),
}
CRAFT_ACTIONS = {
    "craft_planks": "planks",
    "craft_sticks": "stick",
    "craft_crafting_table": "crafting_table",
    "craft_wooden_pickaxe": "wooden_pickaxe",
    "craft_stone_pickaxe": "stone_pickaxe",
    "craft_furnace": "furnace",
    "craft_iron_pickaxe": "iron_pickaxe",
}
EQUIP_ACTIONS = {
    "equip_wooden_pickaxe": "wooden_pickaxe",
    "equip_stone_pickaxe": "stone_pickaxe",
}
MILESTONE_NAMES = (
    "logs_obtained",
    "crafting_capability_established",
    "wooden_pickaxe",
    "required_cobblestone",
    "stone_pickaxe",
    "furnace_available",
    "raw_iron_3",
    "iron_ingots_3",
    "iron_pickaxe",
)
FIXED_PLAN = (
    ("collect_logs", 4, "Collect four logs."),
    ("craft_planks", 1, "Craft planks."),
    ("craft_crafting_table", 1, "Craft a crafting table."),
    ("craft_sticks", 1, "Craft sticks."),
    ("craft_wooden_pickaxe", 1, "Craft a wooden pickaxe."),
    ("collect_cobblestone", 3, "Obtain three cobblestone."),
    ("craft_stone_pickaxe", 1, "Craft a stone pickaxe."),
    ("collect_cobblestone", 8, "Obtain eight cobblestone for a furnace."),
    ("craft_furnace", 1, "Craft a furnace."),
    ("collect_iron_ore", 3, "Obtain three iron ore."),
    ("craft_sticks", 1, "Ensure sticks are available for the iron pickaxe."),
    ("smelt_iron", 3, "Smelt three iron ingots."),
    ("craft_iron_pickaxe", 1, "Craft an iron pickaxe."),
)


class PlannerDecision(BaseModel):
    global_objective: Literal["Build an iron pickaxe from a fresh random survival world."]
    current_state_summary: str = Field(min_length=3, max_length=400)
    short_term_objective: str = Field(min_length=3, max_length=220)
    immediate_instruction: str = Field(min_length=3, max_length=220)
    route: Literal["SYSTEM0", "SYSTEM1"]
    action: Literal[
        "collect_logs",
        "collect_cobblestone",
        "collect_iron_ore",
        "collect_coal",
        "explore",
        "recover",
        "craft_planks",
        "craft_sticks",
        "craft_crafting_table",
        "craft_wooden_pickaxe",
        "craft_stone_pickaxe",
        "craft_furnace",
        "craft_iron_pickaxe",
        "equip_wooden_pickaxe",
        "equip_stone_pickaxe",
        "smelt_iron",
    ]
    target_count: int = Field(ge=1, le=16)
    fuel_item: Literal["coal", "log", "planks"]
    max_steps: int = Field(ge=200, le=1600)
    world_memory: list[str] = Field(max_length=10)
    failure_interpretation: str = Field(min_length=3, max_length=400)

    @field_validator(
        "current_state_summary",
        "short_term_objective",
        "immediate_instruction",
        "failure_interpretation",
        mode="before",
    )
    @classmethod
    def truncate_bounded_text(cls, value: object, info) -> object:
        limits = {
            "current_state_summary": 400,
            "short_term_objective": 220,
            "immediate_instruction": 220,
            "failure_interpretation": 400,
        }
        if isinstance(value, str):
            return value[: limits[info.field_name]]
        return value


class FailureCritique(BaseModel):
    primary_owner: Literal["System 2", "System 1", "System 0", "Interface/router", "Environment/exploration"]
    primary_cause: Literal[
        "bad strategy",
        "failed state tracking",
        "bad next objective",
        "failed replanning",
        "navigation/perception",
        "target acquisition",
        "terrain recovery",
        "combat/hazard",
        "camera/control",
        "deterministic mechanic failed",
        "wrong handoff",
        "repeated switching",
        "missing context",
        "resource not found within budget",
    ]
    secondary_contributors: list[str] = Field(max_length=5)
    evidence: list[str] = Field(min_length=1, max_length=8)
    causal_summary: str = Field(min_length=3, max_length=700)
    last_useful_milestone: str
    avoidable: bool


class EpisodeBudgetExceeded(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory_counts(info: dict) -> dict[str, float]:
    counts: dict[str, float] = {}
    for stack in info.get("inventory", {}).values():
        item_name = normalize_identifier(str(stack.get("type", "air")))
        quantity = float(stack.get("quantity", 0))
        if item_name not in {"air", "none"} and quantity > 0:
            counts[item_name] = counts.get(item_name, 0.0) + quantity
    return counts


def numeric_mapping(info: dict, source: str) -> dict[str, float]:
    return {
        normalize_identifier(str(name)): float(np.asarray(value))
        for name, value in info.get(source, {}).items()
        if float(np.asarray(value)) != 0.0
    }


def grouped_value(values: dict[str, float], target: str) -> float:
    return sum(value for name, value in values.items() if item_matches(name, target))


def raw_iron_count(inventory: dict[str, float]) -> float:
    return inventory.get("iron_ore", 0.0) + inventory.get("raw_iron", 0.0)


def state_snapshot(info: dict) -> dict:
    position = info.get("player_pos", {})
    return {
        "inventory": inventory_counts(info),
        "mine_block": numeric_mapping(info, "mine_block"),
        "craft_item": numeric_mapping(info, "craft_item"),
        "pickup": numeric_mapping(info, "pickup"),
        "place_block": numeric_mapping(info, "place_block"),
        "break_item": numeric_mapping(info, "break_item"),
        "kill_entity": numeric_mapping(info, "kill_entity"),
        "equipped_mainhand": normalize_identifier(str(info.get("equipped_items", {}).get("mainhand", {}).get("type", "air"))),
        "player_pos": {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", 0.0)),
            "yaw": float(position.get("yaw", 0.0)),
            "pitch": float(position.get("pitch", 0.0)),
        },
        "health": float(info.get("health", 20.0)),
        "food_level": float(info.get("food_level", 20.0)),
        "is_gui_open": bool(info.get("is_gui_open", False)),
    }


def encode_frame(image: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not success:
        raise RuntimeError("Could not encode frame")
    return encoded.tobytes()


def retry_generate(client: genai.Client, contents: list, config: types.GenerateContentConfig):
    last_error = None
    for attempt in range(5):
        try:
            return client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
        except (httpx.HTTPError, errors.APIError) as error:
            last_error = error
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Gemini request failed after retries: {last_error}")


def planner_prompt() -> str:
    action_catalog = {
        "SYSTEM1": {
            "collect_logs": "Open-world tree search, approach, break, and pickup.",
            "collect_cobblestone": "Open-world stone acquisition. The stone specialist is used only in FULL SYSTEM.",
            "collect_iron_ore": "Open-world iron search, approach, mining, and pickup.",
            "collect_coal": "Open-world coal search, mining, and pickup.",
            "explore": "Open-ended local exploration for resources or navigable terrain.",
            "recover": "Uncertain terrain or hazard recovery.",
        },
        "SYSTEM0": {
            "craft_planks": "Craft planks when a log is available.",
            "craft_sticks": "Craft sticks when planks are available.",
            "craft_crafting_table": "Craft a crafting table.",
            "craft_wooden_pickaxe": "Craft a wooden pickaxe using a table.",
            "craft_stone_pickaxe": "Craft a stone pickaxe using a table.",
            "craft_furnace": "Craft a furnace using a table.",
            "craft_iron_pickaxe": "Craft an iron pickaxe using a table.",
            "equip_wooden_pickaxe": "Equip a wooden pickaxe already in inventory.",
            "equip_stone_pickaxe": "Equip a stone pickaxe already in inventory.",
            "smelt_iron": "Use a furnace to smelt the requested iron count with the specified available fuel.",
        },
    }
    recipes = {
        "planks": {"input": {"log": 1}, "output": 4},
        "sticks": {"input": {"planks": 2}, "output": 4},
        "crafting_table": {"input": {"planks": 4}, "output": 1},
        "wooden_pickaxe": {"input": {"planks": 3, "stick": 2}, "output": 1},
        "stone_pickaxe": {"input": {"cobblestone": 3, "stick": 2}, "output": 1},
        "furnace": {"input": {"cobblestone": 8}, "output": 1},
        "iron_ingot": {"input": {"iron_ore_or_raw_iron": 1, "furnace": 1, "fuel": "required"}, "output": 1},
        "iron_pickaxe": {"input": {"iron_ingot": 3, "stick": 2}, "output": 1},
    }
    return f"""
You are System 2 for a Minecraft 1.16 survival agent. Your global objective is exactly: {GLOBAL_OBJECTIVE}

Maintain the global objective, interpret exact state and verifier outcomes, preserve compact world memory, and select one immediate action. You may change plans after failure. Do not issue keyboard or mouse actions. Do not claim success without exact state evidence. Do not use hidden coordinates, seed knowledge, or resources not visible to a player. Do not repeat a failed action without explaining why its preconditions or execution conditions are now different.

System 0 performs known deterministic GUI and station mechanics and verifies them. System 1 handles uncertain perception, exploration, resource acquisition, navigation, and terrain. A deterministic precondition failure is information for replanning, not progress. target_count is the desired total inventory count for collection actions and the requested output count for smelting. For other actions use 1. fuel_item must name an actually available fuel when selecting smelt_iron. Minecraft 1.16 represents mined iron as iron_ore rather than raw_iron.

The action catalog and recipe facts are unordered capabilities, not a prescribed plan:
{json.dumps(action_catalog, indent=2)}
{json.dumps(recipes, indent=2)}
""".strip()


def request_planner_decision(client: genai.Client, state: dict, milestones: dict, attempts: list[dict], frames: deque[bytes], memory: list[str], event: str) -> PlannerDecision:
    payload = {
        "event": event,
        "global_objective": GLOBAL_OBJECTIVE,
        "exact_state": state,
        "milestones": milestones,
        "prior_world_memory": memory,
        "recent_attempts": attempts[-8:],
        "frames": {"order": "oldest_to_newest", "count": len(frames)},
        "request": "Update state and memory, diagnose the previous outcome, and select one feasible next action.",
    }
    contents = [types.Part.from_text(text=json.dumps(payload, separators=(",", ":")))]
    contents.extend(types.Part.from_bytes(data=frame, mime_type="image/jpeg") for frame in frames)
    config = types.GenerateContentConfig(
        system_instruction=planner_prompt(),
        response_mime_type="application/json",
        response_schema=PlannerDecision,
        max_output_tokens=4000,
        temperature=0.15,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    validation_error = None
    for _ in range(4):
        response = retry_generate(client, contents, config)
        try:
            decision = PlannerDecision.model_validate_json(response.text)
            break
        except ValidationError as error:
            validation_error = error
    else:
        raise RuntimeError(f"Gemini returned invalid planner JSON after retries: {validation_error}")
    if decision.route == "SYSTEM0" and decision.action not in SYSTEM0_ACTIONS:
        raise ValueError(f"Invalid route/action combination: {decision.route}/{decision.action}")
    if decision.route == "SYSTEM1" and decision.action not in SYSTEM1_ACTIONS:
        raise ValueError(f"Invalid route/action combination: {decision.route}/{decision.action}")
    return decision


def fixed_decision(plan_index: int) -> PlannerDecision:
    action, count, objective = FIXED_PLAN[min(plan_index, len(FIXED_PLAN) - 1)]
    route = "SYSTEM1" if action in SYSTEM1_ACTIONS else "SYSTEM0"
    return PlannerDecision(
        global_objective=GLOBAL_OBJECTIVE,
        current_state_summary=f"Fixed tech-tree phase {plan_index + 1} of {len(FIXED_PLAN)}.",
        short_term_objective=objective,
        immediate_instruction=objective,
        route=route,
        action=action,
        target_count=count,
        fuel_item="planks",
        max_steps=1600 if route == "SYSTEM1" else 800,
        world_memory=[],
        failure_interpretation="Fixed controller retries the current verifier-defined phase.",
    )


def milestone_predicates(state: dict) -> dict[str, bool]:
    inventory = state["inventory"]
    crafted = state["craft_item"]
    placed = state["place_block"]
    return {
        "logs_obtained": grouped_value(inventory, "log") >= 1,
        "crafting_capability_established": inventory.get("crafting_table", 0) >= 1 or crafted.get("crafting_table", 0) >= 1 or placed.get("crafting_table", 0) >= 1,
        "wooden_pickaxe": inventory.get("wooden_pickaxe", 0) >= 1 or crafted.get("wooden_pickaxe", 0) >= 1,
        "required_cobblestone": inventory.get("cobblestone", 0) >= 3,
        "stone_pickaxe": inventory.get("stone_pickaxe", 0) >= 1 or crafted.get("stone_pickaxe", 0) >= 1,
        "furnace_available": inventory.get("furnace", 0) >= 1 or crafted.get("furnace", 0) >= 1 or placed.get("furnace", 0) >= 1,
        "raw_iron_3": raw_iron_count(inventory) >= 3,
        "iron_ingots_3": inventory.get("iron_ingot", 0) >= 3,
        "iron_pickaxe": inventory.get("iron_pickaxe", 0) >= 1 or crafted.get("iron_pickaxe", 0) >= 1,
    }


def update_milestones(milestones: dict, state: dict, step: int, elapsed: float, frame: np.ndarray, episode_directory: Path) -> None:
    for name, achieved in milestone_predicates(state).items():
        if achieved and milestones[name]["step"] is None:
            milestones[name] = {"step": step, "seconds": elapsed}
            path = episode_directory / f"milestone_{name}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def action_satisfied(action: str, target_count: int, baseline: dict, current: dict, result: dict | None = None) -> bool:
    inventory = current["inventory"]
    if action == "collect_logs":
        return grouped_value(inventory, "log") >= target_count
    if action == "collect_cobblestone":
        return inventory.get("cobblestone", 0) >= target_count
    if action == "collect_iron_ore":
        return raw_iron_count(inventory) >= target_count
    if action == "collect_coal":
        return inventory.get("coal", 0) >= target_count
    if action == "explore":
        dx = current["player_pos"]["x"] - baseline["player_pos"]["x"]
        dz = current["player_pos"]["z"] - baseline["player_pos"]["z"]
        return math.hypot(dx, dz) >= 8
    if action == "recover":
        return current["player_pos"]["y"] >= baseline["player_pos"]["y"] + 1 or math.hypot(current["player_pos"]["x"] - baseline["player_pos"]["x"], current["player_pos"]["z"] - baseline["player_pos"]["z"]) >= 4
    return bool(result and result.get("success"))


def fixed_phase_satisfied(plan_index: int, state: dict, milestones: dict) -> bool:
    action, count, objective = FIXED_PLAN[plan_index]
    inventory = state["inventory"]
    if action == "collect_logs":
        return grouped_value(inventory, "log") >= count
    if action == "collect_cobblestone":
        return inventory.get("cobblestone", 0) >= count
    if action == "collect_iron_ore":
        return raw_iron_count(inventory) >= count
    if action == "craft_planks":
        return inventory.get("planks", 0) >= 12
    if action == "craft_sticks":
        return inventory.get("stick", 0) >= 2
    target = CRAFT_ACTIONS.get(action)
    if target is not None:
        return inventory.get(target, 0) >= 1 or milestones[{"crafting_table": "crafting_capability_established", "wooden_pickaxe": "wooden_pickaxe", "stone_pickaxe": "stone_pickaxe", "furnace": "furnace_available", "iron_pickaxe": "iron_pickaxe"}[target]]["step"] is not None
    if action == "smelt_iron":
        return inventory.get("iron_ingot", 0) >= count
    return False


def choose_tool(action: str, info: dict) -> str | None:
    if action == "collect_cobblestone":
        if grouped_inventory_value(info, "stone_pickaxe") >= 1:
            return "stone_pickaxe"
        return "wooden_pickaxe"
    if action in {"collect_iron_ore", "collect_coal"}:
        return "stone_pickaxe"
    return None


def execute_system0(action: str, decision: PlannerDecision, controller: ClosedLoopOptions, info: dict) -> tuple[dict, dict]:
    if action in CRAFT_ACTIONS:
        result, info = controller.craft_recipe(CRAFT_ACTIONS[action], info)
        if result.get("success") and action in {"craft_stone_pickaxe", "craft_furnace"}:
            reclaim, info = controller.reclaim_block("crafting_table", info, max_steps=200)
            result["station_reclaim"] = reclaim
            result["success"] = bool(reclaim.get("success"))
            if not result["success"]:
                result["reason"] = f"crafted output but station lifecycle failed: {reclaim.get('reason')}"
        return result, info
    if action in EQUIP_ACTIONS:
        return controller.equip_item(EQUIP_ACTIONS[action], info)
    if action == "smelt_iron":
        input_item = "raw_iron" if grouped_inventory_value(info, "raw_iron") >= decision.target_count else "iron_ore"
        result, info = controller.smelt(input_item, "iron_ingot", decision.target_count, decision.fuel_item, info)
        if result.get("success"):
            equip, info = controller.equip_item("stone_pickaxe", info)
            result["reclaim_tool"] = equip
            reclaim, info = controller.reclaim_block("furnace", info, max_steps=200)
            result["station_reclaim"] = reclaim
            result["success"] = bool(reclaim.get("success"))
            if not result["success"]:
                result["reason"] = f"smelted output but station lifecycle failed: {reclaim.get('reason')}"
        return result, info
    return {"option": action, "success": False, "reason": "unsupported System 0 action"}, info


def request_critique(client: genai.Client, summary: dict, frames: deque[bytes]) -> dict:
    prompt = """
You are a causal evaluator of a failed Minecraft long-horizon episode. Use exact milestones, inventory, action outcomes, and chronological frames. Identify the earliest actionable primary cause, not merely the last symptom. Respect the System 2/System 1/System 0/interface/environment ownership taxonomy in the response schema. Do not invent unseen resources or events. A resource-not-found diagnosis is valid only when execution searched competently within the finite budget and no earlier controllable failure dominates.
""".strip()
    compact = {
        "arm": summary["arm"],
        "seed": summary["seed"],
        "milestones": summary["milestones"],
        "steps": summary["total_steps"],
        "death": summary["death"],
        "timeouts": summary["timeouts"],
        "stuck_loops": summary["stuck_loops"],
        "final_inventory": summary["final_inventory"],
        "attempts": summary["attempts"][-12:],
    }
    contents = [types.Part.from_text(text=json.dumps(compact, separators=(",", ":")))]
    contents.extend(types.Part.from_bytes(data=frame, mime_type="image/jpeg") for frame in frames)
    response = retry_generate(
        client,
        contents,
        types.GenerateContentConfig(
            system_instruction=prompt,
            response_mime_type="application/json",
            response_schema=FailureCritique,
            max_output_tokens=1800,
            temperature=0.05,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return FailureCritique.model_validate_json(response.text).model_dump()


def create_simulator(seed: int) -> MinecraftSim:
    return MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=seed,
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[],
    )


def run_episode(seed: int, episode_index: int, vanilla: SteveOnePolicy, specialist: SteveOnePolicy | None, client: genai.Client, arm_directory: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    episode_directory = arm_directory / "episodes" / f"seed_{seed}"
    episode_directory.mkdir(parents=True, exist_ok=True)
    attempts_path = episode_directory / "attempts.jsonl"
    if attempts_path.exists():
        attempts_path.unlink()
    simulator = create_simulator(seed)
    started_at = utc_now()
    started_time = time.monotonic()
    total_steps = 0
    system0_calls = 0
    system1_calls = 0
    router_transitions = 0
    planner_replans = 0
    timeouts = 0
    stuck_loops = 0
    recovery_attempts = 0
    prior_route = None
    memory: list[str] = []
    attempts: list[dict] = []
    milestones = {name: {"step": None, "seconds": None} for name in MILESTONE_NAMES}
    frame_context: deque[bytes] = deque(maxlen=MAX_CONTEXT_FRAMES)
    recurrent_states = {"vanilla": None, "specialist": None}
    fixed_plan_index = 0
    event = "session_start"
    infrastructure_error = None
    terminated = False
    truncated = False
    death = False
    observation = None
    info = None
    current_state = None
    try:
        observation, info = simulator.reset()
        current_state = state_snapshot(info)
        frame_context.append(encode_frame(np.asarray(info["pov"])))
        update_milestones(milestones, current_state, total_steps, time.monotonic() - started_time, np.asarray(info["pov"]), episode_directory)

        def traced_step(agent_action: dict, phase: str):
            nonlocal observation, info, current_state, total_steps, terminated, truncated, death
            if total_steps >= MAX_TOTAL_STEPS:
                raise EpisodeBudgetExceeded("episode step budget exhausted")
            observation, reward, terminated, truncated, info = simulator.step(agent_action)
            total_steps += 1
            current_state = state_snapshot(info)
            death = death or current_state["health"] <= 0
            if total_steps % FRAME_SAMPLE_INTERVAL == 0:
                frame_context.append(encode_frame(np.asarray(info["pov"])))
            update_milestones(milestones, current_state, total_steps, time.monotonic() - started_time, np.asarray(info["pov"]), episode_directory)
            return observation, reward, terminated, truncated, info

        controller = ClosedLoopOptions(simulator, traced_step, pickup_implementation="old")
        for decision_index in range(MAX_DECISIONS):
            if milestones["iron_pickaxe"]["step"] is not None or terminated or truncated or total_steps >= MAX_TOTAL_STEPS:
                break
            if ARM == "no_system2":
                while fixed_plan_index < len(FIXED_PLAN) and fixed_phase_satisfied(fixed_plan_index, current_state, milestones):
                    fixed_plan_index += 1
                if fixed_plan_index >= len(FIXED_PLAN):
                    break
                decision = fixed_decision(fixed_plan_index)
                planner_error = None
            else:
                decision = request_planner_decision(client, current_state, milestones, attempts, frame_context, memory, event)
                planner_error = None
                memory = decision.world_memory
                if decision_index > 0:
                    planner_replans += 1
            if prior_route is not None and prior_route != decision.route:
                router_transitions += 1
            prior_route = decision.route
            baseline = deepcopy(current_state)
            start_position = np.array([baseline["player_pos"][key] for key in ("x", "y", "z")], dtype=np.float32)
            attempt_start_step = total_steps
            attempt_started = time.monotonic()
            option_records = []
            movement_actions = 0
            attempt_result = None
            if decision.route == "SYSTEM0":
                system0_calls += 1
                try:
                    attempt_result, info = execute_system0(decision.action, decision, controller, info)
                except EpisodeBudgetExceeded:
                    attempt_result = {"option": decision.action, "success": False, "reason": "episode step budget exhausted"}
                option_records.append(attempt_result)
                recurrent_states = {"vanilla": None, "specialist": None}
            else:
                system1_calls += 1
                if decision.action == "recover":
                    recovery_attempts += 1
                previous_action = attempts[-1]["decision"]["action"] if attempts else None
                use_specialist = ARM in {"full", "no_system2"} and (
                    decision.action == "collect_cobblestone"
                    or (decision.action == "recover" and previous_action == "collect_cobblestone")
                )
                policy_name = "specialist" if use_specialist else "vanilla"
                policy = specialist if use_specialist else vanilla
                if policy is None:
                    raise RuntimeError("Specialist routing requested without a loaded specialist")
                prompt = ACTION_PROMPTS[decision.action]
                condition = policy.prepare_condition({"cond_scale": CONDITION_SCALE, "text": prompt}, deterministic=False)
                tool = choose_tool(decision.action, info)
                if tool is not None:
                    equip_result, info = controller.equip_item(tool, info)
                    option_records.append(equip_result)
                    system0_calls += 1
                mine_target, drop_target = ACTION_TARGETS.get(decision.action, (None, None))
                last_mined = grouped_value(current_state["mine_block"], mine_target) if mine_target else 0.0
                learned_budget = min(decision.max_steps, MAX_TOTAL_STEPS - total_steps)
                for learned_step in range(learned_budget):
                    if terminated or truncated or total_steps >= MAX_TOTAL_STEPS:
                        break
                    image = torch.from_numpy(observation["image"]).unsqueeze(0).unsqueeze(0).to("cuda")
                    with torch.inference_mode():
                        batched_action, recurrent_states[policy_name] = policy.get_action(
                            {"image": image, "condition": condition},
                            recurrent_states[policy_name],
                            deterministic=False,
                            input_shape="BT*",
                        )
                    raw_action = {name: value[0][0] for name, value in batched_action.items()}
                    action, environment_action, blocked = controller.gate_world_control(raw_action)
                    movement_actions += int(any(int(np.asarray(environment_action.get(name, 0)).item()) != 0 for name in ("forward", "back", "left", "right", "jump")))
                    try:
                        observation, reward, terminated, truncated, info = traced_step(action, f"SYSTEM1/{decision.action}")
                    except EpisodeBudgetExceeded:
                        break
                    if action_satisfied(decision.action, decision.target_count, baseline, current_state):
                        break
                    if mine_target is not None:
                        mined = grouped_value(current_state["mine_block"], mine_target)
                        if mined > last_mined and not action_satisfied(decision.action, decision.target_count, baseline, current_state):
                            pickup_baseline = grouped_inventory_value(info, drop_target)
                            try:
                                pickup_result, info = controller.collect_drop(drop_target, pickup_baseline, info, max_steps=min(120, MAX_TOTAL_STEPS - total_steps))
                            except EpisodeBudgetExceeded:
                                pickup_result = {"option": "COLLECT_DROP", "success": False, "reason": "episode step budget exhausted"}
                            option_records.append(pickup_result)
                            system0_calls += 1
                            recurrent_states[policy_name] = None
                            last_mined = mined
                            if action_satisfied(decision.action, decision.target_count, baseline, current_state):
                                break
                attempt_result = {"option": "SYSTEM1", "policy": policy_name, "success": action_satisfied(decision.action, decision.target_count, baseline, current_state)}
            current_state = state_snapshot(info)
            success = action_satisfied(decision.action, decision.target_count, baseline, current_state, attempt_result)
            displacement = float(np.linalg.norm(np.array([current_state["player_pos"][key] for key in ("x", "y", "z")], dtype=np.float32) - start_position))
            if decision.route == "SYSTEM1" and not success and movement_actions >= 50 and displacement < 1.0:
                stuck_loops += 1
            if not success:
                timeouts += 1
            attempt = {
                "decision_index": decision_index,
                "decision": decision.model_dump(),
                "outcome": "success" if success else "timeout_or_failure",
                "steps": total_steps - attempt_start_step,
                "total_steps": total_steps,
                "seconds": time.monotonic() - attempt_started,
                "displacement": displacement,
                "movement_actions": movement_actions,
                "option_records": option_records,
                "state_before": baseline,
                "state_after": current_state,
                "planner_error": planner_error,
            }
            attempts.append(attempt)
            with attempts_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(attempt, separators=(",", ":")) + "\n")
            frame_context.append(encode_frame(np.asarray(info["pov"])))
            event = "local_success" if success else "local_failure"
            if ARM == "no_system2" and success:
                fixed_plan_index += 1
            print(json.dumps({"arm": ARM, "episode": episode_index, "seed": seed, "decision": decision_index + 1, "action": decision.action, "outcome": event, "steps": total_steps, "milestone": max((name for name in MILESTONE_NAMES if milestones[name]["step"] is not None), default="none")}), flush=True)
    except (RuntimeError, ValidationError, ValueError) as error:
        message = str(error)
        normal_environment_end = message.startswith("Environment ended during") and message.endswith(": None")
        if not normal_environment_end:
            infrastructure_error = f"{type(error).__name__}: {error}"
    finally:
        simulator.close()
    if current_state is None:
        current_state = {}
    success = milestones["iron_pickaxe"]["step"] is not None
    summary = {
        "status": "complete" if infrastructure_error is None else "infrastructure_error",
        "arm": ARM,
        "arm_name": ARM_NAMES[ARM],
        "episode_index": episode_index,
        "seed": seed,
        "global_objective": GLOBAL_OBJECTIVE,
        "success": success,
        "started_at": started_at,
        "ended_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started_time,
        "total_steps": total_steps,
        "milestones": milestones,
        "planner_replans": planner_replans,
        "system0_calls": system0_calls,
        "system1_calls": system1_calls,
        "router_transitions": router_transitions,
        "timeouts": timeouts,
        "stuck_loops": stuck_loops,
        "recovery_attempts": recovery_attempts,
        "death": death,
        "terminated": terminated,
        "truncated": truncated,
        "final_inventory": current_state.get("inventory", {}),
        "attempts": attempts,
        "infrastructure_error": infrastructure_error,
        "failure_critique": None,
    }
    if not success and infrastructure_error is None:
        try:
            summary["failure_critique"] = request_critique(client, summary, frame_context)
        except (RuntimeError, ValidationError, ValueError) as error:
            summary["failure_critique_error"] = f"{type(error).__name__}: {error}"
    for frame_index, frame in enumerate(frame_context):
        (episode_directory / f"critic_{frame_index:02d}.jpg").write_bytes(frame)
    (episode_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def summarize_arm(results: list[dict]) -> dict:
    valid = [row for row in results if row["status"] == "complete"]
    successes = [row for row in valid if row["success"]]
    milestones = {}
    for name in MILESTONE_NAMES:
        achieved = [row for row in valid if row["milestones"][name]["step"] is not None]
        milestones[name] = {
            "achieved": len(achieved),
            "rate": len(achieved) / len(valid) if valid else 0.0,
            "mean_steps_when_achieved": float(np.mean([row["milestones"][name]["step"] for row in achieved])) if achieved else None,
            "median_steps_when_achieved": float(np.median([row["milestones"][name]["step"] for row in achieved])) if achieved else None,
        }
    return {
        "arm": ARM,
        "arm_name": ARM_NAMES[ARM],
        "episodes_target": EPISODES,
        "episodes_valid": len(valid),
        "infrastructure_errors": len(results) - len(valid),
        "successes": len(successes),
        "success_rate": len(successes) / len(valid) if valid else 0.0,
        "milestones": milestones,
        "mean_total_steps": float(np.mean([row["total_steps"] for row in valid])) if valid else None,
        "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in valid])) if valid else None,
        "mean_planner_replans": float(np.mean([row["planner_replans"] for row in valid])) if valid else None,
        "mean_system0_calls": float(np.mean([row["system0_calls"] for row in valid])) if valid else None,
        "mean_system1_calls": float(np.mean([row["system1_calls"] for row in valid])) if valid else None,
        "mean_router_transitions": float(np.mean([row["router_transitions"] for row in valid])) if valid else None,
        "mean_timeouts": float(np.mean([row["timeouts"] for row in valid])) if valid else None,
        "mean_stuck_loops": float(np.mean([row["stuck_loops"] for row in valid])) if valid else None,
        "deaths": sum(int(row["death"]) for row in valid),
        "updated_at": utc_now(),
    }


def main() -> None:
    if ARM not in ARM_NAMES:
        raise ValueError(f"Unknown arm: {ARM}")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(MANIFEST_PATH)
    if not VANILLA_CHECKPOINT.is_dir():
        raise FileNotFoundError(VANILLA_CHECKPOINT)
    if ARM in {"full", "no_system2"} and not SPECIALIST_CHECKPOINT.is_dir():
        raise FileNotFoundError(SPECIALIST_CHECKPOINT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    load_dotenv(PROJECT_DIRECTORY / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["episodes"][:EPISODES]
    arm_directory = SUITE_DIRECTORY / ARM
    arm_directory.mkdir(parents=True, exist_ok=True)
    (arm_directory / "planner_system_prompt.txt").write_text(planner_prompt() + "\n", encoding="utf-8")
    torch.set_float32_matmul_precision("high")
    vanilla = SteveOnePolicy.from_pretrained(VANILLA_CHECKPOINT).to("cuda").eval()
    specialist = SteveOnePolicy.from_pretrained(SPECIALIST_CHECKPOINT).to("cuda").eval() if ARM in {"full", "no_system2"} else None
    client = genai.Client(api_key=api_key)
    results_path = arm_directory / "results.jsonl"
    results_by_seed = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("status") == "complete":
                results_by_seed[int(row["seed"])] = row
    for entry in entries:
        seed = int(entry["seed"])
        if seed in results_by_seed:
            continue
        result = run_episode(seed, int(entry["episode_index"]), vanilla, specialist, client, arm_directory)
        with results_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
        if result["status"] != "complete":
            raise RuntimeError(f"Infrastructure error on {ARM} seed {seed}: {result['infrastructure_error']}")
        results_by_seed[seed] = result
        summary = summarize_arm(list(results_by_seed.values()))
        (arm_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary = summarize_arm(list(results_by_seed.values()))
    (arm_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
