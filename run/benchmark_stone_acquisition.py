import gzip
import hashlib
import json
import math
import multiprocessing
import os
import queue
import random
import time
import traceback
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from closed_loop_options import ClosedLoopOptions, grouped_inventory_value
from minestudio.models import SteveOnePolicy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, VoxelsCallback
from stone_task_router import LegacyStoneTaskRouter, StoneTaskRouter


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = Path(os.environ.get("MINESTUDIO_STONE_CHECKPOINT", PROJECT_DIRECTORY / "checkpoints" / "steve_one_official"))
OUTPUT_ROOT = PROJECT_DIRECTORY / "output" / "stone_acquisition"
EXPERIMENT_NAME = os.environ.get("MINESTUDIO_STONE_EXPERIMENT", "stage0_baseline_v1")
EXPERIMENT_DIRECTORY = OUTPUT_ROOT / EXPERIMENT_NAME
GLOBAL_GOAL = "obtain 3 cobblestone"
POLICY_PROMPT = "mine stone and collect cobblestone, obtain 3 cobblestone"
TASK_BANK_PATH = Path(os.environ.get("MINESTUDIO_STONE_TASK_BANK", PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "dataset_v1" / "gemini_task_bank.json"))
EVENT_DRIVEN_TASKS = os.environ.get("MINESTUDIO_STONE_EVENT_TASKS", "0") == "1"
ROUTER_IMPLEMENTATION = os.environ.get("MINESTUDIO_STONE_ROUTER", "new")
PICKUP_IMPLEMENTATION = os.environ.get("MINESTUDIO_STONE_PICKUP", "hardened")
if ROUTER_IMPLEMENTATION not in {"old", "new"}:
    raise ValueError(f"Unknown router implementation: {ROUTER_IMPLEMENTATION}")
if PICKUP_IMPLEMENTATION not in {"old", "hardened"}:
    raise ValueError(f"Unknown pickup implementation: {PICKUP_IMPLEMENTATION}")
TARGET_COBBLESTONE = 3
EPISODES = int(os.environ.get("MINESTUDIO_STONE_EPISODES", "1000"))
NUM_WORKERS = int(os.environ.get("MINESTUDIO_STONE_WORKERS", "2"))
MAX_STEPS = int(os.environ.get("MINESTUDIO_STONE_MAX_STEPS", "1200"))
MAX_COLLECT_STEPS = 80
MAX_EPISODES_PER_SIMULATOR = 5
MAX_WORKER_WAVES = 8
BASE_SEED = int(os.environ.get("MINESTUDIO_STONE_BASE_SEED", "2026091501"))
BENCHMARK_STAGE = os.environ.get("MINESTUDIO_STONE_STAGE", "Stage 0 baseline")
MANIFEST_KIND = os.environ.get("MINESTUDIO_STONE_MANIFEST_KIND", "standard")
CONDITION_SCALE = 6.0
DETERMINISTIC_TEXT_PRIOR = False
DETERMINISTIC_ACTIONS = False
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
SETUP_SETTLE_STEPS = 3
MAX_SETUP_SETTLE_STEPS = 24
STEPS_PER_SECOND = 20
STUCK_WINDOW = 80
STUCK_MOVEMENT_FRACTION = 0.70
STUCK_DISPLACEMENT = 0.60
NO_PROGRESS_WINDOW = 240
AUXILIARY_HORIZON = 40
WRONG_BLOCK_MINED_MIN = 3
BIOMES = ("forest", "plains", "taiga", "swamp", "extreme_hills")
TASK_CATEGORIES = ("stone_acquisition", "water_recovery", "shore_exit", "stone_reacquisition", "hole_recovery")
SCENARIO_WEIGHTS = {
    "natural": 50,
    "exposed_near": 15,
    "exposed_far": 10,
    "water_near_stone": 10,
    "stone_slope": 8,
    "vegetation_occluded": 7,
}
MINOR_INVENTORIES = (
    (),
    (("torch", 4),),
    (("dirt", 4),),
    (("oak_sapling", 2),),
    (("bread", 2),),
    (("cooked_beef", 2), ("torch", 2)),
)
EVENT_SOURCES = (
    "mine_block",
    "pickup",
    "craft_item",
    "place_block",
    "kill_entity",
    "break_item",
    "use_item",
    "damage_dealt",
)
ENV_BUTTONS = (
    "attack",
    "back",
    "forward",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
    "use",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "inventory",
)
MOVEMENT_BUTTONS = ("back", "forward", "jump", "left", "right", "sprint")
PRIMARY_LABEL_ORDER = (
    "SUCCESS",
    "DIED",
    "DROWNING",
    "STONE_BROKEN_NOT_COLLECTED",
    "ENTERED_WATER",
    "TRAPPED_IN_HOLE",
    "STUCK",
    "TARGET_LOST",
    "WRONG_BLOCK_MINED",
    "NO_STONE_FOUND",
    "NO_PROGRESS",
    "TIMEOUT",
)


def normalize_identifier(value: str) -> str:
    return value.lower().replace("minecraft:", "").replace(" ", "_")


def scalar(value, default: float = 0.0) -> float:
    try:
        array = np.asarray(value)
        if array.size == 0:
            return default
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def numeric_mapping(info: dict, source: str) -> dict[str, float]:
    values = info.get(source, {})
    if not isinstance(values, dict):
        return {}
    result = {}
    for name, value in values.items():
        numeric_value = scalar(value)
        if numeric_value != 0.0:
            result[normalize_identifier(str(name))] = numeric_value
    return result


def inventory_counts(info: dict) -> dict[str, float]:
    counts: dict[str, float] = {}
    values = info.get("inventory", {})
    if not isinstance(values, dict):
        return counts
    for stack in values.values():
        if not isinstance(stack, dict):
            continue
        item_name = normalize_identifier(str(stack.get("type", "air")))
        quantity = scalar(stack.get("quantity", 0))
        if item_name not in {"air", "none"} and quantity > 0:
            counts[item_name] = counts.get(item_name, 0.0) + quantity
    return counts


def grouped_mapping_value(values: dict[str, float], target: str) -> float:
    normalized_target = normalize_identifier(target)
    if normalized_target == "stone":
        return sum(quantity for name, quantity in values.items() if name in {"stone", "deepslate"})
    if normalized_target == "cobblestone":
        return sum(quantity for name, quantity in values.items() if name in {"cobblestone", "cobbled_deepslate"})
    return values.get(normalized_target, 0.0)


def life_stats(info: dict) -> dict:
    values = info.get("life_stats", {})
    return values if isinstance(values, dict) else {}


def voxel_tokens(value) -> list[str]:
    tokens = []
    if isinstance(value, dict):
        for item in value.values():
            tokens.extend(voxel_tokens(item))
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            tokens.extend(voxel_tokens(item))
    elif value is not None:
        tokens.append(normalize_identifier(str(value)))
    return tokens


def voxel_records(value) -> list[dict]:
    records = []
    if isinstance(value, dict):
        if {"x", "y", "z", "type"}.issubset(value):
            records.append(value)
        else:
            for item in value.values():
                records.extend(voxel_records(item))
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            records.extend(voxel_records(item))
    return records


def water_state(info: dict) -> int:
    if "voxels" not in info:
        return -1
    records = voxel_records(info.get("voxels"))
    return int(any(
        (normalize_identifier(str(record.get("type", ""))).endswith("water") or normalize_identifier(str(record.get("type", ""))).endswith("bubble_column"))
        and int(scalar(record.get("x"))) == 0
        and int(scalar(record.get("z"))) == 0
        and int(scalar(record.get("y"))) in {0, 1}
        for record in records
    ))


def state_snapshot(info: dict) -> dict:
    position = info.get("player_pos", {})
    lives = life_stats(info)
    events = {source: numeric_mapping(info, source) for source in EVENT_SOURCES}
    health = scalar(lives.get("life", info.get("health", 20.0)), 20.0)
    air = scalar(lives.get("air", 300.0), 300.0)
    is_alive = bool(scalar(lives.get("is_alive", health > 0), float(health > 0)))
    return {
        "inventory": inventory_counts(info),
        "events": events,
        "position": {
            "x": scalar(position.get("x", 0.0)),
            "y": scalar(position.get("y", 0.0)),
            "z": scalar(position.get("z", 0.0)),
            "yaw": scalar(position.get("yaw", 0.0)),
            "pitch": scalar(position.get("pitch", 0.0)),
        },
        "health": health,
        "air": air,
        "is_alive": is_alive,
        "water": water_state(info),
        "voxels": json_value(info.get("voxels")),
    }


def relative_coordinate(value: int) -> str:
    return "~" if value == 0 else f"~{value}"


def direction_pose(generator: random.Random, distance: int) -> tuple[int, int, float, int, int]:
    direction = generator.randrange(4)
    poses = (
        (0, distance, 0.0, 0, 1),
        (-distance, 0, 90.0, -1, 0),
        (0, -distance, 180.0, 0, -1),
        (distance, 0, -90.0, 1, 0),
    )
    return poses[direction]


def weighted_scenario(generator: random.Random) -> str:
    names = tuple(SCENARIO_WEIGHTS)
    weights = tuple(SCENARIO_WEIGHTS.values())
    return generator.choices(names, weights=weights, k=1)[0]


def frozen_task_prompts(seed: int) -> dict[str, str]:
    if not EVENT_DRIVEN_TASKS:
        return {"stone_acquisition": POLICY_PROMPT}
    payload = json.loads(TASK_BANK_PATH.read_text(encoding="utf-8"))
    prompts = {}
    for category in TASK_CATEGORIES:
        values = payload["tasks"][category]
        digest = hashlib.sha256(f"{seed}:{category}".encode("utf-8")).digest()
        prompts[category] = values[int.from_bytes(digest[:8], "big") % len(values)]
    return prompts


def make_episode_config(episode_index: int) -> dict:
    seed = BASE_SEED + episode_index * 7919
    generator = random.Random(seed)
    stress_stratum = None
    if MANIFEST_KIND == "standard":
        scenario = weighted_scenario(generator)
        biome = BIOMES[episode_index % len(BIOMES)]
    elif MANIFEST_KIND == "general_natural":
        scenario = "natural"
        biome = BIOMES[episode_index % len(BIOMES)]
    elif MANIFEST_KIND == "hazard_stress":
        stress_group = episode_index % 3
        if stress_group == 0:
            scenario = "water_near_stone"
            biome = BIOMES[(episode_index // 3) % len(BIOMES)]
            stress_stratum = "water_near_stone"
        elif stress_group == 1:
            scenario = "water_near_stone"
            biome = "swamp"
            stress_stratum = "swamp_water"
        else:
            scenario = "awkward_shore"
            biome = "swamp" if (episode_index // 3) % 2 == 0 else "forest"
            stress_stratum = "awkward_shore"
    else:
        raise ValueError(f"Unknown manifest kind: {MANIFEST_KIND}")
    distance_by_scenario = {
        "natural": generator.randint(4, 10),
        "exposed_near": generator.randint(3, 5),
        "exposed_far": generator.randint(7, 10),
        "water_near_stone": generator.randint(6, 8),
        "awkward_shore": generator.randint(6, 8),
        "stone_slope": generator.randint(4, 7),
        "vegetation_occluded": generator.randint(5, 8),
    }
    distance = distance_by_scenario[scenario]
    target_x, target_z, target_yaw, unit_x, unit_z = direction_pose(generator, distance)
    if scenario == "natural":
        visibility = "unknown_natural"
        initial_yaw = generator.uniform(-180.0, 180.0)
        initial_pitch = generator.uniform(-22.0, 36.0)
    else:
        visible = generator.random() < 0.55 and scenario != "vegetation_occluded"
        visibility = "initially_visible" if visible else "initially_hidden"
        initial_yaw = target_yaw + generator.uniform(-14.0, 14.0) if visible else target_yaw + generator.uniform(125.0, 235.0)
        initial_pitch = generator.uniform(6.0, 20.0) if visible else generator.uniform(-22.0, 36.0)
    initial_yaw = ((initial_yaw + 180.0) % 360.0) - 180.0
    hotbar_slot = generator.randrange(9)
    minor_inventory = [list(item) for item in generator.choice(MINOR_INVENTORIES)]
    task_prompts = frozen_task_prompts(seed)
    terrain = {
        "natural": "natural",
        "exposed_near": "flat",
        "exposed_far": "flat",
        "water_near_stone": "flat_with_shallow_water",
        "awkward_shore": "shallow_water_with_raised_irregular_shore",
        "stone_slope": "constructed_slope",
        "vegetation_occluded": "flat_with_vegetation",
    }[scenario]
    return {
        "episode_id": f"stone:{episode_index:05d}",
        "episode_index": episode_index,
        "seed": seed,
        "world_seed": seed,
        "biome": biome,
        "scenario": scenario,
        "scenario_family": "natural" if scenario == "natural" else "controlled_stratum",
        "stone_visibility": visibility,
        "stone_distance_blocks": None if scenario == "natural" else distance,
        "terrain": terrain,
        "water_proximity": "between_player_and_stone" if scenario == "water_near_stone" else "player_starts_in_water" if scenario == "awkward_shore" else "natural" if scenario == "natural" else "none_constructed",
        "vegetation": "occluding" if scenario == "vegetation_occluded" else "natural" if scenario == "natural" else "cleared",
        "stress_stratum": stress_stratum,
        "target": {"x": target_x, "z": target_z, "yaw": target_yaw, "unit_x": unit_x, "unit_z": unit_z},
        "initial_yaw": initial_yaw,
        "initial_pitch": initial_pitch,
        "hotbar_slot": hotbar_slot,
        "minor_inventory": minor_inventory,
        "active_goal": GLOBAL_GOAL,
        "policy_prompt": task_prompts["stone_acquisition"],
        "task_prompts": task_prompts,
    }


def make_manifest(episodes: int | None = None) -> list[dict]:
    episodes = EPISODES if episodes is None else episodes
    return [make_episode_config(episode_index) for episode_index in range(episodes)]


def scenario_commands(config: dict) -> list[str]:
    commands = [
        "/gamerule sendCommandFeedback false",
        "/clear @p",
        "/effect clear @p",
        "/time set day",
        "/weather clear",
    ]
    for item_name, quantity in config["minor_inventory"]:
        commands.append(f"/give @p minecraft:{item_name} {quantity}")
    commands.extend([
        f"/replaceitem entity @p hotbar.{config['hotbar_slot']} minecraft:wooden_pickaxe 1",
        f"/tp @p ~ ~ ~ {config['initial_yaw']:.2f} {config['initial_pitch']:.2f}",
    ])
    if config["scenario"] == "natural":
        return commands
    radius = 11
    commands.extend([
        f"/fill ~-{radius} ~-4 ~-{radius} ~{radius} ~-2 ~{radius} minecraft:dirt",
        f"/fill ~-{radius} ~-1 ~-{radius} ~{radius} ~-1 ~{radius} minecraft:grass_block",
        f"/fill ~-{radius} ~ ~-{radius} ~{radius} ~6 ~{radius} minecraft:air",
    ])
    target = config["target"]
    target_x = int(target["x"])
    target_z = int(target["z"])
    unit_x = int(target["unit_x"])
    unit_z = int(target["unit_z"])
    perpendicular_x = -unit_z
    perpendicular_z = unit_x
    stone_positions = [
        (target_x, 0, target_z),
        (target_x, 1, target_z),
        (target_x + perpendicular_x, 0, target_z + perpendicular_z),
        (target_x + perpendicular_x, 1, target_z + perpendicular_z),
    ]
    if config["scenario"] == "water_near_stone":
        center_x = unit_x * 3
        center_z = unit_z * 3
        water_positions = []
        for forward_offset in (-1, 0, 1):
            for side_offset in (-1, 0, 1):
                water_x = center_x + unit_x * forward_offset + perpendicular_x * side_offset
                water_z = center_z + unit_z * forward_offset + perpendicular_z * side_offset
                water_positions.append((water_x, water_z))
        for water_x, water_z in water_positions:
            commands.append(f"/setblock {relative_coordinate(water_x)} ~ {relative_coordinate(water_z)} minecraft:water")
    if config["scenario"] == "awkward_shore":
        for forward_offset in range(-1, 4):
            for side_offset in range(-2, 3):
                water_x = unit_x * forward_offset + perpendicular_x * side_offset
                water_z = unit_z * forward_offset + perpendicular_z * side_offset
                commands.append(f"/setblock {relative_coordinate(water_x)} ~ {relative_coordinate(water_z)} minecraft:water")
        for side_offset in (-2, -1, 0, 2):
            shore_x = unit_x * 4 + perpendicular_x * side_offset
            shore_z = unit_z * 4 + perpendicular_z * side_offset
            commands.append(f"/setblock {relative_coordinate(shore_x)} ~ {relative_coordinate(shore_z)} minecraft:dirt")
    if config["scenario"] == "stone_slope":
        for slope_step in range(2, max(3, int(config["stone_distance_blocks"]) - 1)):
            slope_x = unit_x * slope_step
            slope_z = unit_z * slope_step
            height = 0 if slope_step < 4 else 1
            commands.append(f"/setblock {relative_coordinate(slope_x)} {relative_coordinate(height)} {relative_coordinate(slope_z)} minecraft:stone")
    if config["scenario"] == "vegetation_occluded":
        obstacle_x = unit_x * max(2, int(config["stone_distance_blocks"]) - 2)
        obstacle_z = unit_z * max(2, int(config["stone_distance_blocks"]) - 2)
        for height in (0, 1, 2):
            for side_offset in (-1, 0, 1):
                leaf_x = obstacle_x + perpendicular_x * side_offset
                leaf_z = obstacle_z + perpendicular_z * side_offset
                if not (height == 0 and side_offset == 0):
                    commands.append(f"/setblock {relative_coordinate(leaf_x)} {relative_coordinate(height)} {relative_coordinate(leaf_z)} minecraft:oak_leaves")
    for stone_x, stone_y, stone_z in stone_positions:
        commands.append(f"/setblock {relative_coordinate(stone_x)} {relative_coordinate(stone_y)} {relative_coordinate(stone_z)} minecraft:stone")
    return commands


def movement_requested(environment_action: dict) -> int:
    return int(any(scalar(environment_action.get(name, 0.0)) != 0.0 for name in MOVEMENT_BUTTONS))


def agent_action_codes(action: dict) -> tuple[int, int]:
    return int(scalar(action.get("buttons", 0))), int(scalar(action.get("camera", 60), 60))


class TrajectoryRecorder:
    def __init__(self, config: dict):
        self.config = config
        self.frames = []
        self.agent_buttons = []
        self.agent_camera = []
        self.env_buttons = []
        self.env_camera = []
        self.positions = []
        self.health = []
        self.air = []
        self.is_alive = []
        self.water = []
        self.cobblestone = []
        self.stone_mined = []
        self.cobblestone_pickup = []
        self.movement = []
        self.phases = []
        self.inventories = []
        self.events = []
        self.voxels = []
        self.blocked_controls = Counter()

    def append(self, frame: np.ndarray, info: dict, agent_action: dict, environment_action: dict, phase: str, blocked_controls: list[str] | None = None) -> None:
        state = state_snapshot(info)
        button_code, camera_code = agent_action_codes(agent_action)
        camera = np.asarray(environment_action.get("camera", [0.0, 0.0]), dtype=np.float32).reshape(-1)
        if camera.size < 2:
            camera = np.pad(camera, (0, 2 - camera.size))
        self.frames.append(np.asarray(frame, dtype=np.uint8).copy())
        self.agent_buttons.append(button_code)
        self.agent_camera.append(camera_code)
        self.env_buttons.append([int(scalar(environment_action.get(name, 0.0)) != 0.0) for name in ENV_BUTTONS])
        self.env_camera.append(camera[:2])
        self.positions.append([state["position"][name] for name in ("x", "y", "z", "yaw", "pitch")])
        self.health.append(state["health"])
        self.air.append(state["air"])
        self.is_alive.append(state["is_alive"])
        self.water.append(state["water"])
        self.cobblestone.append(grouped_mapping_value(state["inventory"], "cobblestone"))
        self.stone_mined.append(grouped_mapping_value(state["events"]["mine_block"], "stone"))
        self.cobblestone_pickup.append(grouped_mapping_value(state["events"]["pickup"], "cobblestone"))
        self.movement.append(movement_requested(environment_action))
        self.phases.append(phase)
        self.inventories.append(state["inventory"])
        self.events.append(state["events"])
        self.voxels.append(state["voxels"])
        self.blocked_controls.update(blocked_controls or [])

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "rgb": np.asarray(self.frames, dtype=np.uint8),
            "agent_buttons": np.asarray(self.agent_buttons, dtype=np.int32),
            "agent_camera": np.asarray(self.agent_camera, dtype=np.int16),
            "env_buttons": np.asarray(self.env_buttons, dtype=np.uint8),
            "env_camera": np.asarray(self.env_camera, dtype=np.float32),
            "position": np.asarray(self.positions, dtype=np.float32),
            "health": np.asarray(self.health, dtype=np.float32),
            "air": np.asarray(self.air, dtype=np.float32),
            "is_alive": np.asarray(self.is_alive, dtype=np.bool_),
            "water": np.asarray(self.water, dtype=np.int8),
            "cobblestone": np.asarray(self.cobblestone, dtype=np.float32),
            "stone_mined": np.asarray(self.stone_mined, dtype=np.float32),
            "cobblestone_pickup": np.asarray(self.cobblestone_pickup, dtype=np.float32),
            "movement_requested": np.asarray(self.movement, dtype=np.uint8),
        }


def first_progress_step(values: np.ndarray, baseline: float, threshold: float) -> int | None:
    matches = np.flatnonzero(values - baseline >= threshold)
    return int(matches[0]) if len(matches) else None


def stuck_windows(position: np.ndarray, movement: np.ndarray) -> list[dict]:
    windows = []
    if len(position) < STUCK_WINDOW:
        return windows
    for end in range(STUCK_WINDOW, len(position) + 1, STUCK_WINDOW // 4):
        start = end - STUCK_WINDOW
        movement_fraction = float(movement[start:end].mean())
        displacement = float(math.hypot(position[end - 1, 0] - position[start, 0], position[end - 1, 2] - position[start, 2]))
        if movement_fraction >= STUCK_MOVEMENT_FRACTION and displacement < STUCK_DISPLACEMENT:
            windows.append({"start": start, "end": end - 1, "movement_fraction": movement_fraction, "horizontal_displacement": displacement})
    return windows


def future_any(binary_values: np.ndarray, horizon: int) -> np.ndarray:
    result = np.zeros(len(binary_values), dtype=np.uint8)
    for index in range(len(binary_values)):
        result[index] = int(np.any(binary_values[index + 1:min(len(binary_values), index + horizon + 1)]))
    return result


def auxiliary_labels(arrays: dict[str, np.ndarray], final: dict) -> dict[str, np.ndarray]:
    water = arrays["water"] == 1
    air = np.r_[arrays["air"], final["air"]]
    final_cobblestone = grouped_mapping_value(final["inventory"], "cobblestone")
    final_stone_mined = grouped_mapping_value(final["events"]["mine_block"], "stone")
    cobblestone = np.r_[arrays["cobblestone"], final_cobblestone]
    stone_mined = np.r_[arrays["stone_mined"], final_stone_mined]
    progress = np.zeros(len(air), dtype=np.uint8)
    if len(progress) > 1:
        progress[1:] = np.logical_or(
            np.diff(cobblestone) > 0,
            np.diff(stone_mined) > 0,
        ).astype(np.uint8)
    air_decrease = np.zeros(len(air), dtype=np.uint8)
    if len(air) > 1:
        air_decrease[1:] = (np.diff(air) < 0).astype(np.uint8)
    return {
        "will_enter_water_40": future_any(np.r_[water, final["water"] == 1], AUXILIARY_HORIZON)[:-1],
        "will_air_decrease_40": future_any(air_decrease, AUXILIARY_HORIZON)[:-1],
        "will_make_progress_40": future_any(progress, AUXILIARY_HORIZON)[:-1],
    }


def event_delta(baseline: dict, final: dict, source: str, target: str) -> float:
    return grouped_mapping_value(final["events"][source], target) - grouped_mapping_value(baseline["events"][source], target)


def classify_trajectory(arrays: dict[str, np.ndarray], baseline: dict, final: dict, terminated: bool, timed_out: bool, step_budget: int) -> dict:
    baseline_cobble = grouped_mapping_value(baseline["inventory"], "cobblestone")
    final_cobble = grouped_mapping_value(final["inventory"], "cobblestone")
    cobble_delta = final_cobble - baseline_cobble
    stone_mined = event_delta(baseline, final, "mine_block", "stone")
    mined_values = final["events"]["mine_block"]
    baseline_mined_values = baseline["events"]["mine_block"]
    wrong_blocks = 0.0
    wrong_block_types = {}
    for block_name in set(mined_values) | set(baseline_mined_values):
        if block_name in {"stone", "deepslate"}:
            continue
        delta = mined_values.get(block_name, 0.0) - baseline_mined_values.get(block_name, 0.0)
        if delta > 0:
            wrong_blocks += delta
            wrong_block_types[block_name] = delta
    labels = []
    evidence = {}

    def add(label: str, source: str, details: dict) -> None:
        if label not in labels:
            labels.append(label)
            evidence[label] = {"source": source, **details}

    success = cobble_delta >= TARGET_COBBLESTONE
    water_values = np.r_[arrays["water"], final["water"]]
    entered_water = bool(np.any(water_values == 1))
    air_values = np.r_[arrays["air"], final["air"]]
    health_values = np.r_[arrays["health"], final["health"]]
    minimum_air = float(np.min(air_values)) if len(air_values) else final["air"]
    initial_air = float(air_values[0]) if len(air_values) else baseline["air"]
    air_decreased = minimum_air < initial_air
    health_decreased_in_water = False
    if len(health_values) > 1:
        health_decrease = np.r_[False, np.diff(health_values) < 0]
        health_decreased_in_water = bool(np.any(np.logical_and(health_decrease, water_values == 1)))
    died = not final["is_alive"] or final["health"] <= 0.0
    windows = stuck_windows(arrays["position"], arrays["movement_requested"])
    baseline_y = baseline["position"]["y"]
    minimum_y = float(np.min(arrays["position"][:, 1])) if len(arrays["position"]) else final["position"]["y"]
    final_y = final["position"]["y"]
    trapped_hole = minimum_y <= baseline_y - 1.5 and final_y <= baseline_y - 0.8 and bool(windows)
    stone_series = np.r_[arrays["stone_mined"], grouped_mapping_value(final["events"]["mine_block"], "stone")]
    cobble_series = np.r_[arrays["cobblestone"], final_cobble]
    progress_indices = []
    if len(stone_series) > 1:
        progress_indices.extend((np.flatnonzero(np.diff(stone_series) > 0) + 1).tolist())
    if len(cobble_series) > 1:
        progress_indices.extend((np.flatnonzero(np.diff(cobble_series) > 0) + 1).tolist())
    last_progress_step = max(progress_indices) if progress_indices else None
    observed_steps = len(arrays["stone_mined"])
    progress_stale = observed_steps >= NO_PROGRESS_WINDOW and (last_progress_step is None or len(stone_series) - 1 - last_progress_step >= NO_PROGRESS_WINDOW)
    target_lost = stone_mined > 0 and cobble_delta < TARGET_COBBLESTONE and progress_stale
    if success:
        add("SUCCESS", "exact_inventory_delta", {"cobblestone_delta": cobble_delta})
    if died:
        add("DIED", "exact_life_state", {"health": final["health"], "terminated": terminated})
    if air_decreased or health_decreased_in_water:
        add("DROWNING", "exact_air_or_health_sequence", {"initial_air": initial_air, "minimum_air": minimum_air, "health_damage_in_water": health_decreased_in_water})
    if stone_mined > cobble_delta and not success:
        add("STONE_BROKEN_NOT_COLLECTED", "exact_event_inventory_delta", {"stone_mined_delta": stone_mined, "cobblestone_delta": cobble_delta})
    if entered_water:
        add("ENTERED_WATER", "exact_local_voxel_observation", {"water_steps": int(np.sum(water_values == 1))})
    if trapped_hole:
        add("TRAPPED_IN_HOLE", "trajectory_heuristic", {"baseline_y": baseline_y, "minimum_y": minimum_y, "final_y": final_y})
    if windows:
        add("STUCK", "trajectory_heuristic", {"windows": windows})
    if target_lost:
        add("TARGET_LOST", "event_progress_heuristic", {"last_progress_step": last_progress_step, "tail_steps": len(stone_series) - 1 - int(last_progress_step)})
    if wrong_blocks >= WRONG_BLOCK_MINED_MIN and not success:
        add("WRONG_BLOCK_MINED", "exact_block_break_events", {"count": wrong_blocks, "blocks": wrong_block_types})
    if stone_mined <= 0 and not success:
        add("NO_STONE_FOUND", "no_stone_break_proxy", {"stone_mined_delta": stone_mined})
    if progress_stale and not success:
        add("NO_PROGRESS", "event_progress_heuristic", {"last_progress_step": last_progress_step, "window": NO_PROGRESS_WINDOW})
    if timed_out:
        add("TIMEOUT", "exact_step_budget", {"steps": observed_steps, "budget": step_budget})
    if not labels:
        add("TIMEOUT", "fallback_terminal_label", {"steps": len(stone_series)})
    primary_label = next(label for label in PRIMARY_LABEL_ORDER if label in labels)
    water_recovered = entered_water and final["water"] == 0 and final["air"] >= minimum_air
    first_water_step = int(np.flatnonzero(water_values == 1)[0]) if entered_water else None
    water_exit_step = None
    if entered_water:
        stable_land_steps = 10
        for index in range(first_water_step + 1, len(water_values) - stable_land_steps + 1):
            if np.all(water_values[index:index + stable_land_steps] == 0):
                water_exit_step = index
                break
    water_exit_success = water_exit_step is not None
    stone_reacquired_after_water = False
    if water_exit_success and len(stone_series) > 1:
        stone_progress_steps = np.flatnonzero(np.diff(stone_series) > 0) + 1
        stone_reacquired_after_water = bool(np.any(stone_progress_steps > water_exit_step))
    hole_recovered = minimum_y <= baseline_y - 1.5 and final_y >= baseline_y - 0.5
    hazard_eligible = entered_water or minimum_y <= baseline_y - 1.5
    hazard_recovered = bool(water_recovered or hole_recovered)
    return {
        "success": success,
        "primary_label": primary_label,
        "labels": labels,
        "evidence": evidence,
        "cobblestone_delta": cobble_delta,
        "stone_mined_delta": stone_mined,
        "wrong_blocks_mined": wrong_blocks,
        "entered_water": entered_water,
        "success_after_entering_water": bool(entered_water and success),
        "water_exit_success": water_exit_success,
        "water_exit_step": water_exit_step,
        "stone_reacquired_after_water": stone_reacquired_after_water,
        "timeout_after_water": bool(entered_water and timed_out),
        "drowning": "DROWNING" in labels,
        "died": died,
        "stuck": "STUCK" in labels,
        "trapped_in_hole": "TRAPPED_IN_HOLE" in labels,
        "stone_contact_proxy": stone_mined > 0,
        "hazard_recovery_eligible": hazard_eligible,
        "hazard_recovered": hazard_recovered,
    }


def write_trajectory(output_directory: Path, config: dict, recorder: TrajectoryRecorder, baseline: dict, final: dict, classification: dict, option_records: list[dict], task_transitions: list[dict], terminated: bool, truncated: bool, elapsed_seconds: float) -> tuple[str, str]:
    trajectory_directory = output_directory / "trajectories"
    trajectory_directory.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{config['episode_index']:05d}_{config['seed']}"
    arrays = recorder.arrays()
    arrays.update(auxiliary_labels(arrays, final))
    npz_path = trajectory_directory / f"{stem}.npz"
    npz_temporary = trajectory_directory / f"{stem}.npz.tmp"
    with npz_temporary.open("wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    npz_temporary.replace(npz_path)
    metadata = {
        "schema_version": 1,
        "transition_contract": "row t stores observation/state before action t",
        "config": config,
        "active_goal": GLOBAL_GOAL,
        "policy_prompt": config["policy_prompt"],
        "task_prompts": config["task_prompts"],
        "task_transitions": task_transitions,
        "environment_button_columns": list(ENV_BUTTONS),
        "position_columns": ["x", "y", "z", "yaw", "pitch"],
        "phases": recorder.phases,
        "inventories": recorder.inventories,
        "cumulative_events": recorder.events,
        "local_voxels": recorder.voxels,
        "blocked_controls": dict(recorder.blocked_controls),
        "option_records": option_records,
        "baseline_state": baseline,
        "final_state": final,
        "classification": classification,
        "steps": len(recorder.frames),
        "terminated": terminated,
        "truncated": truncated,
        "elapsed_seconds": elapsed_seconds,
    }
    metadata_path = trajectory_directory / f"{stem}.json.gz"
    metadata_temporary = trajectory_directory / f"{stem}.json.gz.tmp"
    with gzip.open(metadata_temporary, "wt", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, separators=(",", ":"))
    metadata_temporary.replace(metadata_path)
    return str(npz_path.relative_to(output_directory)), str(metadata_path.relative_to(output_directory))


def create_simulator(config: dict, command_callback: CommandsCallback) -> MinecraftSim:
    return MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=config["seed"],
        preferred_spawn_biome=config["biome"],
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[command_callback, VoxelsCallback([-1, 1, -1, 2, -1, 1])],
        include_life_stats=True,
    )


def run_episode(model: SteveOnePolicy, simulator: MinecraftSim, command_callback: CommandsCallback, config: dict, output_directory: Path, max_steps: int = MAX_STEPS) -> dict:
    random.seed(config["seed"])
    np.random.seed(config["seed"] % (2**32))
    torch.manual_seed(config["seed"])
    command_callback.commands = scenario_commands(config)
    simulator.seed = config["seed"]
    simulator.env.seed(config["seed"])
    observation, info = simulator.reset()
    setup_steps = 0
    for setup_step in range(1, MAX_SETUP_SETTLE_STEPS + 1):
        observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
        setup_steps = setup_step
        pick_ready = grouped_inventory_value(info, "wooden_pickaxe") >= 1
        state_ready = "life_stats" in info and bool(voxel_tokens(info.get("voxels")))
        if setup_step >= SETUP_SETTLE_STEPS and pick_ready and state_ready:
            break
    if grouped_inventory_value(info, "wooden_pickaxe") < 1:
        raise RuntimeError(f"Wooden pickaxe setup failed for {config['episode_id']}: {inventory_counts(info)}")
    if "life_stats" not in info or not voxel_tokens(info.get("voxels")):
        raise RuntimeError(f"Required hazard instrumentation missing for {config['episode_id']}: life_stats={'life_stats' in info}, voxel_count={len(voxel_tokens(info.get('voxels')))}")
    baseline = state_snapshot(info)
    recorder = TrajectoryRecorder(config)
    option_records = []
    terminated = False
    truncated = False
    environment_error = None
    started = time.monotonic()

    def recorded_step(agent_action: dict, phase: str):
        nonlocal observation, info, terminated, truncated, environment_error
        environment_action = simulator.agent_action_to_env_action(deepcopy(agent_action))
        recorder.append(observation["image"], info, agent_action, environment_action, phase)
        observation, reward, terminated, truncated, info = simulator.step(agent_action)
        environment_error = info.get("error")
        return observation, reward, terminated, truncated, info

    controller = ClosedLoopOptions(simulator, recorded_step, pickup_implementation=PICKUP_IMPLEMENTATION)
    equip_result, info = controller.equip_item("wooden_pickaxe", info)
    option_records.append(equip_result)
    if not equip_result["success"]:
        raise RuntimeError(f"EQUIP_ITEM failed for {config['episode_id']}: {equip_result}")
    recurrent_state = None
    conditions = {
        category: model.prepare_condition(
            {"cond_scale": CONDITION_SCALE, "text": prompt},
            deterministic=DETERMINISTIC_TEXT_PRIOR,
        )
        for category, prompt in config["task_prompts"].items()
    }
    active_task_category = None
    task_transitions = []
    baseline_y = baseline["position"]["y"]
    recent_positions = deque(maxlen=40)
    router_class = LegacyStoneTaskRouter if ROUTER_IMPLEMENTATION == "old" else StoneTaskRouter
    task_router = router_class(baseline_y)
    last_stone_mined = grouped_mapping_value(state_snapshot(info)["events"]["mine_block"], "stone")
    while len(recorder.frames) < max_steps and not terminated and not truncated and environment_error is None:
        cobble_before = grouped_inventory_value(info, "cobblestone")
        pre_action_state = state_snapshot(info)
        recent_positions.append((pre_action_state["position"]["x"], pre_action_state["position"]["y"], pre_action_state["position"]["z"]))
        pre_action_stone_mined = grouped_mapping_value(pre_action_state["events"]["mine_block"], "stone")
        if not EVENT_DRIVEN_TASKS:
            next_task_category = "stone_acquisition"
        else:
            low_displacement = len(recent_positions) == recent_positions.maxlen and math.dist(recent_positions[0], recent_positions[-1]) < 0.6
            routing = task_router.update(pre_action_state, len(recorder.frames), low_displacement, pre_action_stone_mined)
            next_task_category = routing.category
        if next_task_category != active_task_category:
            active_task_category = next_task_category
            recurrent_state = None
            task_transitions.append({
                "step": len(recorder.frames),
                "category": active_task_category,
                "instruction": config["task_prompts"][active_task_category],
            })
        condition = conditions[active_task_category]
        image = torch.from_numpy(observation["image"]).unsqueeze(0).unsqueeze(0).to("cuda")
        model_input = {"image": image, "condition": condition}
        batched_action, recurrent_state = model.get_action(
            model_input,
            recurrent_state,
            deterministic=DETERMINISTIC_ACTIONS,
            input_shape="BT*",
        )
        raw_action = {name: value[0][0] for name, value in batched_action.items()}
        action, environment_action, blocked_controls = controller.gate_world_control(raw_action)
        recorder.blocked_controls.update(blocked_controls)
        observation, reward, terminated, truncated, info = recorded_step(action, f"STEVE_1/{active_task_category}")
        current = state_snapshot(info)
        current_stone_mined = grouped_mapping_value(current["events"]["mine_block"], "stone")
        current_cobble = grouped_inventory_value(info, "cobblestone")
        if current_stone_mined > last_stone_mined and current_cobble <= cobble_before and len(recorder.frames) < max_steps and not terminated and not truncated:
            collect_budget = min(MAX_COLLECT_STEPS, max_steps - len(recorder.frames))
            try:
                collect_result, info = controller.collect_drop("cobblestone", cobble_before, info, max_steps=collect_budget)
                option_records.append(collect_result)
            except RuntimeError as error:
                option_records.append({"option": "COLLECT_DROP", "success": False, "reason": str(error), "steps": controller.steps})
            recurrent_state = None
            current = state_snapshot(info)
            current_cobble = grouped_inventory_value(info, "cobblestone")
        last_stone_mined = current_stone_mined
        if current_cobble - grouped_mapping_value(baseline["inventory"], "cobblestone") >= TARGET_COBBLESTONE:
            break
    if environment_error is not None:
        raise RuntimeError(f"Environment error for {config['episode_id']}: {environment_error}")
    final = state_snapshot(info)
    arrays = recorder.arrays()
    timed_out = len(recorder.frames) >= max_steps and grouped_mapping_value(final["inventory"], "cobblestone") - grouped_mapping_value(baseline["inventory"], "cobblestone") < TARGET_COBBLESTONE
    classification = classify_trajectory(arrays, baseline, final, terminated, timed_out, max_steps)
    elapsed_seconds = time.monotonic() - started
    npz_file, metadata_file = write_trajectory(
        output_directory,
        config,
        recorder,
        baseline,
        final,
        classification,
        option_records,
        task_transitions,
        terminated,
        truncated,
        elapsed_seconds,
    )
    final_cobblestone = grouped_mapping_value(final["inventory"], "cobblestone")
    cobblestone_series = np.r_[arrays["cobblestone"], final_cobblestone]
    success_step = first_progress_step(cobblestone_series, grouped_mapping_value(baseline["inventory"], "cobblestone"), TARGET_COBBLESTONE)
    return {
        "result_id": config["episode_id"],
        "episode_index": config["episode_index"],
        "seed": config["seed"],
        "biome": config["biome"],
        "scenario": config["scenario"],
        "scenario_family": config["scenario_family"],
        "stone_visibility": config["stone_visibility"],
        "terrain": config["terrain"],
        "water_proximity": config["water_proximity"],
        "stress_stratum": config.get("stress_stratum"),
        "hotbar_slot": config["hotbar_slot"],
        "minor_inventory": config["minor_inventory"],
        "success": classification["success"],
        "primary_label": classification["primary_label"],
        "labels": classification["labels"],
        "label_evidence": classification["evidence"],
        "steps": len(recorder.frames),
        "success_step": success_step,
        "environment_seconds_to_success": success_step / STEPS_PER_SECOND if success_step is not None else None,
        "episode_seconds": elapsed_seconds,
        "setup_steps": setup_steps,
        "terminated": terminated,
        "truncated": truncated,
        "cobblestone_delta": classification["cobblestone_delta"],
        "stone_mined_delta": classification["stone_mined_delta"],
        "wrong_blocks_mined": classification["wrong_blocks_mined"],
        "entered_water": classification["entered_water"],
        "success_after_entering_water": classification["success_after_entering_water"],
        "water_exit_success": classification["water_exit_success"],
        "water_exit_step": classification["water_exit_step"],
        "stone_reacquired_after_water": classification["stone_reacquired_after_water"],
        "timeout_after_water": classification["timeout_after_water"],
        "drowning": classification["drowning"],
        "died": classification["died"],
        "stuck": classification["stuck"],
        "trapped_in_hole": classification["trapped_in_hole"],
        "stone_contact_proxy": classification["stone_contact_proxy"],
        "hazard_recovery_eligible": classification["hazard_recovery_eligible"],
        "hazard_recovered": classification["hazard_recovered"],
        "planner_retries": 0,
        "task_transitions": task_transitions,
        "router_diagnostics": task_router.diagnostics() if EVENT_DRIVEN_TASKS else {},
        "router_implementation": ROUTER_IMPLEMENTATION,
        "pickup_implementation": PICKUP_IMPLEMENTATION,
        "blocked_controls": dict(recorder.blocked_controls),
        "option_records": option_records,
        "trajectory_npz": npz_file,
        "trajectory_metadata": metadata_file,
        "baseline_state": baseline,
        "final_state": final,
    }


def run_worker(worker_index: int, assignments: list[dict], experiment_directory: str, max_steps: int, result_queue: multiprocessing.Queue) -> None:
    current_config = None
    simulator = None
    try:
        torch.set_float32_matmul_precision("high")
        model = SteveOnePolicy.from_pretrained(CHECKPOINT_DIRECTORY).to("cuda").eval()
        command_callback = CommandsCallback([])
        simulator_biome = None
        simulator_episodes = 0
        output_directory = Path(experiment_directory)
        for config in assignments:
            current_config = config
            needs_new_simulator = simulator is None or simulator_biome != config["biome"] or simulator_episodes >= MAX_EPISODES_PER_SIMULATOR
            if needs_new_simulator:
                if simulator is not None:
                    simulator.close()
                command_callback = CommandsCallback([])
                simulator = create_simulator(config, command_callback)
                simulator_biome = config["biome"]
                simulator_episodes = 0
            result = run_episode(model, simulator, command_callback, config, output_directory, max_steps=max_steps)
            simulator_episodes += 1
            result["worker_index"] = worker_index
            result_queue.put({"kind": "episode", "result": result})
    except Exception:
        result_queue.put({
            "kind": "error",
            "worker_index": worker_index,
            "episode_id": current_config["episode_id"] if current_config else None,
            "traceback": traceback.format_exc(),
        })
    finally:
        if simulator is not None:
            simulator.close()


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_json_atomic(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def metrics_for(results: list[dict]) -> dict:
    total = len(results)
    successes = sum(result["success"] for result in results)
    primary_counts = Counter(result["primary_label"] for result in results)
    primary_failure_counts = Counter(result["primary_label"] for result in results if not result["success"])
    label_counts = Counter(label for result in results for label in result["labels"])
    success_steps = [result["success_step"] for result in results if result["success_step"] is not None]
    success_seconds = [result["environment_seconds_to_success"] for result in results if result["environment_seconds_to_success"] is not None]
    hazard_eligible = [result for result in results if result["hazard_recovery_eligible"]]
    hazard_recovered = sum(result["hazard_recovered"] for result in hazard_eligible)
    entered_water_results = [result for result in results if result["entered_water"]]
    entered_water_count = len(entered_water_results)
    return {
        "episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "success_wilson_95": wilson_interval(successes, total),
        "primary_outcome_distribution": dict(primary_counts),
        "primary_failure_distribution": dict(primary_failure_counts),
        "multi_label_distribution": dict(label_counts),
        "mean_steps_to_success": mean(success_steps),
        "mean_environment_seconds_to_success": mean(success_seconds),
        "mean_wall_seconds_per_episode": mean([result["episode_seconds"] for result in results]),
        "water_entry_rate": sum(result["entered_water"] for result in results) / total if total else 0.0,
        "entered_water_episodes": entered_water_count,
        "success_given_entered_water": sum(result.get("success_after_entering_water", False) for result in entered_water_results) / entered_water_count if entered_water_count else None,
        "successful_water_exit_rate": sum(result.get("water_exit_success", False) for result in entered_water_results) / entered_water_count if entered_water_count else None,
        "stone_reacquisition_after_water_rate": sum(result.get("stone_reacquired_after_water", False) for result in entered_water_results) / entered_water_count if entered_water_count else None,
        "timeout_after_water_rate": sum(result.get("timeout_after_water", False) for result in entered_water_results) / entered_water_count if entered_water_count else None,
        "drowning_rate": sum(result["drowning"] for result in results) / total if total else 0.0,
        "death_rate": sum(result["died"] for result in results) / total if total else 0.0,
        "stuck_rate": sum(result["stuck"] for result in results) / total if total else 0.0,
        "trapped_in_hole_rate": sum(result["trapped_in_hole"] for result in results) / total if total else 0.0,
        "stone_contact_proxy_rate": sum(result["stone_contact_proxy"] for result in results) / total if total else 0.0,
        "stone_break_rate": sum(result["stone_mined_delta"] > 0 for result in results) / total if total else 0.0,
        "cobblestone_acquisition_rate": sum(result["cobblestone_delta"] > 0 for result in results) / total if total else 0.0,
        "mean_cobblestone_acquired": mean([result["cobblestone_delta"] for result in results]),
        "hazard_recovery_episodes": len(hazard_eligible),
        "hazard_recovery_success_rate": hazard_recovered / len(hazard_eligible) if hazard_eligible else None,
        "planner_retries": sum(result["planner_retries"] for result in results),
    }


def build_summary(results: list[dict], started_at: str, elapsed_seconds: float) -> dict:
    ordered = sorted(results, key=lambda result: result["episode_index"])
    by_scenario = {
        scenario: metrics_for([result for result in ordered if result["scenario"] == scenario])
        for scenario in sorted({result["scenario"] for result in ordered})
    }
    by_biome = {
        biome: metrics_for([result for result in ordered if result["biome"] == biome])
        for biome in BIOMES
    }
    stress_strata = sorted({result.get("stress_stratum") for result in ordered if result.get("stress_stratum")})
    by_stress_stratum = {
        stratum: metrics_for([result for result in ordered if result.get("stress_stratum") == stratum])
        for stratum in stress_strata
    }
    return {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "stage": BENCHMARK_STAGE,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "episodes_target": EPISODES,
        "episodes_complete": len(ordered),
        "checkpoint": str(CHECKPOINT_DIRECTORY),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "global_goal": GLOBAL_GOAL,
        "policy_prompt": POLICY_PROMPT,
        "event_driven_tasks": EVENT_DRIVEN_TASKS,
        "router_implementation": ROUTER_IMPLEMENTATION,
        "pickup_implementation": PICKUP_IMPLEMENTATION,
        "task_bank": str(TASK_BANK_PATH) if EVENT_DRIVEN_TASKS else None,
        "task_bank_sha256": hashlib.sha256(TASK_BANK_PATH.read_bytes()).hexdigest() if EVENT_DRIVEN_TASKS else None,
        "target_cobblestone": TARGET_COBBLESTONE,
        "max_steps": MAX_STEPS,
        "base_seed": BASE_SEED,
        "manifest_kind": MANIFEST_KIND,
        "num_workers": NUM_WORKERS,
        "condition_scale": CONDITION_SCALE,
        "deterministic_text_prior": DETERMINISTIC_TEXT_PRIOR,
        "deterministic_actions": DETERMINISTIC_ACTIONS,
        "architecture": f"STEVE-1 world control plus EQUIP_ITEM; pickup={PICKUP_IMPLEMENTATION}; router={ROUTER_IMPLEMENTATION}; frozen event-driven objectives when enabled; no live Gemini calls, no crafting controller, no hidden target coordinates supplied to policy",
        "benchmark_design": "Frozen manifest with separate natural and controlled visibility/hazard strata. Aggregate and stratum metrics are both reported.",
        "metric_note": "stone_contact_proxy is the first exact stone block-break event; it is not claimed to measure visual recognition",
        "source_sha256": {
            "benchmark_stone_acquisition.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "closed_loop_options.py": hashlib.sha256((PROJECT_DIRECTORY / "run" / "closed_loop_options.py").read_bytes()).hexdigest(),
            "simulator_entry.py": hashlib.sha256((PROJECT_DIRECTORY / "minestudio" / "simulator" / "entry.py").read_bytes()).hexdigest(),
            "human_survival_specs.py": hashlib.sha256((PROJECT_DIRECTORY / "minestudio" / "simulator" / "minerl" / "herobraine" / "env_specs" / "human_survival_specs.py").read_bytes()).hexdigest(),
            "lifestats.py": hashlib.sha256((PROJECT_DIRECTORY / "minestudio" / "simulator" / "minerl" / "herobraine" / "hero" / "handlers" / "agent" / "observations" / "lifestats.py").read_bytes()).hexdigest(),
        },
        "overall": metrics_for(ordered),
        "by_scenario": by_scenario,
        "by_biome": by_biome,
        "by_stress_stratum": by_stress_stratum,
        "episode_results_file": "episodes.jsonl",
    }


def load_or_create_manifest(experiment_directory: Path) -> list[dict]:
    manifest_path = experiment_directory / "manifest.jsonl"
    expected = make_manifest()
    if manifest_path.exists():
        existing = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if existing != expected:
            raise RuntimeError(f"Frozen manifest differs from current benchmark definition: {manifest_path}")
        return existing
    manifest_path.write_text("".join(json.dumps(config, separators=(",", ":")) + "\n" for config in expected), encoding="utf-8")
    return expected


def worker_assignments(configs: list[dict], worker_count: int) -> list[list[dict]]:
    ordered = sorted(configs, key=lambda config: (config["biome"], config["episode_index"]))
    assignments = [[] for _ in range(worker_count)]
    for index, config in enumerate(ordered):
        assignments[index % worker_count].append(config)
    for assignment in assignments:
        assignment.sort(key=lambda config: (config["biome"], config["episode_index"]))
    return assignments


def main() -> None:
    if not CHECKPOINT_DIRECTORY.is_dir():
        raise FileNotFoundError(CHECKPOINT_DIRECTORY)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_manifest(EXPERIMENT_DIRECTORY)
    episodes_path = EXPERIMENT_DIRECTORY / "episodes.jsonl"
    summary_path = EXPERIMENT_DIRECTORY / "summary.json"
    existing_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    started_at = existing_summary.get("started_at", datetime.now(timezone.utc).isoformat())
    started_time = time.monotonic()
    results_by_id = {}
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            results_by_id[result["result_id"]] = result
    print(json.dumps({"status": "starting", "resumed": len(results_by_id), "target": EPISODES, "output": str(EXPERIMENT_DIRECTORY)}), flush=True)
    context = multiprocessing.get_context("spawn")
    for wave_index in range(MAX_WORKER_WAVES):
        pending = [config for config in manifest if config["episode_id"] not in results_by_id]
        if not pending:
            break
        worker_count = min(NUM_WORKERS, len(pending))
        assignments = worker_assignments(pending, worker_count)
        result_queue = context.Queue()
        processes = [
            context.Process(target=run_worker, args=(worker_index, assignments[worker_index], str(EXPERIMENT_DIRECTORY), MAX_STEPS, result_queue))
            for worker_index in range(worker_count)
        ]
        for process in processes:
            process.start()
        wave_failed = None
        wave_completed = 0
        while wave_completed < len(pending):
            try:
                message = result_queue.get(timeout=180)
            except queue.Empty:
                dead_processes = [process for process in processes if not process.is_alive() and process.exitcode != 0]
                if dead_processes:
                    wave_failed = {"kind": "error", "traceback": f"Workers exited without a result: {[process.exitcode for process in dead_processes]}"}
                    break
                print(json.dumps({"status": "waiting", "complete": len(results_by_id), "target": EPISODES, "wave": wave_index + 1}), flush=True)
                continue
            if message["kind"] == "error":
                wave_failed = message
                break
            result = message["result"]
            if result["result_id"] in results_by_id:
                continue
            results_by_id[result["result_id"]] = result
            wave_completed += 1
            with episodes_path.open("a", encoding="utf-8") as episodes_file:
                episodes_file.write(json.dumps(result, separators=(",", ":")) + "\n")
            summary = build_summary(list(results_by_id.values()), started_at, time.monotonic() - started_time)
            write_json_atomic(summary_path, summary)
            print(json.dumps({
                "status": "episode_complete",
                "complete": len(results_by_id),
                "target": EPISODES,
                "episode": result["episode_index"],
                "scenario": result["scenario"],
                "biome": result["biome"],
                "success": result["success"],
                "primary_label": result["primary_label"],
                "cobblestone_delta": result["cobblestone_delta"],
                "stone_mined_delta": result["stone_mined_delta"],
                "steps": result["steps"],
                "seconds": round(result["episode_seconds"], 2),
                "wave": wave_index + 1,
            }), flush=True)
        if wave_failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=30)
            with (EXPERIMENT_DIRECTORY / "worker_errors.jsonl").open("a", encoding="utf-8") as error_file:
                error_file.write(json.dumps({"wave": wave_index + 1, **wave_failed}, separators=(",", ":")) + "\n")
            print(json.dumps({"status": "restarting_failed_workers", "complete": len(results_by_id), "wave": wave_index + 1}), flush=True)
            continue
        for process in processes:
            process.join(timeout=30)
    missing = [config["episode_id"] for config in manifest if config["episode_id"] not in results_by_id]
    if missing:
        raise RuntimeError(f"Benchmark exhausted worker restart waves with {len(missing)} missing episodes; first missing: {missing[:10]}")
    summary = build_summary(list(results_by_id.values()), started_at, time.monotonic() - started_time)
    write_json_atomic(summary_path, summary)
    print(json.dumps({"status": "complete", "episodes": len(results_by_id), "success_rate": summary["overall"]["success_rate"], "output": str(EXPERIMENT_DIRECTORY)}), flush=True)


if __name__ == "__main__":
    main()
