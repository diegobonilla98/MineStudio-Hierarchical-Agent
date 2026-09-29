import gzip
import hashlib
import json
import math
import multiprocessing
import os
import queue
import random
import statistics
import time
import traceback
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from benchmark_stone_acquisition import ENV_BUTTONS, TrajectoryRecorder, grouped_mapping_value, normalize_identifier, numeric_mapping, relative_coordinate, scalar, state_snapshot, voxel_records
from closed_loop_options import ClosedLoopOptions
from minestudio.models import SteveOnePolicy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, VoxelsCallback
from stone_task_router import LegacyStoneTaskRouter


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = Path(os.environ.get("MINESTUDIO_MICRO_CHECKPOINT", PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "architecture_sweeps" / "20260917T213751Z" / "upper_lora_rank32" / "best_model"))
OUTPUT_DIRECTORY = Path(os.environ.get("MINESTUDIO_MICRO_OUTPUT", PROJECT_DIRECTORY / "output" / "system1_recovery_micro" / "development"))
TASK_NAME = os.environ.get("MINESTUDIO_MICRO_TASK", "EXIT_WATER").upper()
EPISODES = int(os.environ.get("MINESTUDIO_MICRO_EPISODES", "50"))
NUM_WORKERS = int(os.environ.get("MINESTUDIO_MICRO_WORKERS", "2"))
BASE_SEED = int(os.environ.get("MINESTUDIO_MICRO_BASE_SEED", "2026092501"))
MAX_WORKER_WAVES = 8
MAX_EPISODES_PER_SIMULATOR = 5
CONDITION_SCALE = 6.0
DETERMINISTIC_TEXT_PRIOR = False
DETERMINISTIC_ACTIONS = False
CHECKPOINT_ROLE = os.environ.get("MINESTUDIO_MICRO_CHECKPOINT_ROLE", "immutable Upper LoRA r32 step 6400 champion")
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
SETUP_SETTLE_STEPS = 5
MAX_SETUP_SETTLE_STEPS = 32
STABLE_STEPS = 10
CAMERA_STABLE_STEPS = 8
TASKS = (
    "EXIT_WATER",
    "CLIMB_SHORE",
    "REACQUIRE_STONE",
    "ESCAPE_HOLE",
    "AVOID_DIGGING_TRAP",
    "RECOVER_CAMERA",
    "AVOID_WATER",
)
TASK_PROMPTS = {
    "EXIT_WATER": "Escape the water and reach stable dry land.",
    "CLIMB_SHORE": "Climb out of the water onto the raised stable shore.",
    "REACQUIRE_STONE": "Reacquire the exposed stone target and resume productive mining.",
    "ESCAPE_HOLE": "Escape the local hole and return to stable navigable ground.",
    "AVOID_DIGGING_TRAP": "Mine the exposed stone ahead without digging downward into a terrain trap.",
    "RECOVER_CAMERA": "Restore a useful camera view and center the exposed stone target.",
    "AVOID_WATER": "Reach and mine the exposed stone without entering the water.",
}
TASK_TIMEOUTS = {
    "EXIT_WATER": 320,
    "CLIMB_SHORE": 400,
    "REACQUIRE_STONE": 500,
    "ESCAPE_HOLE": 360,
    "AVOID_DIGGING_TRAP": 500,
    "RECOVER_CAMERA": 240,
    "AVOID_WATER": 600,
}
BIOMES = ("forest", "plains", "taiga", "swamp", "extreme_hills")


def direction_pose(generator: random.Random, distance: int) -> tuple[int, int, float, int, int]:
    direction = generator.randrange(4)
    poses = (
        (0, distance, 0.0, 0, 1),
        (-distance, 0, 90.0, -1, 0),
        (0, -distance, 180.0, 0, -1),
        (distance, 0, -90.0, 1, 0),
    )
    return poses[direction]


def make_episode_config(episode_index: int) -> dict:
    seed = BASE_SEED + episode_index * 7919
    generator = random.Random(seed)
    distance_ranges = {
        "EXIT_WATER": (2, 3),
        "CLIMB_SHORE": (3, 4),
        "REACQUIRE_STONE": (4, 7),
        "ESCAPE_HOLE": (1, 2),
        "AVOID_DIGGING_TRAP": (4, 6),
        "RECOVER_CAMERA": (4, 7),
        "AVOID_WATER": (7, 9),
    }
    minimum_distance, maximum_distance = distance_ranges[TASK_NAME]
    distance = generator.randint(minimum_distance, maximum_distance)
    target_x, target_z, target_yaw, unit_x, unit_z = direction_pose(generator, distance)
    perpendicular_x = -unit_z
    perpendicular_z = unit_x
    if TASK_NAME == "RECOVER_CAMERA":
        initial_yaw = target_yaw + generator.choice((-1.0, 1.0)) * generator.uniform(135.0, 220.0)
        initial_pitch = generator.choice((generator.uniform(-82.0, -62.0), generator.uniform(62.0, 82.0)))
    elif TASK_NAME == "REACQUIRE_STONE":
        initial_yaw = target_yaw + generator.choice((-1.0, 1.0)) * generator.uniform(120.0, 210.0)
        initial_pitch = generator.uniform(-20.0, 42.0)
    elif TASK_NAME == "AVOID_DIGGING_TRAP":
        initial_yaw = target_yaw + generator.uniform(-22.0, 22.0)
        initial_pitch = generator.uniform(38.0, 68.0)
    elif TASK_NAME == "AVOID_WATER":
        initial_yaw = target_yaw + generator.uniform(-12.0, 12.0)
        initial_pitch = generator.uniform(7.0, 20.0)
    elif TASK_NAME == "CLIMB_SHORE":
        initial_yaw = target_yaw + generator.uniform(-35.0, 35.0)
        initial_pitch = generator.uniform(4.0, 24.0)
    else:
        initial_yaw = generator.uniform(-180.0, 180.0)
        initial_pitch = generator.uniform(-24.0, 36.0)
    initial_yaw = ((initial_yaw + 180.0) % 360.0) - 180.0
    return {
        "episode_id": f"{TASK_NAME.lower()}:{episode_index:05d}",
        "episode_index": episode_index,
        "seed": seed,
        "world_seed": seed,
        "task": TASK_NAME,
        "policy_prompt": TASK_PROMPTS[TASK_NAME],
        "timeout_steps": TASK_TIMEOUTS[TASK_NAME],
        "biome": BIOMES[(episode_index // 10) % len(BIOMES)],
        "target": {
            "x": target_x,
            "z": target_z,
            "yaw": target_yaw,
            "unit_x": unit_x,
            "unit_z": unit_z,
            "perpendicular_x": perpendicular_x,
            "perpendicular_z": perpendicular_z,
        },
        "distance": distance,
        "initial_yaw": initial_yaw,
        "initial_pitch": initial_pitch,
        "pool_radius": generator.randint(2, 3),
        "shore_width": generator.randint(2, 3),
        "hole_length": generator.randint(1, 2),
        "bypass_side": generator.choice((-1, 1)),
        "trench_width": generator.randint(2, 3),
    }


def make_manifest() -> list[dict]:
    return [make_episode_config(index) for index in range(EPISODES)]


def setblock(x: int, y: int, z: int, block: str) -> str:
    return f"/setblock {relative_coordinate(x)} {relative_coordinate(y)} {relative_coordinate(z)} minecraft:{block}"


def base_commands(config: dict) -> list[str]:
    radius = 12
    return [
        "/gamerule sendCommandFeedback false",
        "/clear @p",
        "/effect clear @p",
        "/time set day",
        "/weather clear",
        "/gamemode survival @p",
        f"/fill ~-{radius} ~-4 ~-{radius} ~{radius} ~-2 ~{radius} minecraft:dirt",
        f"/fill ~-{radius} ~-1 ~-{radius} ~{radius} ~-1 ~{radius} minecraft:grass_block",
        f"/fill ~-{radius} ~ ~-{radius} ~{radius} ~6 ~{radius} minecraft:air",
        "/replaceitem entity @p hotbar.0 minecraft:wooden_pickaxe 1",
        f"/tp @p ~ ~ ~ {config['initial_yaw']:.2f} {config['initial_pitch']:.2f}",
    ]


def target_wall_commands(config: dict) -> list[str]:
    target = config["target"]
    x = int(target["x"])
    z = int(target["z"])
    side_x = int(target["perpendicular_x"])
    side_z = int(target["perpendicular_z"])
    return [
        setblock(x, 0, z, "stone"),
        setblock(x, 1, z, "stone"),
        setblock(x + side_x, 0, z + side_z, "stone"),
        setblock(x + side_x, 1, z + side_z, "stone"),
    ]


def scenario_commands(config: dict) -> list[str]:
    commands = base_commands(config)
    target = config["target"]
    unit_x = int(target["unit_x"])
    unit_z = int(target["unit_z"])
    side_x = int(target["perpendicular_x"])
    side_z = int(target["perpendicular_z"])
    task = config["task"]
    if task == "EXIT_WATER":
        radius = int(config["pool_radius"])
        for forward in range(-radius, radius + 1):
            for side in range(-radius, radius + 1):
                commands.append(setblock(unit_x * forward + side_x * side, 0, unit_z * forward + side_z * side, "water"))
    elif task == "CLIMB_SHORE":
        width = int(config["shore_width"])
        for forward in range(-2, int(config["distance"])):
            for side in range(-width, width + 1):
                commands.append(setblock(unit_x * forward + side_x * side, 0, unit_z * forward + side_z * side, "water"))
        for forward in range(int(config["distance"]), int(config["distance"]) + 4):
            for side in range(-width, width + 1):
                commands.append(setblock(unit_x * forward + side_x * side, 0, unit_z * forward + side_z * side, "grass_block"))
    elif task == "ESCAPE_HOLE":
        for forward in range(int(config["hole_length"])):
            commands.append(setblock(unit_x * forward, -1, unit_z * forward, "air"))
    elif task == "AVOID_DIGGING_TRAP":
        for forward in range(-1, 2):
            for side in range(-1, 2):
                commands.append(setblock(unit_x * forward + side_x * side, -2, unit_z * forward + side_z * side, "air"))
        commands.extend(target_wall_commands(config))
    elif task in {"REACQUIRE_STONE", "RECOVER_CAMERA"}:
        commands.extend(target_wall_commands(config))
    elif task == "AVOID_WATER":
        trench_width = int(config["trench_width"])
        bypass_side = int(config["bypass_side"])
        for forward in range(2, 2 + trench_width):
            for side in range(-3, 4):
                if side == bypass_side * 3:
                    continue
                commands.append(setblock(unit_x * forward + side_x * side, -1, unit_z * forward + side_z * side, "water"))
        commands.extend(target_wall_commands(config))
    return commands


def voxel_type(state: dict, x: int, y: int, z: int) -> str | None:
    for record in voxel_records(state.get("voxels")):
        if int(scalar(record.get("x"))) == x and int(scalar(record.get("y"))) == y and int(scalar(record.get("z"))) == z:
            return normalize_identifier(str(record.get("type", "")))
    return None


def has_stable_support(state: dict) -> bool:
    block = voxel_type(state, 0, -1, 0)
    return block is not None and block not in {"air", "water", "bubble_column", "cave_air", "void_air"}


def wrapped_angle(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


class MicroVerifier:
    def __init__(self, config: dict, baseline: dict):
        self.config = config
        self.baseline = baseline
        self.task = config["task"]
        self.dry_streak = 0
        self.camera_streak = 0
        self.ever_water = baseline["water"] == 1
        self.ever_trapped = False
        self.minimum_y = float(baseline["position"]["y"])
        self.best_yaw_error = 180.0
        self.best_pitch_error = 180.0
        self.failure_reason = None
        self.success_details = {}
        self.reference_grade_y = float(baseline["position"]["y"] + (1.0 if self.task == "ESCAPE_HOLE" else 0.0))
        self.target_x = float(baseline["position"]["x"] + config["target"]["x"])
        self.target_z = float(baseline["position"]["z"] + config["target"]["z"])
        self.target_y = float(self.reference_grade_y + 0.5)
        self.baseline_stone = grouped_mapping_value(baseline["events"]["mine_block"], "stone")
        self.baseline_wrong = self.wrong_blocks(baseline)

    def wrong_blocks(self, state: dict) -> float:
        mined = state["events"]["mine_block"]
        return sum(value for name, value in mined.items() if name not in {"stone", "deepslate"})

    def camera_errors(self, state: dict) -> tuple[float, float]:
        position = state["position"]
        delta_x = self.target_x - float(position["x"])
        delta_z = self.target_z - float(position["z"])
        horizontal = max(1e-6, math.hypot(delta_x, delta_z))
        target_yaw = math.degrees(math.atan2(-delta_x, delta_z))
        target_pitch = math.degrees(math.atan2(float(position["y"]) + 1.62 - self.target_y, horizontal))
        yaw_error = abs(wrapped_angle(float(position["yaw"]) - target_yaw))
        pitch_error = abs(float(position["pitch"]) - target_pitch)
        return yaw_error, pitch_error

    def update(self, state: dict, step: int) -> tuple[str | None, dict]:
        if not state["is_alive"] or state["health"] <= 0:
            self.failure_reason = "DIED"
            return "FAILURE", {"reason": self.failure_reason}
        self.ever_water = self.ever_water or state["water"] == 1
        self.minimum_y = min(self.minimum_y, float(state["position"]["y"]))
        if float(state["position"]["y"]) <= float(self.baseline["position"]["y"]) - 0.75:
            self.ever_trapped = True
        dry_supported = state["water"] == 0 and has_stable_support(state)
        self.dry_streak = self.dry_streak + 1 if dry_supported else 0
        stone_delta = grouped_mapping_value(state["events"]["mine_block"], "stone") - self.baseline_stone
        wrong_delta = self.wrong_blocks(state) - self.baseline_wrong
        if self.task == "EXIT_WATER" and self.dry_streak >= STABLE_STEPS:
            self.success_details = {"stable_dry_steps": self.dry_streak}
            return "SUCCESS", self.success_details
        if self.task == "CLIMB_SHORE" and self.dry_streak >= STABLE_STEPS and float(state["position"]["y"]) >= float(self.baseline["position"]["y"]) + 0.75:
            self.success_details = {"stable_dry_steps": self.dry_streak, "height_gain": float(state["position"]["y"]) - float(self.baseline["position"]["y"])}
            return "SUCCESS", self.success_details
        if self.task == "REACQUIRE_STONE" and stone_delta >= 1:
            self.success_details = {"stone_mined_delta": stone_delta}
            return "SUCCESS", self.success_details
        if self.task == "ESCAPE_HOLE" and self.dry_streak >= STABLE_STEPS and float(state["position"]["y"]) >= self.reference_grade_y - 0.25:
            self.success_details = {"stable_steps": self.dry_streak, "height_gain": float(state["position"]["y"]) - float(self.baseline["position"]["y"])}
            return "SUCCESS", self.success_details
        if self.task == "AVOID_DIGGING_TRAP":
            if self.ever_trapped or wrong_delta >= 1:
                self.failure_reason = "DUG_INTO_TRAP"
                return "FAILURE", {"reason": self.failure_reason, "wrong_blocks_mined": wrong_delta, "minimum_y": self.minimum_y}
            if stone_delta >= 1:
                self.success_details = {"stone_mined_delta": stone_delta, "minimum_y": self.minimum_y}
                return "SUCCESS", self.success_details
        if self.task == "RECOVER_CAMERA":
            yaw_error, pitch_error = self.camera_errors(state)
            self.best_yaw_error = min(self.best_yaw_error, yaw_error)
            self.best_pitch_error = min(self.best_pitch_error, pitch_error)
            aligned = yaw_error <= 18.0 and pitch_error <= 18.0
            self.camera_streak = self.camera_streak + 1 if aligned else 0
            if self.camera_streak >= CAMERA_STABLE_STEPS:
                self.success_details = {"aligned_steps": self.camera_streak, "yaw_error": yaw_error, "pitch_error": pitch_error}
                return "SUCCESS", self.success_details
        if self.task == "AVOID_WATER":
            if state["water"] == 1:
                self.failure_reason = "ENTERED_WATER"
                return "FAILURE", {"reason": self.failure_reason}
            if stone_delta >= 1:
                self.success_details = {"stone_mined_delta": stone_delta}
                return "SUCCESS", self.success_details
        return None, {}

    def timeout_reason(self, final: dict) -> str:
        if self.failure_reason:
            return self.failure_reason
        if self.task == "EXIT_WATER":
            return "STILL_IN_WATER" if final["water"] == 1 else "UNSTABLE_EXIT"
        if self.task == "CLIMB_SHORE":
            return "FAILED_SHORE_CLIMB" if final["position"]["y"] < self.baseline["position"]["y"] + 0.75 else "UNSTABLE_SHORE_EXIT"
        if self.task == "REACQUIRE_STONE":
            return "STONE_NOT_REACQUIRED"
        if self.task == "ESCAPE_HOLE":
            return "REMAINED_IN_HOLE" if final["position"]["y"] < self.reference_grade_y - 0.25 else "UNSTABLE_HOLE_EXIT"
        if self.task == "AVOID_DIGGING_TRAP":
            return "NO_PRODUCTIVE_STONE_MINING"
        if self.task == "RECOVER_CAMERA":
            if self.best_yaw_error <= 30.0 and self.best_pitch_error <= 30.0:
                return "CAMERA_ALIGNMENT_NOT_STABLE"
            return "CAMERA_NOT_RECOVERED"
        if self.task == "AVOID_WATER":
            return "STONE_NOT_REACHED"
        return "TIMEOUT"

    def diagnostics(self, final: dict) -> dict:
        stone_delta = grouped_mapping_value(final["events"]["mine_block"], "stone") - self.baseline_stone
        wrong_delta = self.wrong_blocks(final) - self.baseline_wrong
        return {
            "ever_water": self.ever_water,
            "ever_trapped": self.ever_trapped,
            "minimum_y": self.minimum_y,
            "final_y": float(final["position"]["y"]),
            "dry_streak": self.dry_streak,
            "camera_streak": self.camera_streak,
            "best_yaw_error": self.best_yaw_error if self.task == "RECOVER_CAMERA" else None,
            "best_pitch_error": self.best_pitch_error if self.task == "RECOVER_CAMERA" else None,
            "stone_mined_delta": stone_delta,
            "wrong_blocks_mined": wrong_delta,
            "stable_support_final": has_stable_support(final),
        }


def validate_initial_state(config: dict, baseline: dict) -> None:
    task = config["task"]
    if task in {"EXIT_WATER", "CLIMB_SHORE"} and baseline["water"] != 1:
        raise RuntimeError(f"{task} initial state is not in water: {baseline['water']}")
    if task not in {"EXIT_WATER", "CLIMB_SHORE", "ESCAPE_HOLE"} and baseline["water"] != 0:
        raise RuntimeError(f"{task} initial state unexpectedly contains water: {baseline['water']}")
    if task == "ESCAPE_HOLE" and not has_stable_support(baseline):
        raise RuntimeError("ESCAPE_HOLE initial state has no stable floor")
    if "life_stats" not in baseline and "health" not in baseline:
        raise RuntimeError(f"{task} life-state instrumentation missing")


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


def write_trajectory(output_directory: Path, config: dict, recorder: TrajectoryRecorder, baseline: dict, final: dict, result: dict, router_transitions: list[dict], elapsed_seconds: float) -> tuple[str, str]:
    trajectory_directory = output_directory / "trajectories"
    trajectory_directory.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{config['episode_index']:05d}_{config['seed']}"
    arrays = recorder.arrays()
    npz_path = trajectory_directory / f"{stem}.npz"
    npz_temporary = trajectory_directory / f"{stem}.npz.tmp"
    with npz_temporary.open("wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    npz_temporary.replace(npz_path)
    metadata = {
        "schema_version": 1,
        "transition_contract": "row t stores observation/state before action t",
        "config": config,
        "router_transitions": router_transitions,
        "environment_button_columns": list(ENV_BUTTONS),
        "position_columns": ["x", "y", "z", "yaw", "pitch"],
        "phases": recorder.phases,
        "inventories": recorder.inventories,
        "cumulative_events": recorder.events,
        "local_voxels": recorder.voxels,
        "blocked_controls": dict(recorder.blocked_controls),
        "baseline_state": baseline,
        "final_state": final,
        "result": result,
        "steps": len(recorder.frames),
        "elapsed_seconds": elapsed_seconds,
    }
    metadata_path = trajectory_directory / f"{stem}.json.gz"
    metadata_temporary = trajectory_directory / f"{stem}.json.gz.tmp"
    with gzip.open(metadata_temporary, "wt", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, separators=(",", ":"))
    metadata_temporary.replace(metadata_path)
    return str(npz_path.relative_to(output_directory)), str(metadata_path.relative_to(output_directory))


def run_episode(model: SteveOnePolicy, simulator: MinecraftSim, command_callback: CommandsCallback, config: dict, output_directory: Path) -> dict:
    random.seed(config["seed"])
    np.random.seed(config["seed"] % (2**32))
    torch.manual_seed(config["seed"])
    command_callback.commands = scenario_commands(config)
    simulator.seed = config["seed"]
    simulator.env.seed(config["seed"])
    observation, info = simulator.reset()
    terminated = False
    truncated = False
    setup_steps = 0
    for setup_step in range(1, MAX_SETUP_SETTLE_STEPS + 1):
        observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
        setup_steps = setup_step
        state = state_snapshot(info)
        ready = "life_stats" in info and bool(voxel_records(info.get("voxels")))
        if setup_step >= SETUP_SETTLE_STEPS and ready and has_stable_support(state):
            break
    baseline = state_snapshot(info)
    validate_initial_state(config, baseline)
    recorder = TrajectoryRecorder(config)
    controller = None
    recurrent_state = None
    condition = model.prepare_condition({"cond_scale": CONDITION_SCALE, "text": config["policy_prompt"]}, deterministic=DETERMINISTIC_TEXT_PRIOR)
    verifier = MicroVerifier(config, baseline)
    recent_positions = deque(maxlen=40)
    router = LegacyStoneTaskRouter(verifier.reference_grade_y)
    active_category = None
    router_transitions = []
    outcome = None
    outcome_details = {}
    environment_error = None
    started = time.monotonic()

    def recorded_step(agent_action: dict, phase: str):
        nonlocal observation, info, terminated, truncated, environment_error
        environment_action = simulator.agent_action_to_env_action(deepcopy(agent_action))
        recorder.append(observation["image"], info, agent_action, environment_action, phase)
        observation, reward, terminated, truncated, info = simulator.step(agent_action)
        environment_error = info.get("error")
        return observation, reward, terminated, truncated, info

    controller = ClosedLoopOptions(simulator, recorded_step, pickup_implementation="old")
    last_stone_mined = grouped_mapping_value(baseline["events"]["mine_block"], "stone")
    while len(recorder.frames) < int(config["timeout_steps"]) and not terminated and not truncated and environment_error is None and outcome is None:
        pre_state = state_snapshot(info)
        recent_positions.append((pre_state["position"]["x"], pre_state["position"]["y"], pre_state["position"]["z"]))
        low_displacement = len(recent_positions) == recent_positions.maxlen and math.dist(recent_positions[0], recent_positions[-1]) < 0.6
        stone_mined = grouped_mapping_value(pre_state["events"]["mine_block"], "stone")
        routing = router.update(pre_state, len(recorder.frames), low_displacement, stone_mined)
        if routing.category != active_category:
            active_category = routing.category
            recurrent_state = None
            router_transitions.append({"step": len(recorder.frames), "category": active_category})
        image = torch.from_numpy(observation["image"]).unsqueeze(0).unsqueeze(0).to("cuda")
        model_input = {"image": image, "condition": condition}
        with torch.inference_mode():
            batched_action, recurrent_state = model.get_action(model_input, recurrent_state, deterministic=DETERMINISTIC_ACTIONS, input_shape="BT*")
        raw_action = {name: value[0][0] for name, value in batched_action.items()}
        action, environment_action, blocked_controls = controller.gate_world_control(raw_action)
        recorder.blocked_controls.update(blocked_controls)
        observation, reward, terminated, truncated, info = recorded_step(action, f"STEVE_1/{config['task']}/{active_category}")
        current = state_snapshot(info)
        outcome, outcome_details = verifier.update(current, len(recorder.frames))
        current_stone_mined = grouped_mapping_value(current["events"]["mine_block"], "stone")
        last_stone_mined = max(last_stone_mined, current_stone_mined)
    if environment_error is not None:
        raise RuntimeError(f"Environment error for {config['episode_id']}: {environment_error}")
    final = state_snapshot(info)
    if outcome is None:
        outcome = "FAILURE"
        outcome_details = {"reason": verifier.timeout_reason(final)}
    success = outcome == "SUCCESS"
    failure_reason = None if success else str(outcome_details.get("reason", verifier.timeout_reason(final)))
    completion_step = len(recorder.frames) if success else None
    elapsed_seconds = time.monotonic() - started
    diagnostics = verifier.diagnostics(final)
    result = {
        "result_id": config["episode_id"],
        "episode_index": config["episode_index"],
        "seed": config["seed"],
        "task": config["task"],
        "biome": config["biome"],
        "success": success,
        "failure_reason": failure_reason,
        "steps": len(recorder.frames),
        "completion_step": completion_step,
        "episode_seconds": elapsed_seconds,
        "setup_steps": setup_steps,
        "terminated": terminated,
        "truncated": truncated,
        "policy_prompt": config["policy_prompt"],
        "router_implementation": "old",
        "pickup_implementation": "old",
        "router_transitions": router_transitions,
        "router_transition_count": len(router_transitions),
        "blocked_controls": dict(recorder.blocked_controls),
        "outcome_details": outcome_details,
        "diagnostics": diagnostics,
        "baseline_state": baseline,
        "final_state": final,
    }
    npz_file, metadata_file = write_trajectory(output_directory, config, recorder, baseline, final, result, router_transitions, elapsed_seconds)
    result["trajectory_npz"] = npz_file
    result["trajectory_metadata"] = metadata_file
    return result


def run_worker(worker_index: int, assignments: list[dict], output_directory: str, result_queue: multiprocessing.Queue) -> None:
    current_config = None
    simulator = None
    try:
        torch.set_float32_matmul_precision("high")
        model = SteveOnePolicy.from_pretrained(CHECKPOINT_DIRECTORY).to("cuda").eval()
        command_callback = CommandsCallback([])
        simulator_biome = None
        simulator_episodes = 0
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
            result = run_episode(model, simulator, command_callback, config, Path(output_directory))
            simulator_episodes += 1
            result["worker_index"] = worker_index
            result_queue.put({"kind": "episode", "result": result})
    except Exception:
        result_queue.put({"kind": "error", "worker_index": worker_index, "episode_id": current_config["episode_id"] if current_config else None, "traceback": traceback.format_exc()})
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


def metrics_for(results: list[dict]) -> dict:
    ordered = sorted(results, key=lambda row: row["episode_index"])
    successful = [row for row in ordered if row["success"]]
    completion_steps = [int(row["completion_step"]) for row in successful]
    failures = Counter(row["failure_reason"] for row in ordered if not row["success"])
    successes = len(successful)
    return {
        "episodes": len(ordered),
        "successes": successes,
        "success_rate": successes / len(ordered) if ordered else 0.0,
        "success_wilson_95": wilson_interval(successes, len(ordered)),
        "mean_completion_steps": statistics.mean(completion_steps) if completion_steps else None,
        "median_completion_steps": statistics.median(completion_steps) if completion_steps else None,
        "mean_episode_steps": statistics.mean([row["steps"] for row in ordered]) if ordered else None,
        "failure_reasons": dict(sorted(failures.items())),
        "water_entry_rate": statistics.mean([float(row["diagnostics"]["ever_water"]) for row in ordered]) if ordered else None,
        "trap_entry_rate": statistics.mean([float(row["diagnostics"]["ever_trapped"]) for row in ordered]) if ordered else None,
        "mean_router_transitions": statistics.mean([row["router_transition_count"] for row in ordered]) if ordered else None,
        "mean_wall_seconds": statistics.mean([row["episode_seconds"] for row in ordered]) if ordered else None,
    }


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_summary(results: list[dict], started_at: str, elapsed_seconds: float) -> dict:
    return {
        "schema_version": 1,
        "task": TASK_NAME,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "episodes_target": EPISODES,
        "episodes_complete": len(results),
        "checkpoint": str(CHECKPOINT_DIRECTORY),
        "checkpoint_role": CHECKPOINT_ROLE,
        "policy_modified": False,
        "training": False,
        "router_implementation": "old",
        "pickup_implementation": "old",
        "live_gemini_planning": False,
        "policy_prompt": TASK_PROMPTS[TASK_NAME],
        "timeout_steps": TASK_TIMEOUTS[TASK_NAME],
        "base_seed": BASE_SEED,
        "num_workers": NUM_WORKERS,
        "success_predicate": {
            "EXIT_WATER": "10 consecutive dry supported steps",
            "CLIMB_SHORE": "10 consecutive dry supported steps at least 0.75 blocks above start",
            "REACQUIRE_STONE": "exact stone mine event delta at least 1",
            "ESCAPE_HOLE": "10 supported steps at surrounding grade",
            "AVOID_DIGGING_TRAP": "exact stone mine event before wrong-block mining or vertical trap entry",
            "RECOVER_CAMERA": "target angular error within 18 degrees yaw and pitch for 8 consecutive steps",
            "AVOID_WATER": "exact stone mine event before any local water entry",
        }[TASK_NAME],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "overall": metrics_for(results),
        "by_biome": {biome: metrics_for([row for row in results if row["biome"] == biome]) for biome in BIOMES},
        "episode_results_file": "episodes.jsonl",
    }


def load_or_create_manifest() -> list[dict]:
    manifest_path = OUTPUT_DIRECTORY / "manifest.jsonl"
    expected = make_manifest()
    if manifest_path.exists():
        existing = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        if existing != expected:
            raise RuntimeError(f"Frozen manifest differs from current definition: {manifest_path}")
        return existing
    manifest_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in expected), encoding="utf-8")
    return expected


def worker_assignments(configs: list[dict], worker_count: int) -> list[list[dict]]:
    ordered = sorted(configs, key=lambda row: (row["biome"], row["episode_index"]))
    assignments = [[] for _ in range(worker_count)]
    for index, config in enumerate(ordered):
        assignments[index % worker_count].append(config)
    return assignments


def main() -> None:
    if TASK_NAME not in TASKS:
        raise ValueError(f"Unknown task: {TASK_NAME}")
    if not CHECKPOINT_DIRECTORY.is_dir():
        raise FileNotFoundError(CHECKPOINT_DIRECTORY)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    manifest = load_or_create_manifest()
    episodes_path = OUTPUT_DIRECTORY / "episodes.jsonl"
    summary_path = OUTPUT_DIRECTORY / "summary.json"
    existing_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    started_at = existing_summary.get("started_at", datetime.now(timezone.utc).isoformat())
    started_time = time.monotonic()
    results_by_id = {}
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            results_by_id[row["result_id"]] = row
    print(json.dumps({"status": "starting", "task": TASK_NAME, "resumed": len(results_by_id), "target": EPISODES, "output": str(OUTPUT_DIRECTORY)}), flush=True)
    context = multiprocessing.get_context("spawn")
    for wave_index in range(MAX_WORKER_WAVES):
        pending = [config for config in manifest if config["episode_id"] not in results_by_id]
        if not pending:
            break
        worker_count = min(NUM_WORKERS, len(pending))
        assignments = worker_assignments(pending, worker_count)
        result_queue = context.Queue()
        processes = [context.Process(target=run_worker, args=(worker_index, assignments[worker_index], str(OUTPUT_DIRECTORY), result_queue)) for worker_index in range(worker_count)]
        for process in processes:
            process.start()
        wave_failed = None
        wave_completed = 0
        while wave_completed < len(pending):
            try:
                message = result_queue.get(timeout=180)
            except queue.Empty:
                dead = [process for process in processes if not process.is_alive() and process.exitcode != 0]
                if dead:
                    wave_failed = {"kind": "error", "traceback": f"Workers exited without results: {[process.exitcode for process in dead]}"}
                    break
                continue
            if message["kind"] == "error":
                wave_failed = message
                break
            result = message["result"]
            if result["result_id"] in results_by_id:
                continue
            results_by_id[result["result_id"]] = result
            wave_completed += 1
            with episodes_path.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(result, separators=(",", ":")) + "\n")
            summary = build_summary(list(results_by_id.values()), started_at, time.monotonic() - started_time)
            write_json_atomic(summary_path, summary)
            print(json.dumps({"status": "episode_complete", "task": TASK_NAME, "complete": len(results_by_id), "target": EPISODES, "episode": result["episode_index"], "success": result["success"], "failure_reason": result["failure_reason"], "steps": result["steps"], "seconds": round(result["episode_seconds"], 2), "wave": wave_index + 1}), flush=True)
        if wave_failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=30)
            with (OUTPUT_DIRECTORY / "worker_errors.jsonl").open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps({"wave": wave_index + 1, **wave_failed}, separators=(",", ":")) + "\n")
            continue
        for process in processes:
            process.join(timeout=30)
    missing = [config["episode_id"] for config in manifest if config["episode_id"] not in results_by_id]
    if missing:
        raise RuntimeError(f"Benchmark exhausted worker restart waves with {len(missing)} missing episodes: {missing[:10]}")
    summary = build_summary(list(results_by_id.values()), started_at, time.monotonic() - started_time)
    write_json_atomic(summary_path, summary)
    print(json.dumps({"status": "complete", "task": TASK_NAME, "episodes": len(results_by_id), "success_rate": summary["overall"]["success_rate"], "output": str(OUTPUT_DIRECTORY)}), flush=True)


if __name__ == "__main__":
    main()
