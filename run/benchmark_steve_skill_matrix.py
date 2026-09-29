import json
import math
import multiprocessing
import queue
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from minestudio.models import SteveOnePolicy
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = PROJECT_DIRECTORY / "checkpoints" / "steve_one_official"
OUTPUT_ROOT = PROJECT_DIRECTORY / "output" / "steve_skill_matrix"
EXPERIMENT_NAME = "controlled_15_skill_100"
CONDITION_SCALE = 6.0
DETERMINISTIC_TEXT_PRIOR = False
DETERMINISTIC_ACTIONS = False
EPISODES_PER_SKILL = 100
NUM_WORKERS = 2
BASE_WORLD_SEED = 20260915
PREFERRED_SPAWN_BIOME = "forest"
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
SETUP_SETTLE_STEPS = 2
MAX_SETUP_SETTLE_STEPS = 20
VIDEO_FPS = 20
SAVE_FIRST_EPISODE_VIDEO = True
RESUME_DIRECTORY_NAME = "controlled_15_skill_100"
MAX_WORKER_WAVES = 8
MAX_EPISODES_PER_SIMULATOR = 5
CONTROLLED_SCENE_RADIUS = 5
ITEM_GROUP_SUFFIXES = {
    "log": ("_log", "_stem", "hyphae"),
    "planks": ("_planks",),
    "leaves": ("_leaves",),
    "sapling": ("_sapling",),
    "wool": ("_wool",),
    "coal_ore": ("coal_ore", "deepslate_coal_ore"),
}
SKILLS = [
    {
        "skill_id": "explore",
        "prompt": "go explore",
        "max_steps": 400,
        "scenario": "natural",
        "primary": {"source": "player_pos", "target": "any", "threshold": 8},
    },
    {
        "skill_id": "break_log",
        "prompt": "mine a log",
        "max_steps": 600,
        "scenario": "oak_log",
        "primary": {"source": "mine_block", "target": "log", "threshold": 1},
    },
    {
        "skill_id": "collect_log",
        "prompt": "chop down the tree, gather wood, pick up wood, chop it down, break tree",
        "max_steps": 800,
        "scenario": "oak_log",
        "primary": {"source": "inventory", "target": "log", "threshold": 1},
        "secondary": {"source": "mine_block", "target": "log", "threshold": 1},
    },
    {
        "skill_id": "collect_dirt",
        "prompt": "collect dirt",
        "max_steps": 600,
        "scenario": "dirt",
        "primary": {"source": "inventory", "target": "dirt", "threshold": 1},
        "secondary": {"source": "mine_block", "target": "dirt", "threshold": 1},
    },
    {
        "skill_id": "collect_sand",
        "prompt": "collect sand",
        "max_steps": 600,
        "scenario": "sand",
        "primary": {"source": "inventory", "target": "sand", "threshold": 1},
        "secondary": {"source": "mine_block", "target": "sand", "threshold": 1},
    },
    {
        "skill_id": "craft_planks",
        "prompt": "craft wooden planks",
        "max_steps": 600,
        "scenario": "oak_logs_inventory",
        "primary": {"source": "craft_item", "target": "planks", "threshold": 4},
    },
    {
        "skill_id": "craft_sticks",
        "prompt": "craft sticks",
        "max_steps": 600,
        "scenario": "planks_inventory",
        "primary": {"source": "craft_item", "target": "stick", "threshold": 4},
    },
    {
        "skill_id": "craft_crafting_table",
        "prompt": "craft a crafting table",
        "max_steps": 600,
        "scenario": "planks_inventory",
        "primary": {"source": "craft_item", "target": "crafting_table", "threshold": 1},
    },
    {
        "skill_id": "place_crafting_table",
        "prompt": "place the crafting table",
        "max_steps": 500,
        "scenario": "crafting_table_inventory",
        "primary": {"source": "place_block", "target": "crafting_table", "threshold": 1},
    },
    {
        "skill_id": "craft_wooden_pickaxe",
        "prompt": "craft a wooden pickaxe",
        "max_steps": 800,
        "scenario": "wooden_pickaxe_ingredients",
        "primary": {"source": "craft_item", "target": "wooden_pickaxe", "threshold": 1},
    },
    {
        "skill_id": "mine_cobblestone",
        "prompt": "mine cobblestone",
        "max_steps": 700,
        "scenario": "stone_with_wooden_pickaxe",
        "primary": {"source": "inventory", "target": "cobblestone", "threshold": 1},
        "secondary": {"source": "mine_block", "target": "stone", "threshold": 1},
    },
    {
        "skill_id": "craft_stone_pickaxe",
        "prompt": "craft a stone pickaxe",
        "max_steps": 800,
        "scenario": "stone_pickaxe_ingredients",
        "primary": {"source": "craft_item", "target": "stone_pickaxe", "threshold": 1},
    },
    {
        "skill_id": "mine_coal",
        "prompt": "mine coal",
        "max_steps": 700,
        "scenario": "coal_with_stone_pickaxe",
        "primary": {"source": "inventory", "target": "coal", "threshold": 1},
        "secondary": {"source": "mine_block", "target": "coal_ore", "threshold": 1},
    },
    {
        "skill_id": "kill_cow",
        "prompt": "hunt and kill the cow",
        "max_steps": 700,
        "scenario": "cow_with_wooden_sword",
        "primary": {"source": "kill_entity", "target": "cow", "threshold": 1},
    },
    {
        "skill_id": "build_pillar",
        "prompt": "build a tall cobblestone pillar",
        "max_steps": 800,
        "scenario": "cobblestone_inventory",
        "primary": {"source": "place_block", "target": "cobblestone", "threshold": 3},
    },
]
SCENARIO_EXPECTED_INVENTORY = {
    "oak_logs_inventory": {"oak_log": 3},
    "planks_inventory": {"oak_planks": 8},
    "crafting_table_inventory": {"crafting_table": 1},
    "wooden_pickaxe_ingredients": {"oak_planks": 8, "stick": 4},
    "stone_with_wooden_pickaxe": {"wooden_pickaxe": 1},
    "stone_pickaxe_ingredients": {"cobblestone": 6, "stick": 4},
    "coal_with_stone_pickaxe": {"stone_pickaxe": 1},
    "cow_with_wooden_sword": {"wooden_sword": 1},
    "cobblestone_inventory": {"cobblestone": 16},
}


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


def verifier_progress(verifier: dict, baseline: dict, current: dict) -> float:
    if verifier["source"] == "player_pos":
        start = baseline["player_pos"]
        end = current["player_pos"]
        return math.hypot(end["x"] - start["x"], end["z"] - start["z"])
    return grouped_value(current[verifier["source"]], verifier["target"]) - grouped_value(
        baseline[verifier["source"]], verifier["target"]
    )


def expected_inventory_ready(scenario: str, state: dict) -> bool:
    expected = SCENARIO_EXPECTED_INVENTORY.get(scenario, {})
    return all(state["inventory"].get(item, 0.0) >= quantity for item, quantity in expected.items())


def target_pose(seed: int) -> tuple[int, int, float]:
    generator = random.Random(seed)
    poses = [(0, 3, 0.0), (-3, 0, 90.0), (0, -3, 180.0), (3, 0, -90.0)]
    dx, dz, yaw = generator.choice(poses)
    return dx, dz, yaw + generator.uniform(-18.0, 18.0)


def scenario_commands(scenario: str, seed: int) -> list[str]:
    dx, dz, yaw = target_pose(seed)
    commands = [
        "/clear @p",
        "/effect clear @p",
        "/time set day",
        "/weather clear",
        f"/tp @p ~ ~ ~ {yaw:.2f} 0",
    ]
    if scenario == "natural":
        return commands
    radius = CONTROLLED_SCENE_RADIUS
    commands.extend([
        f"/fill ~-{radius} ~-1 ~-{radius} ~{radius} ~-1 ~{radius} minecraft:stone",
        f"/fill ~-{radius} ~ ~-{radius} ~{radius} ~4 ~{radius} minecraft:air",
    ])
    if scenario == "oak_log":
        commands.extend([
            f"/setblock ~{dx} ~ ~{dz} minecraft:oak_log",
            f"/setblock ~{dx} ~1 ~{dz} minecraft:oak_log",
            f"/setblock ~{dx} ~2 ~{dz} minecraft:oak_log",
        ])
    elif scenario in {"dirt", "sand"}:
        commands.extend([
            f"/setblock ~{dx} ~ ~{dz} minecraft:{scenario}",
            f"/setblock ~{dx} ~1 ~{dz} minecraft:{scenario}",
        ])
    elif scenario == "oak_logs_inventory":
        commands.append("/give @p minecraft:oak_log 3")
    elif scenario == "planks_inventory":
        commands.append("/give @p minecraft:oak_planks 8")
    elif scenario == "crafting_table_inventory":
        commands.append("/give @p minecraft:crafting_table 1")
    elif scenario == "wooden_pickaxe_ingredients":
        commands.extend([
            "/give @p minecraft:oak_planks 8",
            "/give @p minecraft:stick 4",
            f"/setblock ~{dx} ~ ~{dz} minecraft:crafting_table",
        ])
    elif scenario == "stone_with_wooden_pickaxe":
        commands.extend([
            "/give @p minecraft:wooden_pickaxe 1",
            f"/setblock ~{dx} ~ ~{dz} minecraft:stone",
            f"/setblock ~{dx} ~1 ~{dz} minecraft:stone",
        ])
    elif scenario == "stone_pickaxe_ingredients":
        commands.extend([
            "/give @p minecraft:cobblestone 6",
            "/give @p minecraft:stick 4",
            f"/setblock ~{dx} ~ ~{dz} minecraft:crafting_table",
        ])
    elif scenario == "coal_with_stone_pickaxe":
        commands.extend([
            "/give @p minecraft:stone_pickaxe 1",
            f"/setblock ~{dx} ~ ~{dz} minecraft:coal_ore",
            f"/setblock ~{dx} ~1 ~{dz} minecraft:coal_ore",
        ])
    elif scenario == "cow_with_wooden_sword":
        commands.extend([
            "/kill @e[type=minecraft:cow,distance=..16]",
            "/give @p minecraft:wooden_sword 1",
            f"/summon minecraft:cow ~{dx} ~1 ~{dz}",
        ])
    elif scenario == "cobblestone_inventory":
        commands.append("/give @p minecraft:cobblestone 16")
    else:
        raise ValueError(scenario)
    return commands


def unbatch_action(action: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value[0][0] for name, value in action.items()}


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


def close_video(container: av.container.OutputContainer, stream: av.video.stream.VideoStream) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def overlay_frame(frame: np.ndarray, skill: dict, episode_index: int, step: int, primary: float, secondary: float) -> np.ndarray:
    image = np.ascontiguousarray(frame.copy())
    lines = [
        f"STEVE-1: {skill['prompt']}",
        f"{skill['skill_id']} episode {episode_index + 1}/{EPISODES_PER_SKILL} step {step}/{skill['max_steps']}",
        f"primary {primary:.1f}/{skill['primary']['threshold']:g} secondary {secondary:.1f}",
    ]
    cv2.rectangle(image, (0, 0), (image.shape[1], 82), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        cv2.putText(image, line, (12, 23 + 25 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def episode_seed(skill_index: int, episode_index: int) -> int:
    return BASE_WORLD_SEED + skill_index * 1000003 + episode_index * 7919


def create_simulator(seed: int, command_callback: CommandsCallback) -> MinecraftSim:
    return MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=seed,
        preferred_spawn_biome=PREFERRED_SPAWN_BIOME,
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[command_callback],
    )


def run_worker(worker_index: int, assignments: list[tuple[int, int]], experiment_directory: str, result_queue: multiprocessing.Queue) -> None:
    current_assignment = None
    try:
        torch.set_float32_matmul_precision("high")
        model = SteveOnePolicy.from_pretrained(CHECKPOINT_DIRECTORY).to("cuda").eval()
        first_skill_index, first_episode_index = assignments[0]
        command_callback = CommandsCallback([])
        simulator = create_simulator(episode_seed(first_skill_index, first_episode_index), command_callback)
        output_directory = Path(experiment_directory)
        try:
            for worker_assignment_index, (skill_index, episode_index) in enumerate(assignments):
                current_assignment = (skill_index, episode_index)
                skill = SKILLS[skill_index]
                seed = episode_seed(skill_index, episode_index)
                if worker_assignment_index > 0 and worker_assignment_index % MAX_EPISODES_PER_SIMULATOR == 0:
                    simulator.close()
                    command_callback = CommandsCallback([])
                    simulator = create_simulator(seed, command_callback)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                condition = model.prepare_condition(
                    {"cond_scale": CONDITION_SCALE, "text": skill["prompt"]},
                    deterministic=DETERMINISTIC_TEXT_PRIOR,
                )
                command_callback.commands = scenario_commands(skill["scenario"], seed)
                simulator.seed = seed
                simulator.env.seed(seed)
                observation, info = simulator.reset()
                setup_state = state_snapshot(info)
                setup_steps = 0
                for setup_step in range(1, MAX_SETUP_SETTLE_STEPS + 1):
                    observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
                    setup_steps = setup_step
                    setup_state = state_snapshot(info)
                    if setup_step >= SETUP_SETTLE_STEPS and expected_inventory_ready(skill["scenario"], setup_state):
                        break
                if not expected_inventory_ready(skill["scenario"], setup_state):
                    raise RuntimeError(f"Scenario inventory did not settle for {skill['skill_id']}: {setup_state['inventory']}")
                baseline = setup_state
                current = baseline
                recurrent_state = None
                primary_progress = 0.0
                secondary_progress = 0.0
                terminated = False
                truncated = False
                steps_taken = 0
                record_video = SAVE_FIRST_EPISODE_VIDEO and episode_index == 0
                video_path = output_directory / f"{skill['skill_id']}_{episode_index:03d}_{seed}.mp4"
                container = None
                stream = None
                if record_video:
                    container, stream = create_video(video_path)
                    write_video_frame(container, stream, overlay_frame(np.asarray(info["pov"]), skill, episode_index, 0, 0.0, 0.0))
                button_counts = {}
                camera_counts = {}
                episode_started = time.monotonic()
                for step in range(1, skill["max_steps"] + 1):
                    image = torch.from_numpy(observation["image"]).unsqueeze(0).unsqueeze(0).to("cuda")
                    model_input = {"image": image, "condition": condition}
                    batched_action, recurrent_state = model.get_action(
                        model_input,
                        recurrent_state,
                        deterministic=DETERMINISTIC_ACTIONS,
                        input_shape="BT*",
                    )
                    action = unbatch_action(batched_action)
                    button_code = str(int(action["buttons"].detach().cpu().numpy().item()))
                    camera_code = str(int(action["camera"].detach().cpu().numpy().item()))
                    button_counts[button_code] = button_counts.get(button_code, 0) + 1
                    camera_counts[camera_code] = camera_counts.get(camera_code, 0) + 1
                    observation, reward, terminated, truncated, info = simulator.step(action)
                    steps_taken = step
                    current = state_snapshot(info)
                    primary_progress = verifier_progress(skill["primary"], baseline, current)
                    if "secondary" in skill:
                        secondary_progress = verifier_progress(skill["secondary"], baseline, current)
                    if container is not None:
                        write_video_frame(
                            container,
                            stream,
                            overlay_frame(np.asarray(info["pov"]), skill, episode_index, step, primary_progress, secondary_progress),
                        )
                    if primary_progress >= skill["primary"]["threshold"] or terminated or truncated:
                        break
                if container is not None:
                    close_video(container, stream)
                final_frame_path = output_directory / f"{skill['skill_id']}_{episode_index:03d}_{seed}_final.jpg"
                if record_video:
                    final_frame = cv2.cvtColor(np.asarray(info["pov"]), cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(final_frame_path), final_frame):
                        raise RuntimeError(f"Could not write {final_frame_path}")
                result_queue.put({
                    "kind": "episode",
                    "result": {
                        "result_id": f"{skill['skill_id']}:{episode_index}",
                        "skill_index": skill_index,
                        "skill_id": skill["skill_id"],
                        "episode_index": episode_index,
                        "worker_index": worker_index,
                        "seed": seed,
                        "prompt": skill["prompt"],
                        "scenario": skill["scenario"],
                        "scenario_commands": command_callback.commands,
                        "setup_steps": setup_steps,
                        "primary_verifier": skill["primary"],
                        "primary_progress": primary_progress,
                        "success": primary_progress >= skill["primary"]["threshold"],
                        "secondary_verifier": skill.get("secondary"),
                        "secondary_progress": secondary_progress,
                        "secondary_success": "secondary" in skill and secondary_progress >= skill["secondary"]["threshold"],
                        "steps": steps_taken,
                        "episode_seconds": time.monotonic() - episode_started,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "agent_button_counts": button_counts,
                        "agent_camera_counts": camera_counts,
                        "baseline_state": baseline,
                        "final_state": current,
                        "video": video_path.name if record_video else None,
                        "final_frame": final_frame_path.name if record_video else None,
                    },
                })
        finally:
            simulator.close()
    except Exception:
        result_queue.put({"kind": "error", "worker_index": worker_index, "assignment": current_assignment, "traceback": traceback.format_exc()})


def wilson_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def build_summary(results: list[dict], started_at: str, elapsed_seconds: float) -> dict:
    ordered = sorted(results, key=lambda result: (result["skill_index"], result["episode_index"]))
    skill_summaries = {}
    for skill in SKILLS:
        matching = [result for result in ordered if result["skill_id"] == skill["skill_id"]]
        successes = sum(result["success"] for result in matching)
        secondary_successes = sum(result["secondary_success"] for result in matching)
        skill_summaries[skill["skill_id"]] = {
            "prompt": skill["prompt"],
            "scenario": skill["scenario"],
            "primary_verifier": skill["primary"],
            "secondary_verifier": skill.get("secondary"),
            "episodes_complete": len(matching),
            "episodes_target": EPISODES_PER_SKILL,
            "successes": successes,
            "success_rate": successes / len(matching) if matching else 0.0,
            "wilson_95": wilson_interval(successes, len(matching)),
            "secondary_successes": secondary_successes,
            "secondary_success_rate": secondary_successes / len(matching) if matching else 0.0,
            "mean_steps": sum(result["steps"] for result in matching) / len(matching) if matching else 0.0,
        }
    return {
        "experiment_name": EXPERIMENT_NAME,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(CHECKPOINT_DIRECTORY),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "condition_scale": CONDITION_SCALE,
        "deterministic_text_prior": DETERMINISTIC_TEXT_PRIOR,
        "deterministic_actions": DETERMINISTIC_ACTIONS,
        "experimental_design": "Random world seed and randomized cardinal target pose with task prerequisites injected before the baseline snapshot.",
        "episodes_per_skill": EPISODES_PER_SKILL,
        "skills_target": len(SKILLS),
        "episodes_target": len(SKILLS) * EPISODES_PER_SKILL,
        "episodes_complete": len(ordered),
        "num_workers": NUM_WORKERS,
        "skills": skill_summaries,
        "episodes": ordered,
    }


def main() -> None:
    if not CHECKPOINT_DIRECTORY.is_dir():
        raise FileNotFoundError(CHECKPOINT_DIRECTORY)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_directory = OUTPUT_ROOT / RESUME_DIRECTORY_NAME if RESUME_DIRECTORY_NAME else OUTPUT_ROOT / f"{timestamp}_{EXPERIMENT_NAME}"
    experiment_directory.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_directory / "summary.json"
    episodes_path = experiment_directory / "episodes.jsonl"
    existing_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    started_at = existing_summary.get("started_at", datetime.now(timezone.utc).isoformat())
    started_time = time.monotonic()
    context = multiprocessing.get_context("spawn")
    results_by_id = {}
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            results_by_id[result["result_id"]] = result
    all_assignments = [(skill_index, episode_index) for skill_index in range(len(SKILLS)) for episode_index in range(EPISODES_PER_SKILL)]
    print(json.dumps({"resumed": len(results_by_id), "total": len(all_assignments), "output": str(experiment_directory)}), flush=True)
    for wave_index in range(MAX_WORKER_WAVES):
        pending = [assignment for assignment in all_assignments if f"{SKILLS[assignment[0]]['skill_id']}:{assignment[1]}" not in results_by_id]
        if not pending:
            break
        worker_count = min(NUM_WORKERS, len(pending))
        worker_assignments = [pending[worker_index::worker_count] for worker_index in range(worker_count)]
        result_queue = context.Queue()
        processes = [
            context.Process(target=run_worker, args=(worker_index, worker_assignments[worker_index], str(experiment_directory), result_queue))
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
                print(json.dumps({"complete": len(results_by_id), "total": len(all_assignments), "wave": wave_index + 1, "status": "waiting"}), flush=True)
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
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "complete": len(results_by_id),
                "total": len(all_assignments),
                "skill": result["skill_id"],
                "episode": result["episode_index"],
                "success": result["success"],
                "primary_progress": result["primary_progress"],
                "secondary_success": result["secondary_success"],
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
            with (experiment_directory / "worker_errors.jsonl").open("a", encoding="utf-8") as error_file:
                error_file.write(json.dumps({"wave": wave_index + 1, **wave_failed}, separators=(",", ":")) + "\n")
            print(json.dumps({"wave": wave_index + 1, "status": "restarting_failed_workers", "complete": len(results_by_id)}), flush=True)
            continue
        for process in processes:
            process.join(timeout=30)
    missing = [assignment for assignment in all_assignments if f"{SKILLS[assignment[0]]['skill_id']}:{assignment[1]}" not in results_by_id]
    if missing:
        raise RuntimeError(f"Benchmark exhausted worker restart waves with missing assignments: {missing}")
    summary = build_summary(list(results_by_id.values()), started_at, time.monotonic() - started_time)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
