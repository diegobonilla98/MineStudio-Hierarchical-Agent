import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from minestudio.models import SteveOnePolicy
from minestudio.simulator import MinecraftSim


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = PROJECT_DIRECTORY / "checkpoints" / "steve_one_official"
OUTPUT_ROOT = PROJECT_DIRECTORY / "output" / "steve_skill_evaluation"
EXPERIMENT_NAME = "paper_wood_prompt_smoke"
DEVICE = "cuda"
CONDITION_SCALE = 6.0
DETERMINISTIC_TEXT_PRIOR = False
DETERMINISTIC_ACTIONS = False
EPISODES_PER_SKILL = 3
MAX_STEPS = 1000
BASE_WORLD_SEED = 20260915
PREFERRED_SPAWN_BIOME = "forest"
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
VIDEO_FPS = 20
SAVE_VIDEO = True
ENABLED_SKILL_IDS = {"mine_log_paper"}
SKILLS = [
    {
        "skill_id": "mine_log",
        "prompt": "mine a log",
        "primary_source": "inventory",
        "primary_target": "log",
        "primary_threshold": 1,
        "secondary_source": "mine_block",
        "secondary_target": "log",
        "secondary_threshold": 1,
    },
    {
        "skill_id": "mine_log_paper",
        "prompt": "chop down the tree, gather wood, pick up wood, chop it down, break tree",
        "primary_source": "inventory",
        "primary_target": "log",
        "primary_threshold": 1,
        "secondary_source": "mine_block",
        "secondary_target": "log",
        "secondary_threshold": 1,
    },
]
ACTIVE_SKILLS = [skill for skill in SKILLS if skill["skill_id"] in ENABLED_SKILL_IDS]
ITEM_GROUP_SUFFIXES = {
    "log": ("_log", "_stem", "hyphae"),
    "planks": ("_planks",),
    "leaves": ("_leaves",),
    "sapling": ("_sapling",),
    "wool": ("_wool",),
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
        "kill_entity": numeric_mapping(info, "kill_entity"),
        "player_pos": {
            "x": float(player_pos.get("x", 0.0)),
            "y": float(player_pos.get("y", 0.0)),
            "z": float(player_pos.get("z", 0.0)),
        },
        "health": float(info.get("health", 0.0)),
        "food_level": float(info.get("food_level", 0.0)),
    }


def progress(source: str, target: str, baseline: dict, current: dict) -> float:
    return grouped_value(current[source], target) - grouped_value(baseline[source], target)


def unbatch_action(action: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value[0][0] for name, value in action.items()}


def serializable_action(action: dict) -> dict:
    result = {}
    for name, value in action.items():
        array = np.asarray(value)
        result[name] = array.item() if array.ndim == 0 else array.tolist()
    return result


def overlay_frame(frame: np.ndarray, skill: dict, episode_index: int, step: int, primary_progress: float, secondary_progress: float) -> np.ndarray:
    image = np.ascontiguousarray(frame.copy())
    lines = [
        f"STEVE-1: {skill['prompt']}",
        f"episode {episode_index + 1}/{EPISODES_PER_SKILL}  step {step}/{MAX_STEPS}",
        f"log acquired: {primary_progress:.0f}  log broken: {secondary_progress:.0f}",
    ]
    cv2.rectangle(image, (0, 0), (image.shape[1], 82), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        cv2.putText(image, line, (12, 23 + 25 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def write_video_frame(container: av.container.OutputContainer, stream: av.video.stream.VideoStream, frame: np.ndarray) -> None:
    video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def close_video(container: av.container.OutputContainer, stream: av.video.stream.VideoStream) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def create_video(path: Path) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream]:
    container = av.open(path, mode="w", format="mp4")
    stream = container.add_stream("h264", rate=VIDEO_FPS)
    stream.width = RENDER_SIZE[0]
    stream.height = RENDER_SIZE[1]
    stream.pix_fmt = "yuv420p"
    return container, stream


def episode_seed(skill_index: int, episode_index: int) -> int:
    return BASE_WORLD_SEED + skill_index * 1000003 + episode_index * 7919


def summarize_results(results: list[dict], started_at: str, elapsed_seconds: float) -> dict:
    summaries = {}
    for skill in ACTIVE_SKILLS:
        matching = [result for result in results if result["skill_id"] == skill["skill_id"]]
        successes = sum(result["success"] for result in matching)
        broken = sum(result["secondary_success"] for result in matching)
        summaries[skill["skill_id"]] = {
            "prompt": skill["prompt"],
            "episodes_complete": len(matching),
            "episodes_target": EPISODES_PER_SKILL,
            "successes": successes,
            "success_rate": successes / len(matching) if matching else 0.0,
            "broke_log_rate": broken / len(matching) if matching else 0.0,
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
        "scenario": "random Minecraft seed with a forest-biome spawn preference and no starting inventory",
        "max_steps": MAX_STEPS,
        "skills": summaries,
        "episodes": results,
    }


def main() -> None:
    if not CHECKPOINT_DIRECTORY.is_dir():
        raise FileNotFoundError(CHECKPOINT_DIRECTORY)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_directory = OUTPUT_ROOT / f"{timestamp}_{EXPERIMENT_NAME}"
    experiment_directory.mkdir(parents=True, exist_ok=False)
    summary_path = experiment_directory / "summary.json"
    episodes_path = experiment_directory / "episodes.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()
    started_time = time.monotonic()
    model = SteveOnePolicy.from_pretrained(CHECKPOINT_DIRECTORY).to(DEVICE).eval()
    first_seed = episode_seed(0, 0)
    simulator = MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=first_seed,
        preferred_spawn_biome=PREFERRED_SPAWN_BIOME,
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[],
    )
    results = []
    try:
        for skill_index, skill in enumerate(ACTIVE_SKILLS):
            for current_episode in range(EPISODES_PER_SKILL):
                seed = episode_seed(skill_index, current_episode)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                condition = model.prepare_condition(
                    {"cond_scale": CONDITION_SCALE, "text": skill["prompt"]},
                    deterministic=DETERMINISTIC_TEXT_PRIOR,
                )
                simulator.seed = seed
                simulator.env.seed(seed)
                observation, info = simulator.reset()
                baseline = state_snapshot(info)
                current = baseline
                recurrent_state = None
                terminated = False
                truncated = False
                primary_progress = 0.0
                secondary_progress = 0.0
                video_path = experiment_directory / f"{skill['skill_id']}_{current_episode:03d}_{seed}.mp4"
                container = None
                stream = None
                if SAVE_VIDEO:
                    container, stream = create_video(video_path)
                    initial_frame = overlay_frame(np.asarray(info["pov"]), skill, current_episode, 0, primary_progress, secondary_progress)
                    write_video_frame(container, stream, initial_frame)
                action_counts = {"attack": 0, "forward": 0, "jump": 0, "inventory": 0, "use": 0, "camera": 0}
                agent_button_counts = {}
                agent_camera_counts = {}
                episode_started = time.monotonic()
                steps_taken = 0
                for step in range(1, MAX_STEPS + 1):
                    image = torch.from_numpy(observation["image"]).unsqueeze(0).unsqueeze(0).to(DEVICE)
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
                    agent_button_counts[button_code] = agent_button_counts.get(button_code, 0) + 1
                    agent_camera_counts[camera_code] = agent_camera_counts.get(camera_code, 0) + 1
                    observation, reward, terminated, truncated, info = simulator.step(action)
                    steps_taken = step
                    for name in action_counts:
                        if name == "camera":
                            action_counts[name] += int(np.any(np.asarray(action.get(name, [0.0, 0.0]))) != 0)
                        else:
                            action_counts[name] += int(np.asarray(action.get(name, 0)).item() != 0)
                    current = state_snapshot(info)
                    primary_progress = progress(skill["primary_source"], skill["primary_target"], baseline, current)
                    secondary_progress = progress(skill["secondary_source"], skill["secondary_target"], baseline, current)
                    if SAVE_VIDEO:
                        rendered = overlay_frame(np.asarray(info["pov"]), skill, current_episode, step, primary_progress, secondary_progress)
                        write_video_frame(container, stream, rendered)
                    if primary_progress >= skill["primary_threshold"] or terminated or truncated:
                        break
                if SAVE_VIDEO:
                    close_video(container, stream)
                final_frame_path = experiment_directory / f"{skill['skill_id']}_{current_episode:03d}_{seed}_final.jpg"
                final_frame = cv2.cvtColor(np.asarray(info["pov"]), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(final_frame_path), final_frame):
                    raise RuntimeError(f"Could not write {final_frame_path}")
                start_pos = baseline["player_pos"]
                end_pos = current["player_pos"]
                horizontal_displacement = math.hypot(end_pos["x"] - start_pos["x"], end_pos["z"] - start_pos["z"])
                result = {
                    "skill_id": skill["skill_id"],
                    "prompt": skill["prompt"],
                    "episode_index": current_episode,
                    "seed": seed,
                    "success": primary_progress >= skill["primary_threshold"],
                    "primary_progress": primary_progress,
                    "primary_verifier": {
                        "source": skill["primary_source"],
                        "target": skill["primary_target"],
                        "threshold": skill["primary_threshold"],
                    },
                    "secondary_success": secondary_progress >= skill["secondary_threshold"],
                    "secondary_progress": secondary_progress,
                    "secondary_verifier": {
                        "source": skill["secondary_source"],
                        "target": skill["secondary_target"],
                        "threshold": skill["secondary_threshold"],
                    },
                    "steps": steps_taken,
                    "episode_seconds": time.monotonic() - episode_started,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "horizontal_displacement": horizontal_displacement,
                    "action_counts": action_counts,
                    "agent_button_counts": agent_button_counts,
                    "agent_camera_counts": agent_camera_counts,
                    "condition_embedding": {
                        "mean": float(condition["mineclip_embeds"].mean().item()),
                        "std": float(condition["mineclip_embeds"].std().item()),
                        "norm": float(condition["mineclip_embeds"].norm().item()),
                    },
                    "baseline_state": baseline,
                    "final_state": current,
                    "video": video_path.name if SAVE_VIDEO else None,
                    "final_frame": final_frame_path.name,
                }
                results.append(result)
                with episodes_path.open("a", encoding="utf-8") as episodes_file:
                    episodes_file.write(json.dumps(result, separators=(",", ":")) + "\n")
                summary = summarize_results(results, started_at, time.monotonic() - started_time)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({
                    "complete": len(results),
                    "total": len(ACTIVE_SKILLS) * EPISODES_PER_SKILL,
                    "skill": skill["skill_id"],
                    "seed": seed,
                    "success": result["success"],
                    "broke_log": result["secondary_success"],
                    "steps": steps_taken,
                    "seconds": round(result["episode_seconds"], 2),
                    "output": str(experiment_directory),
                }), flush=True)
    finally:
        simulator.close()
    print(json.dumps(summarize_results(results, started_at, time.monotonic() - started_time), indent=2))


if __name__ == "__main__":
    main()
