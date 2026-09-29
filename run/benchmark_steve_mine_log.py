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


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = PROJECT_DIRECTORY / "checkpoints" / "steve_one_official"
OUTPUT_ROOT = PROJECT_DIRECTORY / "output" / "steve_mine_log_benchmark"
EXPERIMENT_NAME = "mine_log_100"
PROMPT = "mine a log"
CONDITION_SCALE = 6.0
DETERMINISTIC_TEXT_PRIOR = False
DETERMINISTIC_ACTIONS = False
TOTAL_EPISODES = 100
NUM_WORKERS = 4
MAX_STEPS = 1000
BASE_WORLD_SEED = 20260915
PREFERRED_SPAWN_BIOME = "forest"
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
VIDEO_FPS = 20
SAVE_VIDEO_EPISODES = {0, 1, 2, 3}
RESUME_DIRECTORY_NAME = "20260915T092936Z_mine_log_100"
MAX_WORKER_WAVES = 8
MAX_EPISODES_PER_SIMULATOR = 6
ITEM_GROUP_SUFFIXES = {
    "log": ("_log", "_stem", "hyphae"),
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
        "pickup": numeric_mapping(info, "pickup"),
        "player_pos": {
            "x": float(player_pos.get("x", 0.0)),
            "y": float(player_pos.get("y", 0.0)),
            "z": float(player_pos.get("z", 0.0)),
        },
        "health": float(info.get("health", 0.0)),
        "food_level": float(info.get("food_level", 0.0)),
    }


def unbatch_action(action: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value[0][0] for name, value in action.items()}


def overlay_frame(frame: np.ndarray, episode_index: int, step: int, acquired: float, broken: float) -> np.ndarray:
    image = np.ascontiguousarray(frame.copy())
    lines = [
        f"STEVE-1: {PROMPT}",
        f"episode {episode_index + 1}/{TOTAL_EPISODES}  step {step}/{MAX_STEPS}",
        f"log acquired: {acquired:.0f}  log broken: {broken:.0f}",
    ]
    cv2.rectangle(image, (0, 0), (image.shape[1], 82), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        cv2.putText(image, line, (12, 23 + 25 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return image


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


def episode_seed(episode_index: int) -> int:
    return BASE_WORLD_SEED + episode_index * 7919


def run_worker(worker_index: int, episode_indices: list[int], experiment_directory: str, result_queue: multiprocessing.Queue) -> None:
    current_episode = None
    try:
        torch.set_float32_matmul_precision("high")
        model = SteveOnePolicy.from_pretrained(CHECKPOINT_DIRECTORY).to("cuda").eval()
        first_seed = episode_seed(episode_indices[0])
        simulator = MinecraftSim(
            action_type="agent",
            obs_size=OBSERVATION_SIZE,
            render_size=RENDER_SIZE,
            seed=first_seed,
            preferred_spawn_biome=PREFERRED_SPAWN_BIOME,
            num_empty_frames=EMPTY_FRAMES,
            callbacks=[],
        )
        output_directory = Path(experiment_directory)
        try:
            for worker_episode_index, episode_index in enumerate(episode_indices):
                current_episode = episode_index
                if worker_episode_index > 0 and worker_episode_index % MAX_EPISODES_PER_SIMULATOR == 0:
                    simulator.close()
                    simulator = MinecraftSim(
                        action_type="agent",
                        obs_size=OBSERVATION_SIZE,
                        render_size=RENDER_SIZE,
                        seed=episode_seed(episode_index),
                        preferred_spawn_biome=PREFERRED_SPAWN_BIOME,
                        num_empty_frames=EMPTY_FRAMES,
                        callbacks=[],
                    )
                seed = episode_seed(episode_index)
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                condition = model.prepare_condition(
                    {"cond_scale": CONDITION_SCALE, "text": PROMPT},
                    deterministic=DETERMINISTIC_TEXT_PRIOR,
                )
                simulator.seed = seed
                simulator.env.seed(seed)
                observation, info = simulator.reset()
                baseline = state_snapshot(info)
                current = baseline
                recurrent_state = None
                acquired = 0.0
                broken = 0.0
                terminated = False
                truncated = False
                steps_taken = 0
                video_path = output_directory / f"episode_{episode_index:03d}_{seed}.mp4"
                container = None
                stream = None
                if episode_index in SAVE_VIDEO_EPISODES:
                    container, stream = create_video(video_path)
                    write_video_frame(container, stream, overlay_frame(np.asarray(info["pov"]), episode_index, 0, acquired, broken))
                agent_button_counts = {}
                agent_camera_counts = {}
                action_counts = {"attack": 0, "forward": 0, "jump": 0, "inventory": 0, "use": 0, "camera": 0}
                episode_started = time.monotonic()
                for step in range(1, MAX_STEPS + 1):
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
                    acquired = grouped_value(current["inventory"], "log") - grouped_value(baseline["inventory"], "log")
                    broken = grouped_value(current["mine_block"], "log") - grouped_value(baseline["mine_block"], "log")
                    if container is not None:
                        write_video_frame(container, stream, overlay_frame(np.asarray(info["pov"]), episode_index, step, acquired, broken))
                    if acquired >= 1 or terminated or truncated:
                        break
                if container is not None:
                    close_video(container, stream)
                final_frame_path = output_directory / f"episode_{episode_index:03d}_{seed}_final.jpg"
                final_frame = cv2.cvtColor(np.asarray(info["pov"]), cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(final_frame_path), final_frame):
                    raise RuntimeError(f"Could not write {final_frame_path}")
                start_pos = baseline["player_pos"]
                end_pos = current["player_pos"]
                result_queue.put({
                    "kind": "episode",
                    "result": {
                        "episode_index": episode_index,
                        "worker_index": worker_index,
                        "seed": seed,
                        "prompt": PROMPT,
                        "success": acquired >= 1,
                        "acquired_logs": acquired,
                        "broke_log": broken >= 1,
                        "broken_logs": broken,
                        "steps": steps_taken,
                        "episode_seconds": time.monotonic() - episode_started,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "horizontal_displacement": math.hypot(end_pos["x"] - start_pos["x"], end_pos["z"] - start_pos["z"]),
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
                        "video": video_path.name if episode_index in SAVE_VIDEO_EPISODES else None,
                        "final_frame": final_frame_path.name,
                    },
                })
        finally:
            simulator.close()
    except Exception:
        result_queue.put({"kind": "error", "worker_index": worker_index, "episode_index": current_episode, "traceback": traceback.format_exc()})


def build_summary(results: list[dict], started_at: str, elapsed_seconds: float) -> dict:
    ordered = sorted(results, key=lambda result: result["episode_index"])
    success_count = sum(result["success"] for result in ordered)
    broke_count = sum(result["broke_log"] for result in ordered)
    return {
        "experiment_name": EXPERIMENT_NAME,
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(CHECKPOINT_DIRECTORY),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "prompt": PROMPT,
        "condition_scale": CONDITION_SCALE,
        "deterministic_text_prior": DETERMINISTIC_TEXT_PRIOR,
        "deterministic_actions": DETERMINISTIC_ACTIONS,
        "scenario": "random Minecraft seed with a forest-biome spawn preference and no starting inventory",
        "total_episodes": TOTAL_EPISODES,
        "episodes_complete": len(ordered),
        "num_workers": NUM_WORKERS,
        "max_steps": MAX_STEPS,
        "successes": success_count,
        "success_rate": success_count / len(ordered) if ordered else 0.0,
        "broke_log_episodes": broke_count,
        "broke_log_rate": broke_count / len(ordered) if ordered else 0.0,
        "mean_steps": sum(result["steps"] for result in ordered) / len(ordered) if ordered else 0.0,
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
    results_by_index = {}
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            result = json.loads(line)
            results_by_index[result["episode_index"]] = result
    print(json.dumps({"resumed": len(results_by_index), "total": TOTAL_EPISODES, "output": str(experiment_directory)}), flush=True)
    for wave_index in range(MAX_WORKER_WAVES):
        pending_indices = [episode_index for episode_index in range(TOTAL_EPISODES) if episode_index not in results_by_index]
        if not pending_indices:
            break
        worker_count = min(NUM_WORKERS, len(pending_indices))
        assignments = [pending_indices[worker_index::worker_count] for worker_index in range(worker_count)]
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=run_worker,
                args=(worker_index, assignments[worker_index], str(experiment_directory), result_queue),
            )
            for worker_index in range(worker_count)
        ]
        for process in processes:
            process.start()
        wave_failed = None
        wave_completed = 0
        while wave_completed < len(pending_indices):
            try:
                message = result_queue.get(timeout=180)
            except queue.Empty:
                dead_processes = [process for process in processes if not process.is_alive() and process.exitcode != 0]
                if dead_processes:
                    wave_failed = {"kind": "error", "traceback": f"Workers exited without a result: {[process.exitcode for process in dead_processes]}"}
                    break
                print(json.dumps({"complete": len(results_by_index), "total": TOTAL_EPISODES, "wave": wave_index + 1, "status": "waiting"}), flush=True)
                continue
            if message["kind"] == "error":
                wave_failed = message
                break
            result = message["result"]
            if result["episode_index"] in results_by_index:
                continue
            results_by_index[result["episode_index"]] = result
            wave_completed += 1
            with episodes_path.open("a", encoding="utf-8") as episodes_file:
                episodes_file.write(json.dumps(result, separators=(",", ":")) + "\n")
            summary = build_summary(list(results_by_index.values()), started_at, time.monotonic() - started_time)
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "complete": len(results_by_index),
                "total": TOTAL_EPISODES,
                "episode": result["episode_index"],
                "seed": result["seed"],
                "success": result["success"],
                "broke_log": result["broke_log"],
                "steps": result["steps"],
                "seconds": round(result["episode_seconds"], 2),
                "running_success_rate": summary["success_rate"],
                "wave": wave_index + 1,
                "output": str(experiment_directory),
            }), flush=True)
        if wave_failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=30)
            with (experiment_directory / "worker_errors.jsonl").open("a", encoding="utf-8") as error_file:
                error_file.write(json.dumps({"wave": wave_index + 1, **wave_failed}, separators=(",", ":")) + "\n")
            print(json.dumps({"wave": wave_index + 1, "status": "restarting_failed_workers", "complete": len(results_by_index), "error_worker": wave_failed.get("worker_index")}), flush=True)
            continue
        for process in processes:
            process.join(timeout=30)
    results = list(results_by_index.values())
    missing_indices = [episode_index for episode_index in range(TOTAL_EPISODES) if episode_index not in results_by_index]
    if missing_indices:
        raise RuntimeError(f"Benchmark exhausted worker restart waves with missing episodes: {missing_indices}")
    summary = build_summary(results, started_at, time.monotonic() - started_time)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
