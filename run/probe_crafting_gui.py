from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output" / "crafting_gui_probe"
WORLD_SEED = 20260915
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
SETUP_COMMANDS = [
    "/clear @p",
    "/give @p minecraft:oak_log 3",
    "/give @p minecraft:cobblestone 3",
]
CURSOR_START = (320.0, 180.0)
PIXELS_PER_CAMERA_DEGREE = 6.6
MAX_CAMERA_DEGREES = 6.0
HOTBAR_SLOT_1 = (247.0, 248.0)
CRAFT_GRID_TOP_LEFT = (338.0, 123.0)
CRAFT_OUTPUT = (393.0, 133.0)


def environment_action(simulator: MinecraftSim, values: dict) -> dict:
    action = simulator.env.action_space.no_op()
    action.update(values)
    action = {
        key: np.asarray(value, dtype=np.float32).reshape(1, 2)
        if key == "camera"
        else np.asarray([value])
        for key, value in action.items()
        if key != "chat"
    }
    return simulator.env_action_to_agent_action(action)


def save_frame(path: Path, info: dict) -> None:
    frame = cv2.cvtColor(np.asarray(info["pov"]), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save {path}")


def step_values(simulator: MinecraftSim, values: dict) -> tuple[dict, dict]:
    observation, reward, terminated, truncated, info = simulator.step(environment_action(simulator, values))
    if terminated or truncated:
        raise RuntimeError(f"Environment ended: {info.get('error')}")
    return observation, info


def move_cursor(simulator: MinecraftSim, info: dict, current: tuple[float, float], target: tuple[float, float]) -> tuple[dict, tuple[float, float]]:
    cursor_x, cursor_y = current
    for _ in range(10):
        if abs(target[0] - cursor_x) <= 6 and abs(target[1] - cursor_y) <= 6:
            break
        yaw = np.clip((target[0] - cursor_x) / PIXELS_PER_CAMERA_DEGREE, -MAX_CAMERA_DEGREES, MAX_CAMERA_DEGREES)
        pitch = np.clip((target[1] - cursor_y) / PIXELS_PER_CAMERA_DEGREE, -MAX_CAMERA_DEGREES, MAX_CAMERA_DEGREES)
        agent_action = environment_action(simulator, {"camera": np.array([pitch, yaw], dtype=np.float32)})
        actual_camera = np.asarray(simulator.agent_action_to_env_action(deepcopy(agent_action))["camera"]).reshape(-1, 2)[0]
        observation, reward, terminated, truncated, info = simulator.step(agent_action)
        if terminated or truncated:
            raise RuntimeError(f"Environment ended while moving cursor: {info.get('error')}")
        cursor_x += float(actual_camera[1]) * PIXELS_PER_CAMERA_DEGREE
        cursor_y += float(actual_camera[0]) * PIXELS_PER_CAMERA_DEGREE
        if not np.any(actual_camera):
            break
    return info, (cursor_x, cursor_y)


def click(simulator: MinecraftSim) -> tuple[dict, dict]:
    observation, info = step_values(simulator, {"attack": 1})
    return step_values(simulator, {"attack": 0})


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    simulator = MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=WORLD_SEED,
        preferred_spawn_biome="forest",
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[CommandsCallback(SETUP_COMMANDS)],
    )
    try:
        observation, info = simulator.reset()
        save_frame(OUTPUT_DIRECTORY / "00_initial.jpg", info)
        observation, info = step_values(simulator, {"inventory": 1})
        save_frame(OUTPUT_DIRECTORY / "01_inventory_open.jpg", info)
        cursor = CURSOR_START
        info, cursor = move_cursor(simulator, info, cursor, HOTBAR_SLOT_1)
        observation, info = click(simulator)
        save_frame(OUTPUT_DIRECTORY / "02_log_on_cursor.jpg", info)
        info, cursor = move_cursor(simulator, info, cursor, CRAFT_GRID_TOP_LEFT)
        observation, info = click(simulator)
        save_frame(OUTPUT_DIRECTORY / "03_log_in_grid.jpg", info)
        info, cursor = move_cursor(simulator, info, cursor, CRAFT_OUTPUT)
        observation, info = click(simulator)
        save_frame(OUTPUT_DIRECTORY / "04_planks_on_cursor.jpg", info)
        info, cursor = move_cursor(simulator, info, cursor, HOTBAR_SLOT_1)
        observation, info = click(simulator)
        save_frame(OUTPUT_DIRECTORY / "05_planks_stored.jpg", info)
        print({"inventory": info.get("inventory"), "craft_item": info.get("craft_item")})
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
