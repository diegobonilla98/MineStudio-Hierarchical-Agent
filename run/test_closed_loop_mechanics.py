import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from closed_loop_options import ClosedLoopOptions, grouped_inventory_value
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIRECTORY / "output" / "closed_loop_mechanics_smoke.json"
FRAME_DIRECTORY = PROJECT_DIRECTORY / "output" / "closed_loop_mechanics_frames"
WORLD_SEED = 20260915
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
SETUP_COMMANDS = [
    "/clear @p",
    "/time set day",
    "/weather clear",
    "/fill ~-5 ~-1 ~-5 ~5 ~-1 ~5 minecraft:stone",
    "/fill ~-5 ~ ~-5 ~5 ~4 ~5 minecraft:air",
    "/give @p minecraft:oak_log 3",
    "/give @p minecraft:cobblestone 3",
]
RECIPES = ["planks", "crafting_table", "stick"]


def main() -> None:
    simulator = MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=WORLD_SEED,
        preferred_spawn_biome="forest",
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[CommandsCallback(SETUP_COMMANDS)],
    )
    records = []
    frame_index = 0

    def traced_step(action: dict, phase: str):
        nonlocal frame_index
        result = simulator.step(action)
        frame_index += 1
        frame = cv2.cvtColor(np.asarray(result[4]["pov"]), cv2.COLOR_RGB2BGR)
        frame_name = f"{frame_index:04d}_{phase.replace('/', '_')}.jpg"
        if not cv2.imwrite(str(FRAME_DIRECTORY / frame_name), frame):
            raise RuntimeError(f"Could not save {frame_name}")
        return result

    try:
        observation, info = simulator.reset()
        for _ in range(4):
            observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
        FRAME_DIRECTORY.mkdir(parents=True, exist_ok=True)
        controller = ClosedLoopOptions(simulator, traced_step)
        gating_check = True
        for allowed_control in ("forward", "attack"):
            allowed_action = controller.environment_action({allowed_control: 1})
            gated_action, gated_environment_action, blocked_controls = controller.gate_world_control(allowed_action)
            decoded_gated_action = simulator.agent_action_to_env_action(deepcopy(gated_action))
            gating_check = gating_check and int(np.asarray(decoded_gated_action[allowed_control]).item()) == 1 and not blocked_controls
        for blocked_control in ("use", "inventory", "hotbar.2"):
            unsafe_action = controller.environment_action({blocked_control: 1})
            gated_action, gated_environment_action, blocked_controls = controller.gate_world_control(unsafe_action)
            decoded_gated_action = simulator.agent_action_to_env_action(deepcopy(gated_action))
            gating_check = gating_check and int(np.asarray(decoded_gated_action[blocked_control]).item()) == 0 and blocked_control in blocked_controls
        for recipe in RECIPES:
            result, info = controller.craft_recipe(recipe, info)
            records.append(result)
        result, info = controller.place_block("crafting_table", info)
        records.append(result)
        result, info = controller.craft_recipe("wooden_pickaxe", info)
        records.append(result)
        result, info = controller.equip_item("wooden_pickaxe", info)
        records.append(result)
        result, info = controller.craft_recipe("stone_pickaxe", info)
        records.append(result)
        result, info = controller.equip_item("stone_pickaxe", info)
        records.append(result)
        checks = {
            "all_options_succeeded": all(record["success"] for record in records),
            "stone_pickaxe_in_inventory": grouped_inventory_value(info, "stone_pickaxe") >= 1,
            "stone_pickaxe_equipped": info.get("equipped_items", {}).get("mainhand", {}).get("type") == "stone_pickaxe",
            "ingredients_consumed": grouped_inventory_value(info, "log") == 0 and grouped_inventory_value(info, "cobblestone") == 0,
            "world_control_gating": gating_check,
        }
        result = {
            "passed": all(checks.values()),
            "checks": checks,
            "records": records,
            "final_inventory": info.get("inventory"),
            "final_equipped_items": info.get("equipped_items"),
            "total_environment_steps": controller.steps,
        }
    finally:
        simulator.close()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    if not result["passed"]:
        raise RuntimeError(f"Closed-loop mechanics smoke failed: {checks}")


if __name__ == "__main__":
    main()
