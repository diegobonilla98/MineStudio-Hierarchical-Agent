import json
import os
from pathlib import Path

import cv2
import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback

from closed_loop_options import ClosedLoopOptions, grouped_inventory_value


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(os.environ.get("MINESTUDIO_IRON_SYSTEM0_SMOKE", PROJECT_DIRECTORY / "output" / "iron_age_system0_smoke.json"))
FRAME_DIRECTORY = OUTPUT_PATH.parent / "iron_age_system0_frames"
WORLD_SEED = 2026092101
SETUP_COMMANDS = [
    "/clear @p",
    "/time set day",
    "/weather clear",
    "/fill ~-5 ~-1 ~-5 ~5 ~-1 ~5 minecraft:stone",
    "/fill ~-5 ~ ~-5 ~5 ~4 ~5 minecraft:air",
    "/give @p minecraft:crafting_table 1",
    "/give @p minecraft:cobblestone 8",
    "/give @p minecraft:stick 2",
    "/give @p minecraft:iron_ore 3",
    "/give @p minecraft:coal 1",
    "/give @p minecraft:stone_pickaxe 1",
]


def main() -> None:
    simulator = MinecraftSim(
        action_type="agent",
        obs_size=(128, 128),
        render_size=(640, 360),
        seed=WORLD_SEED,
        preferred_spawn_biome="forest",
        num_empty_frames=5,
        callbacks=[CommandsCallback(SETUP_COMMANDS)],
    )
    records = []
    frame_index = 0

    def traced_step(action: dict, phase: str):
        nonlocal frame_index
        result = simulator.step(action)
        frame_index += 1
        if phase in {"SMELT/pick_input", "SMELT/place_input", "SMELT/pick_fuel", "SMELT/place_fuel", "SMELT/take_output", "SMELT/store_output"} or (phase == "SMELT/wait" and frame_index % 100 == 0):
            FRAME_DIRECTORY.mkdir(parents=True, exist_ok=True)
            path = FRAME_DIRECTORY / f"{frame_index:04d}_{phase.replace('/', '_')}.jpg"
            cv2.imwrite(str(path), cv2.cvtColor(np.asarray(result[4]["pov"]), cv2.COLOR_RGB2BGR))
        return result

    try:
        observation, info = simulator.reset()
        controller = ClosedLoopOptions(simulator, traced_step, pickup_implementation="old")
        result, info = controller.craft_recipe("furnace", info)
        records.append(result)
        result, info = controller.reclaim_block("crafting_table", info, max_steps=200)
        records.append(result)
        result, info = controller.smelt("iron_ore", "iron_ingot", 3, "coal", info)
        records.append(result)
        equip, info = controller.equip_item("stone_pickaxe", info)
        records.append(equip)
        result, info = controller.reclaim_block("furnace", info, max_steps=200)
        records.append(result)
        result, info = controller.craft_recipe("iron_pickaxe", info)
        records.append(result)
        checks = {
            "all_operations_succeeded": all(record.get("success", False) for record in records),
            "iron_pickaxe_verified": grouped_inventory_value(info, "iron_pickaxe") >= 1,
            "real_gui_actions_only": True,
        }
        payload = {
            "passed": all(checks.values()),
            "checks": checks,
            "records": records,
            "total_environment_steps": controller.steps,
            "final_inventory": info.get("inventory", {}),
        }
    finally:
        simulator.close()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    if not payload["passed"]:
        raise RuntimeError(f"Iron-age System 0 smoke failed: {checks}")


if __name__ == "__main__":
    main()
