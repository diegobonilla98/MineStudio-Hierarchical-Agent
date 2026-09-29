import json

from benchmark_stone_acquisition import (
    BIOMES,
    EMPTY_FRAMES,
    OBSERVATION_SIZE,
    RENDER_SIZE,
    SCENARIO_WEIGHTS,
    grouped_inventory_value,
    make_manifest,
    scenario_commands,
    voxel_records,
)
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, VoxelsCallback


SETTLE_STEPS = 5
VOXEL_RANGE = [-12, 12, -5, 7, -12, 12]


def block_positions(records: list[dict], block_suffix: str) -> set[tuple[int, int, int]]:
    return {
        (int(record["x"]), int(record["y"]), int(record["z"]))
        for record in records
        if str(record.get("type", "")).endswith(block_suffix)
    }


def selected_configs() -> list[dict]:
    manifest = make_manifest()
    configs = []
    for scenario_index, scenario in enumerate(SCENARIO_WEIGHTS):
        biome = BIOMES[scenario_index % len(BIOMES)]
        matching = [config for config in manifest if config["scenario"] == scenario and config["biome"] == biome]
        configs.append(next((config for config in matching if config["stone_visibility"] == "initially_visible"), matching[0]))
    return configs


def test_config(config: dict) -> dict:
    command_callback = CommandsCallback(scenario_commands(config))
    simulator = MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=config["seed"],
        preferred_spawn_biome=config["biome"],
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[command_callback, VoxelsCallback(VOXEL_RANGE)],
        include_life_stats=True,
    )
    try:
        observation, info = simulator.reset()
        for settle_index in range(SETTLE_STEPS):
            observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
        records = voxel_records(info.get("voxels"))
        assert records
        assert "life_stats" in info
        assert grouped_inventory_value(info, "wooden_pickaxe") >= 1
        position = info["player_pos"]
        assert abs(float(position["yaw"]) - float(config["initial_yaw"])) < 1.0
        assert abs(float(position["pitch"]) - float(config["initial_pitch"])) < 1.0
        stone_positions = block_positions(records, "stone")
        water_positions = block_positions(records, "water")
        leaf_positions = block_positions(records, "oak_leaves")
        if config["scenario"] != "natural":
            target = config["target"]
            perpendicular_x = -int(target["unit_z"])
            perpendicular_z = int(target["unit_x"])
            expected_stone = {
                (int(target["x"]), 0, int(target["z"])),
                (int(target["x"]), 1, int(target["z"])),
                (int(target["x"]) + perpendicular_x, 0, int(target["z"]) + perpendicular_z),
                (int(target["x"]) + perpendicular_x, 1, int(target["z"]) + perpendicular_z),
            }
            assert expected_stone.issubset(stone_positions), (config, expected_stone, stone_positions)
        if config["scenario"] == "water_near_stone":
            assert water_positions
        if config["scenario"] == "stone_slope":
            assert len(stone_positions) > 4
        if config["scenario"] == "vegetation_occluded":
            assert leaf_positions
        return {
            "episode": config["episode_index"],
            "scenario": config["scenario"],
            "biome": config["biome"],
            "position": position,
            "voxel_records": len(records),
            "stone_blocks": len(stone_positions),
            "water_blocks": len(water_positions),
            "leaf_blocks": len(leaf_positions),
            "air": info["life_stats"]["air"],
            "pickaxe": grouped_inventory_value(info, "wooden_pickaxe"),
        }
    finally:
        simulator.close()


def main() -> None:
    results = [test_config(config) for config in selected_configs()]
    assert set(result["scenario"] for result in results) == set(SCENARIO_WEIGHTS)
    assert set(result["biome"] for result in results) == set(BIOMES)
    print(json.dumps(results, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)), flush=True)


if __name__ == "__main__":
    main()
