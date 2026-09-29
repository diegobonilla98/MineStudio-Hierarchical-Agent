import json

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, VoxelsCallback
from benchmark_stone_acquisition import water_state


SEED = 2026091501
BIOME = "forest"
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
VOXEL_RANGES = (
    [-7, 7, -7, 7, -7, 7],
    [-2, 2, -2, 2, -2, 2],
    [0, 1, 0, 2, 0, 1],
    [-1, 1, -1, 2, -1, 1],
)


def main() -> None:
    commands = CommandsCallback([
        "/gamerule sendCommandFeedback false",
        "/time set day",
        "/weather clear",
        "/setblock ~ ~ ~ minecraft:water",
    ])
    voxels = VoxelsCallback(VOXEL_RANGES[0])
    simulator = MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=SEED,
        preferred_spawn_biome=BIOME,
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[commands, voxels],
        include_life_stats=True,
    )
    results = []
    try:
        observation, info = simulator.reset()
        for voxel_range in VOXEL_RANGES:
            voxels.voxels_ins = voxel_range
            for settle_index in range(3):
                observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
            value = info.get("voxels")
            results.append({
                "range": voxel_range,
                "type": type(value).__name__,
                "length": len(value) if hasattr(value, "__len__") else None,
                "sample": value[:3] if isinstance(value, list) else value,
                "body_water": water_state(info),
                "position": info.get("player_pos"),
                "air": info.get("life_stats", {}).get("air"),
            })
    finally:
        simulator.close()
    print(json.dumps(results, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)), flush=True)


if __name__ == "__main__":
    main()
