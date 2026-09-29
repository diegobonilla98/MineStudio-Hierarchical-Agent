import gzip
import json
from datetime import datetime, timezone

import numpy as np
import torch
from benchmark_stone_acquisition import (
    BIOMES,
    CHECKPOINT_DIRECTORY,
    ENV_BUTTONS,
    EXPERIMENT_DIRECTORY,
    SCENARIO_WEIGHTS,
    SteveOnePolicy,
    classify_trajectory,
    create_simulator,
    make_manifest,
    run_episode,
    scenario_commands,
)
from minestudio.simulator.callbacks import CommandsCallback


RUN_LIVE_SMOKE = True
LIVE_MAX_STEPS = 160


def synthetic_state(cobblestone: float = 0.0, stone_mined: float = 0.0, health: float = 20.0, air: float = 300.0, alive: bool = True, y: float = 64.0, water: int = 0) -> dict:
    inventory = {"cobblestone": cobblestone} if cobblestone else {}
    mine_block = {"stone": stone_mined} if stone_mined else {}
    return {
        "inventory": inventory,
        "events": {
            "mine_block": mine_block,
            "pickup": {},
            "craft_item": {},
            "place_block": {},
            "kill_entity": {},
            "break_item": {},
            "use_item": {},
            "damage_dealt": {},
        },
        "position": {"x": 0.0, "y": y, "z": 0.0, "yaw": 0.0, "pitch": 0.0},
        "health": health,
        "air": air,
        "is_alive": alive,
        "water": water,
        "voxels": ["water"] if water else ["air"],
    }


def synthetic_arrays(steps: int = 100) -> dict[str, np.ndarray]:
    return {
        "rgb": np.zeros((steps, 128, 128, 3), dtype=np.uint8),
        "agent_buttons": np.zeros(steps, dtype=np.int32),
        "agent_camera": np.full(steps, 60, dtype=np.int16),
        "env_buttons": np.zeros((steps, len(ENV_BUTTONS)), dtype=np.uint8),
        "env_camera": np.zeros((steps, 2), dtype=np.float32),
        "position": np.column_stack((np.zeros(steps), np.full(steps, 64.0), np.zeros(steps), np.zeros(steps), np.zeros(steps))).astype(np.float32),
        "health": np.full(steps, 20.0, dtype=np.float32),
        "air": np.full(steps, 300.0, dtype=np.float32),
        "is_alive": np.ones(steps, dtype=np.bool_),
        "water": np.zeros(steps, dtype=np.int8),
        "cobblestone": np.zeros(steps, dtype=np.float32),
        "stone_mined": np.zeros(steps, dtype=np.float32),
        "cobblestone_pickup": np.zeros(steps, dtype=np.float32),
        "movement_requested": np.zeros(steps, dtype=np.uint8),
    }


def test_manifest() -> dict:
    manifest = make_manifest()
    assert len(manifest) == 1000
    assert len({config["episode_id"] for config in manifest}) == len(manifest)
    assert set(config["biome"] for config in manifest) == set(BIOMES)
    assert set(config["scenario"] for config in manifest) == set(SCENARIO_WEIGHTS)
    assert set(config["hotbar_slot"] for config in manifest) == set(range(9))
    assert any(config["scenario_family"] == "natural" for config in manifest)
    assert any(config["water_proximity"] == "between_player_and_stone" for config in manifest)
    for scenario in SCENARIO_WEIGHTS:
        config = next(item for item in manifest if item["scenario"] == scenario)
        commands = scenario_commands(config)
        assert any("wooden_pickaxe" in command for command in commands)
        assert (scenario == "natural") == (not any("minecraft:stone" in command for command in commands))
    return {
        "episodes": len(manifest),
        "biomes": sorted(set(config["biome"] for config in manifest)),
        "scenarios": {scenario: sum(config["scenario"] == scenario for config in manifest) for scenario in SCENARIO_WEIGHTS},
    }


def test_taxonomy() -> dict:
    baseline = synthetic_state()
    success_arrays = synthetic_arrays(20)
    success_final = synthetic_state(cobblestone=3.0, stone_mined=3.0)
    success = classify_trajectory(success_arrays, baseline, success_final, False, False, 20)
    assert success["primary_label"] == "SUCCESS"
    assert success["success"]

    pickup_failure_arrays = synthetic_arrays(300)
    pickup_failure_arrays["stone_mined"][50:] = 1.0
    pickup_failure_final = synthetic_state(stone_mined=1.0)
    pickup_failure = classify_trajectory(pickup_failure_arrays, baseline, pickup_failure_final, False, True, 300)
    assert "STONE_BROKEN_NOT_COLLECTED" in pickup_failure["labels"]
    assert "TARGET_LOST" in pickup_failure["labels"]
    assert "TIMEOUT" in pickup_failure["labels"]

    water_arrays = synthetic_arrays(100)
    water_arrays["water"][10:] = 1
    water_arrays["air"][20:] = np.linspace(300.0, 250.0, 80)
    water_arrays["movement_requested"][:] = 1
    water_failure_final = synthetic_state(air=245.0, water=1)
    water_failure = classify_trajectory(water_arrays, baseline, water_failure_final, False, True, 100)
    assert "ENTERED_WATER" in water_failure["labels"]
    assert "DROWNING" in water_failure["labels"]
    assert "STUCK" in water_failure["labels"]
    return {
        "success": success["labels"],
        "pickup_failure": pickup_failure["labels"],
        "water_failure": water_failure["labels"],
    }


def live_smoke() -> dict:
    manifest = make_manifest()
    config = next(item for item in manifest if item["scenario"] == "exposed_near" and item["stone_visibility"] == "initially_visible")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_directory = EXPERIMENT_DIRECTORY.parent / f"smoke_{timestamp}"
    output_directory.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    model = SteveOnePolicy.from_pretrained(CHECKPOINT_DIRECTORY).to("cuda").eval()
    command_callback = CommandsCallback([])
    simulator = create_simulator(config, command_callback)
    try:
        result = run_episode(model, simulator, command_callback, config, output_directory, max_steps=LIVE_MAX_STEPS)
    finally:
        simulator.close()
    npz_path = output_directory / result["trajectory_npz"]
    metadata_path = output_directory / result["trajectory_metadata"]
    with np.load(npz_path) as trajectory:
        assert trajectory["rgb"].shape == (result["steps"], 128, 128, 3)
        assert trajectory["env_buttons"].shape == (result["steps"], len(ENV_BUTTONS))
        assert trajectory["will_enter_water_40"].shape == (result["steps"],)
        assert trajectory["will_air_decrease_40"].shape == (result["steps"],)
        assert trajectory["will_make_progress_40"].shape == (result["steps"],)
        assert np.all(trajectory["air"] >= 0)
        assert np.all(np.isin(trajectory["water"], (-1, 0, 1)))
    with gzip.open(metadata_path, "rt", encoding="utf-8") as input_file:
        metadata = json.load(input_file)
    assert metadata["transition_contract"] == "row t stores observation/state before action t"
    assert len(metadata["phases"]) == result["steps"]
    assert len(metadata["inventories"]) == result["steps"]
    assert len(metadata["cumulative_events"]) == result["steps"]
    assert metadata["baseline_state"]["air"] >= 0
    assert metadata["baseline_state"]["water"] in (-1, 0, 1)
    return {
        "output": str(output_directory),
        "episode": result["episode_index"],
        "scenario": result["scenario"],
        "steps": result["steps"],
        "success": result["success"],
        "labels": result["labels"],
        "cobblestone_delta": result["cobblestone_delta"],
        "stone_mined_delta": result["stone_mined_delta"],
        "life_stats": metadata["baseline_state"]["air"],
        "voxel_state": metadata["baseline_state"]["water"],
        "trajectory_npz": str(npz_path),
        "trajectory_metadata": str(metadata_path),
    }


def main() -> None:
    if not CHECKPOINT_DIRECTORY.is_dir():
        raise FileNotFoundError(CHECKPOINT_DIRECTORY)
    manifest_result = test_manifest()
    taxonomy_result = test_taxonomy()
    live_result = live_smoke() if RUN_LIVE_SMOKE else None
    print(json.dumps({"manifest": manifest_result, "taxonomy": taxonomy_result, "live": live_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
