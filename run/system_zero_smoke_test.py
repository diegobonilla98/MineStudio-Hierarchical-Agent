import json
import os
from pathlib import Path

from closed_loop_options import ClosedLoopOptions
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback
from minestudio.system_zero import LegacyClosedLoopBackend, SystemZeroContext, SystemZeroRequest, SystemZeroStatus, build_priority_one_registry


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIRECTORY / "output" / "system_zero" / "priority_one_smoke.json"
ATTEMPTS_PER_SKILL = int(os.environ.get("MINESTUDIO_SYSTEM_ZERO_ATTEMPTS", "1"))
WORLD_SEED = 20260919
OBSERVATION_SIZE = (128, 128)
RENDER_SIZE = (640, 360)
EMPTY_FRAMES = 5
BASE_COMMANDS = [
    "/clear @p",
    "/gamemode survival @p",
    "/time set day",
    "/weather clear",
    "/fill ~-6 ~-1 ~-6 ~6 ~-1 ~6 minecraft:stone",
    "/fill ~-6 ~ ~-6 ~6 ~5 ~6 minecraft:air",
    "/tp @p ~ ~ ~ 0 0",
]
CASES = (
    {
        "name": "CRAFT_RECIPE",
        "commands": ["/give @p minecraft:cobblestone 3", "/give @p minecraft:stick 2", "/give @p minecraft:crafting_table 1"],
        "request": SystemZeroRequest("CRAFT_RECIPE", {"recipe": "stone_pickaxe"}),
    },
    {
        "name": "EQUIP_ITEM",
        "commands": ["/give @p minecraft:stone_pickaxe 1"],
        "request": SystemZeroRequest("EQUIP_ITEM", {"item": "stone_pickaxe"}),
    },
    {
        "name": "PLACE_BLOCK",
        "commands": ["/give @p minecraft:cobblestone 1"],
        "request": SystemZeroRequest("PLACE_BLOCK", {"item": "cobblestone"}),
    },
    {
        "name": "RECLAIM_BLOCK",
        "commands": ["/setblock ~ ~1 ~3 minecraft:crafting_table"],
        "request": SystemZeroRequest("RECLAIM_BLOCK", {"item": "crafting_table", "target_known": True, "target_reachable": True}),
    },
    {
        "name": "MINE_AND_COLLECT",
        "commands": ["/give @p minecraft:wooden_pickaxe 1", "/setblock ~ ~1 ~3 minecraft:stone"],
        "request": SystemZeroRequest("MINE_AND_COLLECT", {"target_type": "stone", "drop_item": "cobblestone", "tool": "wooden_pickaxe", "target_known": True, "target_reachable": True}),
    },
    {
        "name": "COLLECT_KNOWN_DROP",
        "commands": ["/summon item ~ ~1 ~2 {Item:{id:\"minecraft:cobblestone\",Count:1b}}"],
        "request": SystemZeroRequest("COLLECT_KNOWN_DROP", {"item": "cobblestone", "target_known": True, "target_reachable": True}),
    },
    {
        "name": "PLACE_STATION",
        "commands": ["/give @p minecraft:crafting_table 1"],
        "request": SystemZeroRequest("PLACE_STATION", {"station": "crafting_table"}),
    },
)


def run_case(case: dict, attempt: int) -> dict:
    simulator = MinecraftSim(
        action_type="agent",
        obs_size=OBSERVATION_SIZE,
        render_size=RENDER_SIZE,
        seed=WORLD_SEED + attempt,
        preferred_spawn_biome="plains",
        num_empty_frames=EMPTY_FRAMES,
        callbacks=[CommandsCallback(BASE_COMMANDS + case["commands"])],
    )
    try:
        observation, info = simulator.reset()
        for _ in range(4):
            observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
        controller = ClosedLoopOptions(simulator)
        backend = LegacyClosedLoopBackend(controller)
        context = SystemZeroContext(observation=info, backend=backend)
        result = build_priority_one_registry().execute(case["request"], context)
        return {
            "skill": case["name"],
            "attempt": attempt,
            "passed": result.status == SystemZeroStatus.SUCCESS,
            "result": result.to_dict(),
        }
    finally:
        simulator.close()


def main() -> None:
    records = []
    for case in CASES:
        for attempt in range(ATTEMPTS_PER_SKILL):
            record = run_case(case, attempt)
            records.append(record)
            print(json.dumps(record), flush=True)
    summary = {}
    for case in CASES:
        skill_records = [record for record in records if record["skill"] == case["name"]]
        successes = sum(record["passed"] for record in skill_records)
        summary[case["name"]] = {
            "attempts": len(skill_records),
            "success": successes,
            "success_rate": successes / len(skill_records),
        }
    payload = {
        "passed": all(record["passed"] for record in records),
        "attempts_per_skill": ATTEMPTS_PER_SKILL,
        "summary": summary,
        "records": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(OUTPUT_PATH)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise RuntimeError(f"System Zero smoke failed: {summary}")


if __name__ == "__main__":
    main()
