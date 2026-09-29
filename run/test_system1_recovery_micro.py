import os
import unittest


os.environ.setdefault("MINESTUDIO_MICRO_TASK", "EXIT_WATER")

import benchmark_system1_recovery_micro as benchmark
from benchmark_system1_recovery_micro import MicroVerifier, TASKS, make_episode_config, scenario_commands


def state(y: float = 64.0, water: int = 0, yaw: float = 0.0, pitch: float = 0.0, stone: float = 0.0, dirt: float = 0.0) -> dict:
    mined = {}
    if stone:
        mined["stone"] = stone
    if dirt:
        mined["dirt"] = dirt
    return {
        "inventory": {},
        "events": {"mine_block": mined},
        "position": {"x": 0.0, "y": y, "z": 0.0, "yaw": yaw, "pitch": pitch},
        "health": 20.0,
        "air": 300.0,
        "is_alive": True,
        "water": water,
        "voxels": [{"x": 0, "y": -1, "z": 0, "type": "minecraft:grass_block"}],
    }


class ManifestTests(unittest.TestCase):
    def test_all_tasks_have_valid_configs_and_commands(self) -> None:
        original_task = benchmark.TASK_NAME
        for task in TASKS:
            benchmark.TASK_NAME = task
            first = make_episode_config(0)
            second = make_episode_config(0)
            self.assertEqual(first, second)
            self.assertEqual(first["task"], task)
            commands = scenario_commands(first)
            self.assertTrue(commands)
            self.assertTrue(any("wooden_pickaxe" in command for command in commands))
        benchmark.TASK_NAME = original_task


class VerifierTests(unittest.TestCase):
    def config(self, task: str) -> dict:
        original_task = benchmark.TASK_NAME
        benchmark.TASK_NAME = task
        value = make_episode_config(0)
        benchmark.TASK_NAME = original_task
        return value

    def test_exit_water(self) -> None:
        verifier = MicroVerifier(self.config("EXIT_WATER"), state(water=1))
        outcome = None
        for step in range(10):
            outcome, details = verifier.update(state(water=0), step + 1)
        self.assertEqual(outcome, "SUCCESS")

    def test_climb_shore(self) -> None:
        verifier = MicroVerifier(self.config("CLIMB_SHORE"), state(water=1))
        outcome = None
        for step in range(10):
            outcome, details = verifier.update(state(y=65.0, water=0), step + 1)
        self.assertEqual(outcome, "SUCCESS")

    def test_reacquire_stone(self) -> None:
        verifier = MicroVerifier(self.config("REACQUIRE_STONE"), state())
        outcome, details = verifier.update(state(stone=1), 1)
        self.assertEqual(outcome, "SUCCESS")

    def test_escape_hole(self) -> None:
        verifier = MicroVerifier(self.config("ESCAPE_HOLE"), state(y=63.0))
        outcome = None
        for step in range(10):
            outcome, details = verifier.update(state(y=64.0), step + 1)
        self.assertEqual(outcome, "SUCCESS")

    def test_digging_trap_failure(self) -> None:
        verifier = MicroVerifier(self.config("AVOID_DIGGING_TRAP"), state())
        outcome, details = verifier.update(state(y=63.0, dirt=1), 1)
        self.assertEqual(outcome, "FAILURE")
        self.assertEqual(details["reason"], "DUG_INTO_TRAP")

    def test_camera_recovery(self) -> None:
        config = self.config("RECOVER_CAMERA")
        baseline = state()
        verifier = MicroVerifier(config, baseline)
        target_yaw = config["target"]["yaw"]
        outcome = None
        for step in range(8):
            outcome, details = verifier.update(state(yaw=target_yaw, pitch=15.0), step + 1)
        self.assertEqual(outcome, "SUCCESS")

    def test_avoid_water_failure(self) -> None:
        verifier = MicroVerifier(self.config("AVOID_WATER"), state())
        outcome, details = verifier.update(state(water=1), 1)
        self.assertEqual(outcome, "FAILURE")
        self.assertEqual(details["reason"], "ENTERED_WATER")


if __name__ == "__main__":
    unittest.main()
