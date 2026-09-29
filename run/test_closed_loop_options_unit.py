from copy import deepcopy
from unittest import TestCase

from closed_loop_options import ClosedLoopOptions


def info_with(x: float = 0.0, y: float = 64.0, z: float = 0.0) -> dict:
    return {
        "inventory": {},
        "player_pos": {"x": x, "y": y, "z": z},
        "voxels": [],
    }


class CollectKnownDropTests(TestCase):
    def controller_with(self, updater, pickup_implementation="hardened"):
        controller = ClosedLoopOptions.__new__(ClosedLoopOptions)
        controller.steps = 0
        controller.pickup_implementation = pickup_implementation

        def step_values(values, phase):
            controller.steps += 1
            updated = updater(values, phase, controller.steps)
            return {}, updated

        controller.step_values = step_values
        return controller

    def test_inventory_is_verified_after_each_recovery_step(self):
        current = info_with()

        def update(values, phase, step):
            if values.get("forward"):
                current["player_pos"]["z"] += 0.1
            if step == 10:
                current["inventory"][0] = {"type": "cobblestone", "quantity": 1}
            return deepcopy(current)

        controller = self.controller_with(update)
        result, final = controller.collect_drop("cobblestone", 0, deepcopy(current), max_steps=30)
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["steps"], 10)
        self.assertEqual(final["inventory"][0]["quantity"], 1)

    def test_unexpected_drop_requests_system_one(self):
        current = info_with()

        def update(values, phase, step):
            if values.get("forward"):
                current["player_pos"]["y"] = 61.5
            return deepcopy(current)

        controller = self.controller_with(update)
        result, final = controller.collect_drop("cobblestone", 0, deepcopy(current), max_steps=30)
        self.assertFalse(result["success"])
        self.assertTrue(result["needs_system1"])
        self.assertEqual(result["status"], "NEEDS_SYSTEM1")

    def test_blocked_local_search_requests_system_one(self):
        current = info_with()

        def update(values, phase, step):
            return deepcopy(current)

        controller = self.controller_with(update)
        result, final = controller.collect_drop("cobblestone", 0, deepcopy(current), max_steps=60)
        self.assertFalse(result["success"])
        self.assertTrue(result["needs_system1"])
        self.assertIn("blocked", result["reason"])

    def test_old_pickup_replays_original_sweep_without_handoff(self):
        current = info_with()
        actions = []

        def update(values, phase, step):
            actions.append((deepcopy(values), phase))
            return deepcopy(current)

        controller = self.controller_with(update, pickup_implementation="old")
        result, final = controller.collect_drop("cobblestone", 0, deepcopy(current), max_steps=20)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "timeout")
        self.assertNotIn("needs_system1", result)
        self.assertEqual([action[0] for action in actions[:8]], [{"forward": 1}] * 8)
        self.assertTrue(all(action[1] == "COLLECT_DROP/sweep" for action in actions))

    def test_old_pickup_still_verifies_inventory_each_step(self):
        current = info_with()

        def update(values, phase, step):
            if step == 10:
                current["inventory"][0] = {"type": "cobblestone", "quantity": 1}
            return deepcopy(current)

        controller = self.controller_with(update, pickup_implementation="old")
        result, final = controller.collect_drop("cobblestone", 0, deepcopy(current), max_steps=30)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 10)
