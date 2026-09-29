from copy import deepcopy
from unittest import TestCase

from minestudio.system_zero import LegacyClosedLoopBackend, SystemZeroStatus


def observation_with(items: dict[str, int] | None = None) -> dict:
    return {
        "inventory": {slot: {"type": item, "quantity": quantity} for slot, (item, quantity) in enumerate((items or {}).items())},
        "equipped_items": {"mainhand": {"type": "air", "quantity": 0}},
        "mine_block": {},
    }


class FakeLegacyController:
    def __init__(self, break_target: bool = True, break_item: str = "stone", collect_needs_system1: bool = False):
        self.steps = 0
        self.break_target = break_target
        self.break_item = break_item
        self.collect_needs_system1 = collect_needs_system1
        self.info = observation_with()

    def craft_recipe(self, recipe, info):
        self.steps += 4
        updated = deepcopy(info)
        updated["inventory"][9] = {"type": recipe, "quantity": 1}
        return {"success": True, "reason": "inventory delta verified"}, updated

    def equip_item(self, item, info):
        self.steps += 2
        updated = deepcopy(info)
        updated["equipped_items"]["mainhand"] = {"type": item, "quantity": 1}
        self.info = updated
        return {"success": True, "reason": "mainhand verified"}, updated

    def place_block(self, item, info):
        self.steps += 3
        return {"success": True, "reason": "inventory consumption verified"}, deepcopy(info)

    def reclaim_block(self, item, info, max_steps):
        self.steps += 5
        updated = deepcopy(info)
        updated["inventory"][9] = {"type": item, "quantity": 1}
        return {"success": True, "reason": "inventory delta verified"}, updated

    def collect_drop(self, item, baseline, info, max_steps):
        self.steps += 2
        if self.collect_needs_system1:
            return {"success": False, "needs_system1": True, "status": "NEEDS_SYSTEM1", "reason": "pickup geometry uncertain"}, deepcopy(info)
        updated = deepcopy(info)
        updated["inventory"][9] = {"type": item, "quantity": baseline + 1}
        return {"success": True, "reason": "inventory delta verified"}, updated

    def step_values(self, values, phase):
        self.steps += 1
        updated = deepcopy(self.info)
        if values.get("attack") == 1 and self.break_target and self.steps >= 3:
            updated["mine_block"][self.break_item] = 1
        self.info = updated
        return {}, updated


class LegacyBackendTests(TestCase):
    def test_maps_legacy_methods_to_structured_success(self):
        operations = (
            ("CRAFT_RECIPE", {"recipe": "crafting_table"}),
            ("EQUIP_ITEM", {"item": "stone_pickaxe"}),
            ("PLACE_BLOCK", {"item": "cobblestone"}),
            ("PLACE_STATION", {"station": "crafting_table"}),
        )
        for action, parameters in operations:
            with self.subTest(action=action):
                controller = FakeLegacyController()
                backend = LegacyClosedLoopBackend(controller)
                result = backend.execute(action, parameters, observation_with(), 100, lambda: False)
                self.assertEqual(result.status, SystemZeroStatus.SUCCESS)
                self.assertGreater(result.steps, 0)

    def test_reclaim_breaks_then_collects_within_budget(self):
        controller = FakeLegacyController(break_item="crafting_table")
        controller.info = observation_with()
        backend = LegacyClosedLoopBackend(controller)
        result = backend.execute(
            "RECLAIM_BLOCK",
            {"item": "crafting_table"},
            controller.info,
            40,
            lambda: False,
        )
        self.assertEqual(result.status, SystemZeroStatus.SUCCESS)
        self.assertLessEqual(result.steps, 40)

    def test_mine_and_collect_breaks_then_collects(self):
        controller = FakeLegacyController()
        controller.info = observation_with({"wooden_pickaxe": 1})
        backend = LegacyClosedLoopBackend(controller)
        result = backend.execute(
            "MINE_AND_COLLECT",
            {"target_type": "stone", "drop_item": "cobblestone", "count": 1, "tool": "wooden_pickaxe"},
            controller.info,
            40,
            lambda: False,
        )
        self.assertEqual(result.status, SystemZeroStatus.SUCCESS)
        self.assertGreaterEqual(result.observation["inventory"][9]["quantity"], 1)

    def test_mine_and_collect_requests_system_one_when_target_is_not_broken(self):
        controller = FakeLegacyController(break_target=False)
        controller.info = observation_with()
        backend = LegacyClosedLoopBackend(controller)
        result = backend.execute(
            "MINE_AND_COLLECT",
            {"target_type": "stone", "drop_item": "cobblestone", "count": 1, "tool": None},
            controller.info,
            12,
            lambda: False,
        )
        self.assertEqual(result.status, SystemZeroStatus.NEEDS_SYSTEM1)

    def test_collect_known_drop_succeeds_on_inventory_delta(self):
        controller = FakeLegacyController()
        backend = LegacyClosedLoopBackend(controller)
        result = backend.execute(
            "COLLECT_KNOWN_DROP",
            {"item": "cobblestone", "target_known": True},
            observation_with(),
            20,
            lambda: False,
        )
        self.assertEqual(result.status, SystemZeroStatus.SUCCESS)

    def test_collect_known_drop_requests_system_one_for_uncertain_geometry(self):
        controller = FakeLegacyController(collect_needs_system1=True)
        backend = LegacyClosedLoopBackend(controller)
        result = backend.execute(
            "COLLECT_KNOWN_DROP",
            {"item": "cobblestone", "target_known": True},
            observation_with(),
            20,
            lambda: False,
        )
        self.assertEqual(result.status, SystemZeroStatus.NEEDS_SYSTEM1)

    def test_unknown_operation_requests_system_one(self):
        backend = LegacyClosedLoopBackend(FakeLegacyController())
        result = backend.execute("FIND_NEAREST_IRON", {}, observation_with(), 10, lambda: False)
        self.assertEqual(result.status, SystemZeroStatus.NEEDS_SYSTEM1)
