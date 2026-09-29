from copy import deepcopy
from unittest import TestCase

from minestudio.system_zero import BackendResult, ExecutionRoute, ExecutionRouter, SystemZeroContext, SystemZeroRequest, SystemZeroStatus, build_priority_one_registry
from minestudio.system_zero.inventory import inventory_counts, normalize_identifier
from minestudio.system_zero.skills import RECIPES


def observation_with(items: dict[str, int] | None = None, equipped: str = "air") -> dict:
    inventory = {}
    for slot, (item, quantity) in enumerate((items or {}).items()):
        inventory[slot] = {"type": item, "quantity": quantity}
    return {
        "inventory": inventory,
        "equipped_items": {"mainhand": {"type": equipped, "quantity": 1 if equipped != "air" else 0}},
        "craft_item": {},
        "place_block": {},
        "mine_block": {},
        "is_gui_open": False,
    }


def add_item(observation: dict, item: str, quantity: int) -> None:
    normalized = normalize_identifier(item)
    for stack in observation["inventory"].values():
        if normalize_identifier(str(stack.get("type", "air"))) == normalized:
            stack["quantity"] = int(stack["quantity"]) + quantity
            return
    slot = max([-1, *[int(value) for value in observation["inventory"]]]) + 1
    observation["inventory"][slot] = {"type": normalized, "quantity": quantity}


def remove_item(observation: dict, item: str, quantity: int) -> None:
    remaining = quantity
    normalized = normalize_identifier(item)
    for stack in observation["inventory"].values():
        if normalize_identifier(str(stack.get("type", "air"))) == normalized and remaining > 0:
            removed = min(int(stack["quantity"]), remaining)
            stack["quantity"] = int(stack["quantity"]) - removed
            remaining -= removed


class FakeBackend:
    def __init__(self, mode: str = "success"):
        self.mode = mode
        self.calls = []

    def execute(self, action, parameters, observation, timeout_steps, should_abort):
        self.calls.append((action, dict(parameters), timeout_steps))
        after = deepcopy(observation)
        if self.mode == "timeout":
            return BackendResult(SystemZeroStatus.SUCCESS, after, steps=timeout_steps + 1)
        if self.mode == "needs_system1":
            return BackendResult(SystemZeroStatus.NEEDS_SYSTEM1, after, details={"reason": "target tracking failed"}, steps=1)
        if self.mode == "no_effect":
            return BackendResult(SystemZeroStatus.SUCCESS, after, details={"reason": "backend claimed success"}, steps=1)
        if action == "CRAFT_RECIPE":
            recipe = RECIPES[normalize_identifier(str(parameters["recipe"]))]
            for item, quantity in recipe["ingredients"].items():
                remove_item(after, item, quantity)
            add_item(after, recipe["output"], recipe["output_count"])
            after["craft_item"][recipe["output"]] = recipe["output_count"]
        elif action == "EQUIP_ITEM":
            after["equipped_items"]["mainhand"] = {"type": str(parameters["item"]), "quantity": 1}
        elif action in {"PLACE_BLOCK", "PLACE_STATION"}:
            item = str(parameters.get("item", parameters.get("station")))
            remove_item(after, item, 1)
            after["place_block"][normalize_identifier(item)] = 1
        elif action == "RECLAIM_BLOCK":
            add_item(after, str(parameters["item"]), 1)
        elif action == "MINE_AND_COLLECT":
            add_item(after, str(parameters["drop_item"]), int(parameters["count"]))
            after["mine_block"][normalize_identifier(str(parameters["target_type"]))] = int(parameters["count"])
        elif action == "COLLECT_KNOWN_DROP":
            add_item(after, str(parameters["item"]), 1)
        return BackendResult(SystemZeroStatus.SUCCESS, after, details={"reason": "fake execution"}, steps=1)


class SystemZeroTests(TestCase):
    def setUp(self):
        self.registry = build_priority_one_registry()

    def valid_cases(self):
        return {
            "CRAFT_RECIPE": (
                SystemZeroRequest("CRAFT_RECIPE", {"recipe": "stone_pickaxe"}),
                observation_with({"cobblestone": 3, "stick": 2, "crafting_table": 1}),
            ),
            "EQUIP_ITEM": (
                SystemZeroRequest("EQUIP_ITEM", {"item": "stone_pickaxe"}),
                observation_with({"stone_pickaxe": 1}),
            ),
            "PLACE_BLOCK": (
                SystemZeroRequest("PLACE_BLOCK", {"item": "cobblestone"}),
                observation_with({"cobblestone": 1}),
            ),
            "RECLAIM_BLOCK": (
                SystemZeroRequest("RECLAIM_BLOCK", {"item": "crafting_table", "target_known": True}),
                observation_with(),
            ),
            "MINE_AND_COLLECT": (
                SystemZeroRequest("MINE_AND_COLLECT", {"target_type": "stone", "drop_item": "cobblestone", "target_known": True}),
                observation_with(),
            ),
            "COLLECT_KNOWN_DROP": (
                SystemZeroRequest("COLLECT_KNOWN_DROP", {"item": "cobblestone", "target_known": True}),
                observation_with(),
            ),
            "PLACE_STATION": (
                SystemZeroRequest("PLACE_STATION", {"station": "crafting_table"}),
                observation_with({"crafting_table": 1}),
            ),
        }

    def test_registry_exposes_complete_skill_contracts(self):
        skills = self.registry.list_skills()
        self.assertEqual({skill["name"] for skill in skills}, set(self.valid_cases()))
        for skill in skills:
            self.assertTrue(skill["parameters"])
            self.assertTrue(skill["preconditions"])
            self.assertTrue(skill["execution"])
            self.assertTrue(skill["success_predicate"])
            self.assertTrue(skill["failure_predicates"])
            self.assertGreater(skill["timeout_steps"], 0)
            self.assertTrue(skill["abort_conditions"])

    def test_every_skill_succeeds_and_is_verified(self):
        for name, (request, observation) in self.valid_cases().items():
            with self.subTest(skill=name):
                backend = FakeBackend()
                result = self.registry.execute(request, SystemZeroContext(observation, backend))
                self.assertEqual(result.status, SystemZeroStatus.SUCCESS)
                self.assertTrue(result.details["verified"])
                self.assertEqual(len(backend.calls), 1)

    def test_every_skill_rejects_missing_or_uncertain_preconditions(self):
        requests = {
            "CRAFT_RECIPE": SystemZeroRequest("CRAFT_RECIPE", {"recipe": "stone_pickaxe"}),
            "EQUIP_ITEM": SystemZeroRequest("EQUIP_ITEM", {"item": "stone_pickaxe"}),
            "PLACE_BLOCK": SystemZeroRequest("PLACE_BLOCK", {"item": "cobblestone"}),
            "RECLAIM_BLOCK": SystemZeroRequest("RECLAIM_BLOCK", {"item": "crafting_table", "target_known": False}),
            "MINE_AND_COLLECT": SystemZeroRequest("MINE_AND_COLLECT", {"target_type": "stone", "drop_item": "cobblestone", "target_known": False}),
            "COLLECT_KNOWN_DROP": SystemZeroRequest("COLLECT_KNOWN_DROP", {"item": "cobblestone", "target_known": False}),
            "PLACE_STATION": SystemZeroRequest("PLACE_STATION", {"station": "crafting_table"}),
        }
        for name, request in requests.items():
            with self.subTest(skill=name):
                backend = FakeBackend()
                result = self.registry.execute(request, SystemZeroContext(observation_with(), backend))
                expected = SystemZeroStatus.NEEDS_SYSTEM1 if name in {"RECLAIM_BLOCK", "MINE_AND_COLLECT", "COLLECT_KNOWN_DROP"} else SystemZeroStatus.PRECONDITION_FAILED
                self.assertEqual(result.status, expected)
                self.assertFalse(backend.calls)

    def test_every_skill_enforces_its_verifier(self):
        for name, (request, observation) in self.valid_cases().items():
            with self.subTest(skill=name):
                result = self.registry.execute(request, SystemZeroContext(observation, FakeBackend("no_effect")))
                self.assertEqual(result.status, SystemZeroStatus.FAILURE)
                self.assertFalse(result.details["verified"])

    def test_every_skill_enforces_timeout(self):
        for name, (request, observation) in self.valid_cases().items():
            with self.subTest(skill=name):
                result = self.registry.execute(request, SystemZeroContext(observation, FakeBackend("timeout")))
                self.assertEqual(result.status, SystemZeroStatus.TIMEOUT)

    def test_every_skill_honors_pre_execution_interruption(self):
        for name, (request, observation) in self.valid_cases().items():
            with self.subTest(skill=name):
                backend = FakeBackend()
                result = self.registry.execute(request, SystemZeroContext(observation, backend, should_abort=lambda: True))
                self.assertEqual(result.status, SystemZeroStatus.FAILURE)
                self.assertIn("interrupted", result.details["reason"])
                self.assertFalse(backend.calls)

    def test_equip_is_idempotent(self):
        observation = observation_with(equipped="stone_pickaxe")
        backend = FakeBackend()
        result = self.registry.execute(SystemZeroRequest("EQUIP_ITEM", {"item": "stone_pickaxe"}), SystemZeroContext(observation, backend))
        self.assertEqual(result.status, SystemZeroStatus.SUCCESS)
        self.assertTrue(result.details["idempotent"])
        self.assertFalse(backend.calls)

    def test_router_hands_uncertain_execution_to_system_one(self):
        router = ExecutionRouter(self.registry)
        request = SystemZeroRequest("MINE_AND_COLLECT", {"target_type": "stone", "drop_item": "cobblestone", "target_known": False})
        routed = router.route(request, SystemZeroContext(observation_with(), FakeBackend()))
        self.assertEqual(routed.route, ExecutionRoute.SYSTEM_ONE)
        self.assertIsNone(routed.result)

    def test_router_preserves_deterministic_precondition_failure(self):
        router = ExecutionRouter(self.registry)
        request = SystemZeroRequest("CRAFT_RECIPE", {"recipe": "stone_pickaxe"})
        routed = router.route(request, SystemZeroContext(observation_with(), FakeBackend()))
        self.assertEqual(routed.route, ExecutionRoute.SYSTEM_ZERO)
        self.assertEqual(routed.result.status, SystemZeroStatus.PRECONDITION_FAILED)

    def test_router_hands_unknown_skills_to_system_one(self):
        router = ExecutionRouter(self.registry)
        routed = router.route(SystemZeroRequest("FIND_NEAREST_IRON"), SystemZeroContext(observation_with(), FakeBackend()))
        self.assertEqual(routed.route, ExecutionRoute.SYSTEM_ONE)

    def test_backend_request_for_system_one_is_propagated(self):
        router = ExecutionRouter(self.registry)
        request, observation = self.valid_cases()["MINE_AND_COLLECT"]
        routed = router.route(request, SystemZeroContext(observation, FakeBackend("needs_system1")))
        self.assertEqual(routed.route, ExecutionRoute.SYSTEM_ONE)
        self.assertEqual(routed.result.status, SystemZeroStatus.NEEDS_SYSTEM1)

    def test_inventory_mutation_is_reflected_in_context(self):
        request, observation = self.valid_cases()["CRAFT_RECIPE"]
        context = SystemZeroContext(observation, FakeBackend())
        result = self.registry.execute(request, context)
        self.assertEqual(result.status, SystemZeroStatus.SUCCESS)
        self.assertEqual(inventory_counts(context.observation)["stone_pickaxe"], 1)
