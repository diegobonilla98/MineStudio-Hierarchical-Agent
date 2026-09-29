from collections.abc import Mapping
from typing import Any

from minestudio.system_zero.contracts import ParameterSpec, PreconditionResult, SystemZeroSkill, SystemZeroStatus
from minestudio.system_zero.inventory import count_item, equipped_item, event_count, item_matches, normalize_identifier
from minestudio.system_zero.registry import SystemZeroRegistry


RECIPES = {
    "planks": {"ingredients": {"log": 1}, "output": "planks", "output_count": 4, "station": None},
    "stick": {"ingredients": {"planks": 2}, "output": "stick", "output_count": 4, "station": None},
    "crafting_table": {"ingredients": {"planks": 4}, "output": "crafting_table", "output_count": 1, "station": None},
    "wooden_pickaxe": {"ingredients": {"planks": 3, "stick": 2}, "output": "wooden_pickaxe", "output_count": 1, "station": "crafting_table"},
    "stone_pickaxe": {"ingredients": {"cobblestone": 3, "stick": 2}, "output": "stone_pickaxe", "output_count": 1, "station": "crafting_table"},
    "furnace": {"ingredients": {"cobblestone": 8}, "output": "furnace", "output_count": 1, "station": "crafting_table"},
    "iron_pickaxe": {"ingredients": {"iron_ingot": 3, "stick": 2}, "output": "iron_pickaxe", "output_count": 1, "station": "crafting_table"},
}


def pass_preconditions(parameters: Mapping[str, Any], observation: Mapping[str, Any]) -> PreconditionResult:
    return PreconditionResult()


def require_inventory_item(parameter_name: str):
    def check(parameters: Mapping[str, Any], observation: Mapping[str, Any]) -> PreconditionResult:
        item = str(parameters[parameter_name])
        if count_item(observation, item) > 0:
            return PreconditionResult()
        return PreconditionResult(
            status=SystemZeroStatus.PRECONDITION_FAILED,
            details={"reason": "item missing", "missing": {f"minecraft:{normalize_identifier(item)}": 1}},
        )

    return check


def equip_precondition(parameters: Mapping[str, Any], observation: Mapping[str, Any]) -> PreconditionResult:
    item = str(parameters["item"])
    if item_matches(equipped_item(observation), item) or count_item(observation, item) > 0:
        return PreconditionResult()
    return PreconditionResult(
        status=SystemZeroStatus.PRECONDITION_FAILED,
        details={"reason": "item missing", "missing": {f"minecraft:{normalize_identifier(item)}": 1}},
    )


def known_target_precondition(parameters: Mapping[str, Any], observation: Mapping[str, Any]) -> PreconditionResult:
    if not bool(parameters["target_known"]):
        return PreconditionResult(
            status=SystemZeroStatus.NEEDS_SYSTEM1,
            details={"reason": "target must be selected and grounded before deterministic execution"},
        )
    if not bool(parameters.get("target_reachable", True)):
        return PreconditionResult(
            status=SystemZeroStatus.NEEDS_SYSTEM1,
            details={"reason": "known target is not currently reachable"},
        )
    return PreconditionResult()


def mine_precondition(parameters: Mapping[str, Any], observation: Mapping[str, Any]) -> PreconditionResult:
    target = known_target_precondition(parameters, observation)
    if not target.satisfied:
        return target
    if int(parameters["count"]) <= 0:
        return PreconditionResult(
            status=SystemZeroStatus.PRECONDITION_FAILED,
            details={"reason": "count must be positive", "count": int(parameters["count"])},
        )
    tool = parameters.get("tool")
    if tool is None or item_matches(equipped_item(observation), str(tool)) or count_item(observation, str(tool)) > 0:
        return PreconditionResult()
    return PreconditionResult(
        status=SystemZeroStatus.PRECONDITION_FAILED,
        details={"reason": "required tool missing", "missing": {f"minecraft:{normalize_identifier(str(tool))}": 1}},
    )


def craft_precondition(parameters: Mapping[str, Any], observation: Mapping[str, Any]) -> PreconditionResult:
    recipe_name = normalize_identifier(str(parameters["recipe"]))
    if recipe_name not in RECIPES:
        return PreconditionResult(
            status=SystemZeroStatus.PRECONDITION_FAILED,
            details={"reason": "unsupported recipe", "recipe": recipe_name},
        )
    recipe = RECIPES[recipe_name]
    missing = {}
    for item, required in recipe["ingredients"].items():
        available = count_item(observation, item)
        if available < required:
            missing[f"minecraft:{item}"] = required - available
    station = recipe["station"]
    station_accessible = bool(parameters.get("station_accessible", False)) or bool(observation.get("is_gui_open", False))
    if station is not None and count_item(observation, station) < 1 and not station_accessible:
        missing[f"minecraft:{station}"] = 1
    if missing:
        return PreconditionResult(
            status=SystemZeroStatus.PRECONDITION_FAILED,
            details={"reason": "crafting preconditions missing", "missing": missing},
        )
    return PreconditionResult()


def craft_verifier(parameters: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    recipe = RECIPES[normalize_identifier(str(parameters["recipe"]))]
    output = recipe["output"]
    inventory_delta = count_item(after, output) - count_item(before, output)
    crafted_delta = event_count(after, "craft_item", output) - event_count(before, "craft_item", output)
    verified = inventory_delta >= recipe["output_count"] or crafted_delta >= recipe["output_count"]
    return verified, {"inventory_delta": inventory_delta, "crafted_delta": crafted_delta, "expected_delta": recipe["output_count"]}


def equip_verifier(parameters: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    equipped = equipped_item(after)
    verified = item_matches(equipped, str(parameters["item"]))
    return verified, {"equipped": equipped}


def equip_idempotent(parameters: Mapping[str, Any], observation: Mapping[str, Any]) -> bool:
    return item_matches(equipped_item(observation), str(parameters["item"]))


def placement_verifier(parameters: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    item = str(parameters.get("item", parameters.get("station")))
    inventory_delta = count_item(before, item) - count_item(after, item)
    placed_delta = event_count(after, "place_block", item) - event_count(before, "place_block", item)
    verified = inventory_delta >= 1 or placed_delta >= 1
    return verified, {"inventory_consumed": inventory_delta, "placed_delta": placed_delta}


def reclaim_verifier(parameters: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    item = str(parameters["item"])
    inventory_delta = count_item(after, item) - count_item(before, item)
    verified = inventory_delta >= 1
    return verified, {"inventory_delta": inventory_delta}


def mine_collect_verifier(parameters: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    drop_item = str(parameters["drop_item"])
    expected = int(parameters["count"])
    inventory_delta = count_item(after, drop_item) - count_item(before, drop_item)
    verified = inventory_delta >= expected
    return verified, {"inventory_delta": inventory_delta, "expected_delta": expected}


def collect_drop_verifier(parameters: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    item = str(parameters["item"])
    inventory_delta = count_item(after, item) - count_item(before, item)
    verified = inventory_delta >= 1
    return verified, {"inventory_delta": inventory_delta, "expected_delta": 1}


def build_priority_one_registry() -> SystemZeroRegistry:
    registry = SystemZeroRegistry()
    common_aborts = ("environment ended", "unexpected damage", "health danger", "explicit cancellation")
    registry.register(SystemZeroSkill(
        name="CRAFT_RECIPE",
        description="Execute one supported recipe with known ingredients and an accessible station when required.",
        parameters=(
            ParameterSpec("recipe", "string", "Recipe output identifier."),
            ParameterSpec("station_accessible", "boolean", "Whether a required placed station is already known and reachable.", required=False, default=False),
        ),
        preconditions=("supported recipe", "required ingredients in inventory", "required station available or accessible"),
        execution="CRAFT_RECIPE",
        success_predicate="inventory or crafted-item delta reaches the recipe output quantity",
        failure_predicates=("missing ingredient", "station unavailable", "GUI transaction failure", "output delta missing"),
        timeout_steps=220,
        abort_conditions=common_aborts,
        check_preconditions=craft_precondition,
        verify=craft_verifier,
    ))
    registry.register(SystemZeroSkill(
        name="EQUIP_ITEM",
        description="Move a known inventory item to the hotbar if necessary and select it.",
        parameters=(ParameterSpec("item", "string", "Item identifier or supported item group."),),
        preconditions=("item is already equipped or present in inventory",),
        execution="EQUIP_ITEM",
        success_predicate="requested item is verified in the main hand",
        failure_predicates=("item missing", "inventory transaction failure", "main-hand verification failure"),
        timeout_steps=80,
        abort_conditions=common_aborts,
        check_preconditions=equip_precondition,
        verify=equip_verifier,
        check_idempotent=equip_idempotent,
    ))
    registry.register(SystemZeroSkill(
        name="PLACE_BLOCK",
        description="Place a known inventory block using bounded local surface recovery.",
        parameters=(ParameterSpec("item", "string", "Block item to place."),),
        preconditions=("block item present in inventory", "local placement remains mechanically bounded"),
        execution="PLACE_BLOCK",
        success_predicate="inventory consumption or block-placement event is verified",
        failure_predicates=("item missing", "no valid local surface", "placement not verified"),
        timeout_steps=180,
        abort_conditions=common_aborts + ("unexpected fall", "water ingress"),
        check_preconditions=require_inventory_item("item"),
        verify=placement_verifier,
    ))
    registry.register(SystemZeroSkill(
        name="RECLAIM_BLOCK",
        description="Break and collect a placed block whose target is already known.",
        parameters=(
            ParameterSpec("item", "string", "Expected recovered item."),
            ParameterSpec("target_known", "boolean", "Whether System One or System Two supplied the target.", required=False, default=True),
            ParameterSpec("target_reachable", "boolean", "Whether the supplied target is centered and reachable.", required=False, default=True),
        ),
        preconditions=("target supplied", "target centered and reachable"),
        execution="RECLAIM_BLOCK",
        success_predicate="expected item inventory delta is at least one",
        failure_predicates=("target lost", "block not broken", "drop not collected"),
        timeout_steps=180,
        abort_conditions=common_aborts,
        check_preconditions=known_target_precondition,
        verify=reclaim_verifier,
    ))
    registry.register(SystemZeroSkill(
        name="MINE_AND_COLLECT",
        description="Break one already selected, centered, reachable block and collect its known drop.",
        parameters=(
            ParameterSpec("target_type", "string", "Known block type to break."),
            ParameterSpec("drop_item", "string", "Expected dropped inventory item."),
            ParameterSpec("count", "integer", "Required collected quantity.", required=False, default=1),
            ParameterSpec("tool", "string", "Known tool to equip before mining, if required.", required=False, default=None),
            ParameterSpec("target_known", "boolean", "Whether System One or System Two supplied the target.", required=False, default=True),
            ParameterSpec("target_reachable", "boolean", "Whether the supplied target is centered and reachable.", required=False, default=True),
        ),
        preconditions=("target supplied", "target centered and reachable", "drop identity known"),
        execution="MINE_AND_COLLECT",
        success_predicate="expected drop inventory delta reaches count",
        failure_predicates=("target lost", "block not broken", "drop not collected", "step budget exceeded"),
        timeout_steps=260,
        abort_conditions=common_aborts + ("tool breaks",),
        check_preconditions=mine_precondition,
        verify=mine_collect_verifier,
    ))
    registry.register(SystemZeroSkill(
        name="COLLECT_KNOWN_DROP",
        description="Collect a recently created known drop with bounded local movement and hand uncertain geometry back to System One.",
        parameters=(
            ParameterSpec("item", "string", "Expected dropped inventory item."),
            ParameterSpec("target_known", "boolean", "Whether the drop location is locally known from the preceding interaction.", required=False, default=True),
            ParameterSpec("target_reachable", "boolean", "Whether pickup geometry is currently bounded and locally reachable.", required=False, default=True),
        ),
        preconditions=("drop identity known", "drop was created by the preceding interaction", "pickup remains locally bounded"),
        execution="COLLECT_KNOWN_DROP",
        success_predicate="expected item inventory delta is at least one",
        failure_predicates=("inventory delta missing", "pickup geometry uncertain", "step budget exceeded"),
        timeout_steps=120,
        abort_conditions=common_aborts + ("unexpected fall", "water ingress"),
        check_preconditions=known_target_precondition,
        verify=collect_drop_verifier,
    ))
    registry.register(SystemZeroSkill(
        name="PLACE_STATION",
        description="Place a known workstation from inventory using the bounded placement controller.",
        parameters=(ParameterSpec("station", "string", "Station item to place."),),
        preconditions=("station item present in inventory", "local placement remains mechanically bounded"),
        execution="PLACE_STATION",
        success_predicate="station inventory consumption or placement event is verified",
        failure_predicates=("station missing", "no valid local surface", "placement not verified"),
        timeout_steps=180,
        abort_conditions=common_aborts + ("unexpected fall", "water ingress"),
        check_preconditions=require_inventory_item("station"),
        verify=placement_verifier,
    ))
    return registry
