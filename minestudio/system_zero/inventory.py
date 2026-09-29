from collections.abc import Mapping
from typing import Any


ITEM_SUFFIXES = {
    "log": ("_log", "_wood", "_stem", "_hyphae", "hyphae"),
    "planks": ("_planks",),
}


def normalize_identifier(value: str) -> str:
    return value.lower().replace("minecraft:", "").replace(" ", "_")


def item_matches(item_name: str, target: str) -> bool:
    item_name = normalize_identifier(item_name)
    target = normalize_identifier(target)
    suffixes = ITEM_SUFFIXES.get(target)
    return item_name.endswith(suffixes) if suffixes is not None else item_name == target


def inventory_counts(observation: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for stack in observation.get("inventory", {}).values():
        item_name = normalize_identifier(str(stack.get("type", "air")))
        quantity = int(float(stack.get("quantity", 0)))
        if item_name not in {"air", "none"} and quantity > 0:
            counts[item_name] = counts.get(item_name, 0) + quantity
    return counts


def count_item(observation: Mapping[str, Any], target: str) -> int:
    return sum(quantity for name, quantity in inventory_counts(observation).items() if item_matches(name, target))


def equipped_item(observation: Mapping[str, Any], slot: str = "mainhand") -> str:
    item = observation.get("equipped_items", {}).get(slot, {})
    return normalize_identifier(str(item.get("type", "air")))


def event_count(observation: Mapping[str, Any], source: str, target: str) -> float:
    values = observation.get(source, {})
    return sum(
        float(value)
        for name, value in values.items()
        if item_matches(str(name), target)
    )
