from copy import deepcopy
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np
from minestudio.simulator import MinecraftSim


CURSOR_START = (320.0, 180.0)
PIXELS_PER_CAMERA_DEGREE = 6.6
MAX_CAMERA_REQUEST = 6.0
PERSONAL_GRID = [[(338.0, 123.0), (356.0, 123.0)], [(338.0, 141.0), (356.0, 141.0)]]
PERSONAL_OUTPUT = (393.0, 133.0)
TABLE_GRID = [
    [(271.0, 124.0), (289.0, 124.0), (307.0, 124.0)],
    [(271.0, 142.0), (289.0, 142.0), (307.0, 142.0)],
    [(271.0, 160.0), (289.0, 160.0), (307.0, 160.0)],
]
TABLE_OUTPUT = (365.0, 142.0)
FURNACE_INPUT = (295.0, 121.0)
FURNACE_FUEL = (295.0, 157.0)
FURNACE_OUTPUT = (351.0, 143.0)
CRAFT_RECIPES = {
    "planks": {
        "table": False,
        "ingredients": [("log", [(0, 0)], "left")],
        "output": "planks",
        "output_per_click": 4,
        "preferred_quantity": 12,
    },
    "stick": {
        "table": False,
        "ingredients": [("planks", [(0, 0), (1, 0)], "right")],
        "output": "stick",
        "output_per_click": 4,
        "preferred_quantity": 4,
    },
    "crafting_table": {
        "table": False,
        "ingredients": [("planks", [(0, 0), (0, 1), (1, 0), (1, 1)], "right")],
        "output": "crafting_table",
        "output_per_click": 1,
        "preferred_quantity": 1,
    },
    "wooden_pickaxe": {
        "table": True,
        "ingredients": [
            ("planks", [(0, 0), (0, 1), (0, 2)], "right"),
            ("stick", [(1, 1), (2, 1)], "right"),
        ],
        "output": "wooden_pickaxe",
        "output_per_click": 1,
        "preferred_quantity": 1,
    },
    "stone_pickaxe": {
        "table": True,
        "ingredients": [
            ("cobblestone", [(0, 0), (0, 1), (0, 2)], "right"),
            ("stick", [(1, 1), (2, 1)], "right"),
        ],
        "output": "stone_pickaxe",
        "output_per_click": 1,
        "preferred_quantity": 1,
    },
    "furnace": {
        "table": True,
        "ingredients": [
            ("cobblestone", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)], "right"),
        ],
        "output": "furnace",
        "output_per_click": 1,
        "preferred_quantity": 1,
    },
    "iron_pickaxe": {
        "table": True,
        "ingredients": [
            ("iron_ingot", [(0, 0), (0, 1), (0, 2)], "right"),
            ("stick", [(1, 1), (2, 1)], "right"),
        ],
        "output": "iron_pickaxe",
        "output_per_click": 1,
        "preferred_quantity": 1,
    },
}
ITEM_SUFFIXES = {
    "log": ("_log", "_wood", "_stem", "_hyphae", "hyphae"),
    "planks": ("_planks",),
}
WORLD_CONTROL_KEYS = {"forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "camera"}
PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CURSOR_IMAGE_PATH = PROJECT_DIRECTORY / "minestudio" / "data" / "minecraft" / "tools" / "cursors" / "mouse_cursor_white_16x16.png"
CURSOR_IMAGE = cv2.imread(str(CURSOR_IMAGE_PATH), cv2.IMREAD_UNCHANGED)[:16, :16]
CURSOR_TEMPLATE = CURSOR_IMAGE[:, :, :3]
CURSOR_MASK = CURSOR_IMAGE[:, :, 3]


def normalize_identifier(value: str) -> str:
    return value.lower().replace("minecraft:", "").replace(" ", "_")


def item_matches(item_name: str, target: str) -> bool:
    item_name = normalize_identifier(item_name)
    target = normalize_identifier(target)
    suffixes = ITEM_SUFFIXES.get(target)
    return item_name.endswith(suffixes) if suffixes is not None else item_name == target


def inventory_counts(info: dict) -> dict[str, float]:
    counts: dict[str, float] = {}
    for stack in info.get("inventory", {}).values():
        item_name = normalize_identifier(str(stack.get("type", "air")))
        quantity = float(stack.get("quantity", 0))
        if item_name not in {"air", "none"} and quantity > 0:
            counts[item_name] = counts.get(item_name, 0.0) + quantity
    return counts


def grouped_inventory_value(info: dict, target: str) -> float:
    return sum(quantity for name, quantity in inventory_counts(info).items() if item_matches(name, target))


def player_position(info: dict) -> np.ndarray:
    position = info.get("player_pos", {})
    return np.array([
        float(position.get("x", 0.0)),
        float(position.get("y", 0.0)),
        float(position.get("z", 0.0)),
    ], dtype=np.float32)


def voxel_records(value) -> list[dict]:
    records = []
    if isinstance(value, dict):
        if {"x", "y", "z", "type"}.issubset(value):
            records.append(value)
        else:
            for item in value.values():
                records.extend(voxel_records(item))
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in value:
            records.extend(voxel_records(item))
    return records


def player_in_water(info: dict) -> bool:
    for record in voxel_records(info.get("voxels")):
        block_type = normalize_identifier(str(record.get("type", "")))
        if (
            (block_type.endswith("water") or block_type.endswith("bubble_column"))
            and int(float(record.get("x", 99))) == 0
            and int(float(record.get("z", 99))) == 0
            and int(float(record.get("y", 99))) in {0, 1}
        ):
            return True
    return False


def detect_cursor(info: dict) -> tuple[float, float]:
    frame = cv2.cvtColor(np.asarray(info["pov"]), cv2.COLOR_RGB2BGR)
    scores = cv2.matchTemplate(frame, CURSOR_TEMPLATE, cv2.TM_CCORR_NORMED, mask=CURSOR_MASK)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    minimum, maximum, location, maximum_location = cv2.minMaxLoc(scores)
    if maximum < 0.75:
        raise RuntimeError(f"GUI cursor not detected: score={maximum}")
    return float(maximum_location[0]), float(maximum_location[1])


class ClosedLoopOptions:
    def __init__(self, simulator: MinecraftSim, step_function: Callable | None = None, pickup_implementation: str = "hardened"):
        if pickup_implementation not in {"old", "hardened"}:
            raise ValueError(f"Unknown pickup implementation: {pickup_implementation}")
        self.simulator = simulator
        self.step_function = step_function or self._default_step
        self.pickup_implementation = pickup_implementation
        self.steps = 0

    def _default_step(self, action: dict, phase: str):
        return self.simulator.step(action)

    def environment_action(self, values: dict) -> dict:
        action = self.simulator.env.action_space.no_op()
        action.update(values)
        action = {
            key: np.asarray(value, dtype=np.float32).reshape(1, 2)
            if key == "camera"
            else np.asarray([value])
            for key, value in action.items()
            if key != "chat"
        }
        return self.simulator.env_action_to_agent_action(action)

    def gate_world_control(self, action: dict) -> tuple[dict, dict, list[str]]:
        environment_action = self.simulator.agent_action_to_env_action(deepcopy(action))
        blocked_controls = []
        for key, value in environment_action.items():
            if key not in WORLD_CONTROL_KEYS and np.any(np.asarray(value)):
                blocked_controls.append(key)
            if key not in WORLD_CONTROL_KEYS:
                environment_action[key] = np.zeros_like(value)
        batched_action = {
            key: np.asarray(value).reshape(1, 2) if key == "camera" else np.asarray(value).reshape(1)
            for key, value in environment_action.items()
            if key != "chat"
        }
        return self.simulator.env_action_to_agent_action(batched_action), environment_action, blocked_controls

    def step_values(self, values: dict, phase: str):
        action = self.environment_action(values)
        observation, reward, terminated, truncated, info = self.step_function(action, phase)
        self.steps += 1
        if terminated or truncated or "error" in info:
            raise RuntimeError(f"Environment ended during {phase}: {info.get('error')}")
        return observation, info

    def move_cursor(self, info: dict, current: tuple[float, float], target: tuple[float, float], phase: str):
        for _ in range(12):
            cursor_x, cursor_y = detect_cursor(info)
            if abs(target[0] - cursor_x) <= 6 and abs(target[1] - cursor_y) <= 6:
                break
            yaw = np.clip((target[0] - cursor_x) / PIXELS_PER_CAMERA_DEGREE, -MAX_CAMERA_REQUEST, MAX_CAMERA_REQUEST)
            pitch = np.clip((target[1] - cursor_y) / PIXELS_PER_CAMERA_DEGREE, -MAX_CAMERA_REQUEST, MAX_CAMERA_REQUEST)
            action = self.environment_action({"camera": np.array([pitch, yaw], dtype=np.float32)})
            actual_camera = np.asarray(self.simulator.agent_action_to_env_action(deepcopy(action))["camera"]).reshape(-1, 2)[0]
            observation, reward, terminated, truncated, info = self.step_function(action, phase)
            self.steps += 1
            if terminated or truncated or "error" in info:
                raise RuntimeError(f"Environment ended during {phase}: {info.get('error')}")
            if not np.any(actual_camera):
                break
        return info, detect_cursor(info)

    def click(self, phase: str, right: bool = False):
        button = "use" if right else "attack"
        observation, info = self.step_values({button: 1}, phase)
        return self.step_values({button: 0}, phase)

    def inventory_slot(self, info: dict, target: str) -> int | None:
        for slot, stack in info.get("inventory", {}).items():
            if float(stack.get("quantity", 0)) > 0 and item_matches(str(stack.get("type", "none")), target):
                return int(slot)
        return None

    def empty_slot(self, info: dict) -> int | None:
        slots = info.get("inventory", {})
        for slot in list(range(9)) + list(range(9, 36)):
            stack = slots.get(slot, slots.get(str(slot), {}))
            if float(stack.get("quantity", 0)) <= 0 or normalize_identifier(str(stack.get("type", "none"))) in {"air", "none"}:
                return slot
        return None

    def slot_coordinate(self, slot: int) -> tuple[float, float]:
        if slot < 9:
            return 247.0 + 18.0 * slot, 248.0
        inventory_index = slot - 9
        return 247.0 + 18.0 * (inventory_index % 9), 190.0 + 18.0 * (inventory_index // 9)

    def open_inventory(self, info: dict):
        if not bool(info.get("is_gui_open", False)):
            observation, info = self.step_values({"inventory": 1}, "CRAFT_RECIPE/open_inventory")
            observation, info = self.step_values({"inventory": 0}, "CRAFT_RECIPE/open_inventory")
        if not bool(info.get("is_gui_open", False)):
            raise RuntimeError("Inventory GUI did not open")
        return info, detect_cursor(info)

    def close_gui(self, info: dict):
        if bool(info.get("is_gui_open", False)):
            observation, info = self.step_values({"inventory": 1}, "CRAFT_RECIPE/close_gui")
            observation, info = self.step_values({"inventory": 0}, "CRAFT_RECIPE/close_gui")
        return info

    def open_crafting_table(self, info: dict):
        for reposition in range(3):
            for pitch in (70.0, 55.0, 40.0, 25.0):
                info = self.set_pitch(pitch, info)
                observation, info = self.step_values({"use": 1}, "CRAFT_RECIPE/open_table")
                observation, info = self.step_values({"use": 0}, "CRAFT_RECIPE/open_table")
                if bool(info.get("is_gui_open", False)):
                    return info
            observation, info = self.step_values({"back": 1}, "CRAFT_RECIPE/find_table")
            observation, info = self.step_values({"back": 0}, "CRAFT_RECIPE/find_table")
        return info

    def open_station(self, station_name: str, info: dict):
        phase = f"USE_STATION/{normalize_identifier(station_name)}"
        for reposition in range(4):
            for pitch in (70.0, 55.0, 40.0, 25.0):
                info = self.set_pitch(pitch, info)
                observation, info = self.step_values({"use": 1}, phase)
                observation, info = self.step_values({"use": 0}, phase)
                if bool(info.get("is_gui_open", False)):
                    return info
            movement = "back" if reposition % 2 == 0 else "forward"
            for _ in range(6):
                observation, info = self.step_values({movement: 1}, f"{phase}/reposition")
            observation, info = self.step_values({movement: 0}, f"{phase}/reposition")
        return info

    def move_and_click(self, info: dict, cursor: tuple[float, float], target: tuple[float, float], phase: str, right: bool = False):
        info, cursor = self.move_cursor(info, cursor, target, phase)
        observation, info = self.click(phase, right=right)
        return info, detect_cursor(info)

    def shift_click(self, info: dict, cursor: tuple[float, float], target: tuple[float, float], phase: str):
        info, cursor = self.move_cursor(info, cursor, target, phase)
        observation, info = self.step_values({"sneak": 1}, phase)
        observation, info = self.step_values({"sneak": 1, "attack": 1}, phase)
        observation, info = self.step_values({"sneak": 1, "attack": 0}, phase)
        observation, info = self.step_values({"sneak": 0}, phase)
        return info, detect_cursor(info)

    def hotbar_swap(self, info: dict, cursor: tuple[float, float], target: tuple[float, float], slot: int, phase: str):
        if slot < 0 or slot > 8:
            raise ValueError(f"Hotbar swap requires slot 0-8, got {slot}")
        info, cursor = self.move_cursor(info, cursor, target, phase)
        key = f"hotbar.{slot + 1}"
        observation, info = self.step_values({key: 1}, phase)
        observation, info = self.step_values({key: 0}, phase)
        return info, detect_cursor(info)

    def craft_recipe(self, recipe_name: str, info: dict) -> tuple[dict, dict]:
        recipe = CRAFT_RECIPES[normalize_identifier(recipe_name)]
        baseline_output = grouped_inventory_value(info, recipe["output"])
        start_steps = self.steps
        station_records = []
        ingredient_slots = []
        for ingredient, cells, click_kind in recipe["ingredients"]:
            slot = self.inventory_slot(info, ingredient)
            if slot is None:
                return {"option": "CRAFT_RECIPE", "recipe": recipe_name, "success": False, "reason": f"missing {ingredient}", "steps": 0}, info
            required = len(cells) if click_kind == "right" else 1
            if grouped_inventory_value(info, ingredient) < required:
                return {"option": "CRAFT_RECIPE", "recipe": recipe_name, "success": False, "reason": f"missing {ingredient}: need {required}", "steps": 0}, info
            ingredient_slots.append(slot)
        destination = self.inventory_slot(info, recipe["output"])
        if destination is None:
            destination = ingredient_slots[0] if normalize_identifier(recipe_name) == "planks" else self.empty_slot(info)
        if recipe["table"] and not bool(info.get("is_gui_open", False)) and grouped_inventory_value(info, "crafting_table") >= 1:
            station_result, info = self.place_block("crafting_table", info)
            station_records.append(station_result)
        info, cursor = self.open_inventory(info) if not recipe["table"] else (info, CURSOR_START)
        if recipe["table"] and not bool(info.get("is_gui_open", False)):
            info = self.open_crafting_table(info)
            cursor = detect_cursor(info) if bool(info.get("is_gui_open", False)) else CURSOR_START
        if not bool(info.get("is_gui_open", False)):
            return {"option": "CRAFT_RECIPE", "recipe": recipe_name, "success": False, "reason": "crafting GUI unavailable", "steps": self.steps - start_steps}, info
        grid = TABLE_GRID if recipe["table"] else PERSONAL_GRID
        output_coordinate = TABLE_OUTPUT if recipe["table"] else PERSONAL_OUTPUT
        if recipe["table"]:
            info, cursor = self.move_and_click(info, cursor, (395.0, 228.0), "CRAFT_RECIPE/focus_table")
        for ingredient_index, (ingredient, cells, click_kind) in enumerate(recipe["ingredients"]):
            slot = ingredient_slots[ingredient_index]
            source_coordinate = self.slot_coordinate(slot)
            info, cursor = self.move_and_click(info, cursor, source_coordinate, "CRAFT_RECIPE/pick_ingredient")
            for row, column in cells:
                info, cursor = self.move_and_click(
                    info,
                    cursor,
                    grid[row][column],
                    "CRAFT_RECIPE/fill_grid",
                    right=click_kind == "right",
                )
            if click_kind == "right":
                info, cursor = self.move_and_click(info, cursor, source_coordinate, "CRAFT_RECIPE/return_remainder")
        output_clicks = max(1, int(np.ceil(recipe["preferred_quantity"] / recipe["output_per_click"])))
        if normalize_identifier(recipe_name) == "planks":
            output_clicks += 2
        for _ in range(output_clicks):
            info, cursor = self.move_and_click(info, cursor, output_coordinate, "CRAFT_RECIPE/take_output")
            observation, info = self.step_values({}, "CRAFT_RECIPE/settle_output")
        if destination is None:
            info = self.close_gui(info)
            return {"option": "CRAFT_RECIPE", "recipe": recipe_name, "success": False, "reason": "inventory full", "steps": self.steps - start_steps}, info
        info, cursor = self.move_and_click(info, cursor, self.slot_coordinate(destination), "CRAFT_RECIPE/store_output")
        info = self.close_gui(info)
        output_delta = grouped_inventory_value(info, recipe["output"]) - baseline_output
        success = output_delta >= recipe["output_per_click"]
        result = {
            "option": "CRAFT_RECIPE",
            "recipe": recipe_name,
            "success": success,
            "reason": "inventory delta verified" if success else "output inventory delta missing",
            "output_delta": output_delta,
            "steps": self.steps - start_steps,
            "station_records": station_records,
        }
        if success and normalize_identifier(recipe_name) == "wooden_pickaxe":
            station_result, info = self.reclaim_block("crafting_table", info)
            result["station_records"].append(station_result)
            result["steps"] = self.steps - start_steps
        return result, info

    def smelt(self, input_item: str, output_item: str, count: int, fuel_item: str, info: dict) -> tuple[dict, dict]:
        start_steps = self.steps
        baseline_output = grouped_inventory_value(info, output_item)
        if count <= 0:
            return {"option": "SMELT", "success": False, "reason": "count must be positive", "steps": 0}, info
        if grouped_inventory_value(info, input_item) < count:
            return {"option": "SMELT", "success": False, "reason": f"missing {input_item}", "steps": 0}, info
        fuel_capacity = {"coal": 8, "charcoal": 8, "log": 1.5, "planks": 1.5, "stick": 0.5}
        normalized_fuel = normalize_identifier(fuel_item)
        capacity = next((value for name, value in fuel_capacity.items() if item_matches(normalized_fuel, name)), 0.0)
        if capacity <= 0:
            return {"option": "SMELT", "success": False, "reason": f"unsupported fuel {fuel_item}", "steps": 0}, info
        fuel_count = int(np.ceil(count / capacity))
        if grouped_inventory_value(info, fuel_item) < fuel_count:
            return {"option": "SMELT", "success": False, "reason": f"missing {fuel_item}", "required": fuel_count, "steps": 0}, info
        input_equip, info = self.equip_item(input_item, info)
        fuel_equip, info = self.equip_item(fuel_item, info)
        if not input_equip["success"] or not fuel_equip["success"]:
            return {"option": "SMELT", "success": False, "reason": "could not prepare smelting hotbar", "steps": self.steps - start_steps}, info
        station_records = []
        if grouped_inventory_value(info, "furnace") >= 1:
            station_result, info = self.place_block("furnace", info)
            station_records.append(station_result)
            if not station_result["success"]:
                return {"option": "SMELT", "success": False, "reason": station_result["reason"], "steps": self.steps - start_steps, "station_records": station_records}, info
        if not bool(info.get("is_gui_open", False)):
            info = self.open_station("furnace", info)
        if not bool(info.get("is_gui_open", False)):
            return {"option": "SMELT", "success": False, "reason": "furnace GUI unavailable", "steps": self.steps - start_steps, "station_records": station_records}, info
        cursor = detect_cursor(info)
        info, cursor = self.move_and_click(info, cursor, (395.0, 228.0), "SMELT/focus")
        input_slot = self.inventory_slot(info, input_item)
        fuel_slot = self.inventory_slot(info, fuel_item)
        if input_slot is None or fuel_slot is None:
            info = self.close_gui(info)
            return {"option": "SMELT", "success": False, "reason": "smelting inventory slots unavailable", "steps": self.steps - start_steps, "station_records": station_records}, info
        input_coordinate = self.slot_coordinate(input_slot)
        info, cursor = self.move_and_click(info, cursor, input_coordinate, "SMELT/pick_input")
        info, cursor = self.move_and_click(info, cursor, FURNACE_INPUT, "SMELT/place_input")
        info, cursor = self.hotbar_swap(info, cursor, FURNACE_FUEL, fuel_slot, "SMELT/place_fuel")
        wait_started = time.monotonic()
        wait_steps = 0
        while wait_steps < count * 210 or time.monotonic() - wait_started < count * 11.0:
            observation, info = self.step_values({}, "SMELT/wait")
            wait_steps += 1
        info, cursor = self.shift_click(info, cursor, FURNACE_OUTPUT, "SMELT/take_output")
        info = self.close_gui(info)
        output_delta = grouped_inventory_value(info, output_item) - baseline_output
        while output_delta < count and grouped_inventory_value(info, input_item) > 0:
            recovery_slot = self.inventory_slot(info, input_item)
            info = self.open_station("furnace", info)
            if not bool(info.get("is_gui_open", False)) or recovery_slot is None:
                break
            cursor = detect_cursor(info)
            recovery_coordinate = self.slot_coordinate(recovery_slot)
            info, cursor = self.shift_click(info, cursor, recovery_coordinate, "SMELT/recovery_input")
            recovery_started = time.monotonic()
            recovery_steps = 0
            while recovery_steps < 240 or time.monotonic() - recovery_started < 12.0:
                observation, info = self.step_values({}, "SMELT/recovery_wait")
                recovery_steps += 1
            info, cursor = self.shift_click(info, cursor, FURNACE_OUTPUT, "SMELT/recovery_output")
            info = self.close_gui(info)
            updated_delta = grouped_inventory_value(info, output_item) - baseline_output
            if updated_delta <= output_delta:
                break
            output_delta = updated_delta
        success = output_delta >= count
        return {
            "option": "SMELT",
            "input": input_item,
            "output": output_item,
            "count": count,
            "fuel": fuel_item,
            "success": success,
            "reason": "inventory delta verified" if success else "output inventory delta missing",
            "output_delta": output_delta,
            "steps": self.steps - start_steps,
            "station_records": station_records,
        }, info

    def equip_item(self, item_name: str, info: dict) -> tuple[dict, dict]:
        start_steps = self.steps
        equipped = normalize_identifier(str(info.get("equipped_items", {}).get("mainhand", {}).get("type", "air")))
        if item_matches(equipped, item_name):
            return {"option": "EQUIP_ITEM", "item": item_name, "success": True, "reason": "already equipped", "steps": 0}, info
        slot = self.inventory_slot(info, item_name)
        if slot is None:
            return {"option": "EQUIP_ITEM", "item": item_name, "success": False, "reason": "item missing", "steps": 0}, info
        if slot >= 9:
            info, cursor = self.open_inventory(info)
            destination = self.empty_slot(info)
            if destination is None or destination >= 9:
                destination = 0
            info, cursor = self.move_and_click(info, cursor, self.slot_coordinate(slot), "EQUIP_ITEM/pick")
            info, cursor = self.move_and_click(info, cursor, self.slot_coordinate(destination), "EQUIP_ITEM/hotbar")
            info = self.close_gui(info)
            slot = destination
        observation, info = self.step_values({f"hotbar.{slot + 1}": 1}, "EQUIP_ITEM/select")
        observation, info = self.step_values({f"hotbar.{slot + 1}": 0}, "EQUIP_ITEM/select")
        equipped = normalize_identifier(str(info.get("equipped_items", {}).get("mainhand", {}).get("type", "air")))
        success = item_matches(equipped, item_name)
        return {
            "option": "EQUIP_ITEM",
            "item": item_name,
            "success": success,
            "reason": "mainhand verified" if success else f"mainhand is {equipped}",
            "steps": self.steps - start_steps,
        }, info

    def set_pitch(self, target_pitch: float, info: dict):
        for _ in range(12):
            current_pitch = float(info.get("player_pos", {}).get("pitch", 0.0))
            delta = np.clip(target_pitch - current_pitch, -8.0, 8.0)
            if abs(delta) <= 2.0:
                break
            observation, info = self.step_values({"camera": np.array([delta, 0.0], dtype=np.float32)}, "PLACE_BLOCK/look_down")
        return info

    def place_block(self, item_name: str, info: dict) -> tuple[dict, dict]:
        start_steps = self.steps
        baseline = grouped_inventory_value(info, item_name)
        equip_result, info = self.equip_item(item_name, info)
        if not equip_result["success"]:
            return {"option": "PLACE_BLOCK", "item": item_name, "success": False, "reason": equip_result["reason"], "steps": self.steps - start_steps}, info
        recovery_plan = [
            (80.0, 0.0, "back", 12),
            (70.0, 35.0, "right", 8),
            (60.0, -70.0, "left", 8),
            (50.0, 105.0, "back", 12),
            (65.0, 180.0, "forward", 10),
        ]
        for pitch, yaw_delta, movement, movement_steps in recovery_plan:
            for _ in range(movement_steps):
                observation, info = self.step_values({movement: 1, "jump": 1}, "PLACE_BLOCK/reposition")
            observation, info = self.step_values({movement: 0, "jump": 0}, "PLACE_BLOCK/reposition")
            info = self.set_pitch(pitch, info)
            if yaw_delta:
                remaining_yaw = yaw_delta
                while abs(remaining_yaw) > 2.0:
                    yaw_step = float(np.clip(remaining_yaw, -8.0, 8.0))
                    observation, info = self.step_values({"camera": np.array([0.0, yaw_step], dtype=np.float32)}, "PLACE_BLOCK/search_surface")
                    remaining_yaw -= yaw_step
            for _ in range(2):
                observation, info = self.step_values({"use": 1}, "PLACE_BLOCK/use")
                observation, info = self.step_values({"use": 0}, "PLACE_BLOCK/use")
                if grouped_inventory_value(info, item_name) < baseline:
                    return {"option": "PLACE_BLOCK", "item": item_name, "success": True, "reason": "inventory consumption verified", "steps": self.steps - start_steps}, info
        return {"option": "PLACE_BLOCK", "item": item_name, "success": False, "reason": "block was not consumed", "steps": self.steps - start_steps}, info

    def reclaim_block(self, item_name: str, info: dict, max_steps: int = 100) -> tuple[dict, dict]:
        start_steps = self.steps
        baseline = grouped_inventory_value(info, item_name)
        for _ in range(max_steps):
            observation, info = self.step_values({"attack": 1}, "RECLAIM_BLOCK/attack")
            if grouped_inventory_value(info, item_name) > baseline:
                observation, info = self.step_values({"attack": 0}, "RECLAIM_BLOCK/release")
                return {"option": "RECLAIM_BLOCK", "item": item_name, "success": True, "reason": "inventory delta verified", "steps": self.steps - start_steps}, info
        observation, info = self.step_values({"attack": 0}, "RECLAIM_BLOCK/release")
        collect_result, info = self.collect_drop(item_name, baseline, info, max_steps=80)
        success = grouped_inventory_value(info, item_name) > baseline
        return {
            "option": "RECLAIM_BLOCK",
            "item": item_name,
            "success": success,
            "reason": "inventory delta verified" if success else collect_result["reason"],
            "steps": self.steps - start_steps,
        }, info

    def collect_drop(self, target: str, baseline_quantity: float, info: dict, max_steps: int = 120) -> tuple[dict, dict]:
        if self.pickup_implementation == "old":
            return self._collect_drop_old(target, baseline_quantity, info, max_steps)
        return self._collect_drop_hardened(target, baseline_quantity, info, max_steps)

    def _collect_drop_old(self, target: str, baseline_quantity: float, info: dict, max_steps: int) -> tuple[dict, dict]:
        start_steps = self.steps
        pattern = [
            {"forward": 1},
            {"camera": np.array([8.0, 0.0], dtype=np.float32)},
            {"camera": np.array([0.0, 8.0], dtype=np.float32)},
            {"left": 1},
            {"camera": np.array([0.0, -8.0], dtype=np.float32)},
            {"right": 1},
            {"back": 1},
        ]
        for step_index in range(max_steps):
            values = pattern[(step_index // 8) % len(pattern)]
            observation, info = self.step_values(values, "COLLECT_DROP/sweep")
            if grouped_inventory_value(info, target) > baseline_quantity:
                return {"option": "COLLECT_DROP", "target": target, "success": True, "reason": "inventory delta verified", "steps": self.steps - start_steps}, info
        return {"option": "COLLECT_DROP", "target": target, "success": False, "reason": "timeout", "steps": self.steps - start_steps}, info

    def _collect_drop_hardened(self, target: str, baseline_quantity: float, info: dict, max_steps: int) -> tuple[dict, dict]:
        start_steps = self.steps
        origin = player_position(info)
        movement_samples = [origin]
        plan = [
            ({"forward": 1}, 8, "approach"),
            ({"camera": np.array([8.0, 0.0], dtype=np.float32)}, 8, "look_down"),
            ({"camera": np.array([0.0, 8.0], dtype=np.float32)}, 8, "scan_right"),
            ({"left": 1}, 8, "sweep_left"),
            ({"camera": np.array([0.0, -8.0], dtype=np.float32)}, 8, "scan_left"),
            ({"right": 1}, 8, "sweep_right"),
            ({"back": 1}, 8, "reverse"),
            ({"forward": 1, "jump": 1}, 6, "step_over"),
            ({"camera": np.array([6.0, 12.0], dtype=np.float32)}, 4, "tight_scan_right"),
            ({"forward": 1}, 6, "tight_approach_right"),
            ({"camera": np.array([0.0, -24.0], dtype=np.float32)}, 4, "tight_scan_left"),
            ({"forward": 1}, 6, "tight_approach_left"),
        ]
        executed_steps = 0
        cycle = 0
        while executed_steps < max_steps:
            for values, repetitions, phase in plan:
                for _ in range(min(repetitions, max_steps - executed_steps)):
                    observation, info = self.step_values(values, f"COLLECT_DROP/{phase}")
                    executed_steps += 1
                    quantity = grouped_inventory_value(info, target)
                    if quantity > baseline_quantity:
                        return {
                            "option": "COLLECT_DROP",
                            "target": target,
                            "success": True,
                            "status": "SUCCESS",
                            "reason": "inventory delta verified",
                            "inventory_delta": quantity - baseline_quantity,
                            "steps": self.steps - start_steps,
                        }, info
                    position = player_position(info)
                    movement_samples.append(position)
                    horizontal_distance = float(np.linalg.norm((position - origin)[[0, 2]]))
                    vertical_drop = float(origin[1] - position[1])
                    in_water = player_in_water(info)
                    recent_displacement = float(np.linalg.norm(movement_samples[-1] - movement_samples[max(0, len(movement_samples) - 13)]))
                    uncertain = in_water or vertical_drop > 1.5 or horizontal_distance > 6.0
                    blocked = executed_steps >= 24 and recent_displacement < 0.12 and any(values.get(key, 0) for key in ("forward", "back", "left", "right"))
                    if uncertain or blocked:
                        reason = "pickup geometry uncertain"
                        if in_water:
                            reason = "pickup geometry uncertain: entered water"
                        elif vertical_drop > 1.5:
                            reason = "pickup geometry uncertain: unexpected vertical drop"
                        elif horizontal_distance > 6.0:
                            reason = "pickup geometry uncertain: drop no longer local"
                        elif blocked:
                            reason = "pickup geometry uncertain: local movement blocked"
                        return {
                            "option": "COLLECT_DROP",
                            "target": target,
                            "success": False,
                            "status": "NEEDS_SYSTEM1",
                            "needs_system1": True,
                            "reason": reason,
                            "steps": self.steps - start_steps,
                        }, info
                if executed_steps >= max_steps:
                    break
            cycle += 1
            if cycle >= 2 and executed_steps < max_steps:
                return {
                    "option": "COLLECT_DROP",
                    "target": target,
                    "success": False,
                    "status": "NEEDS_SYSTEM1",
                    "needs_system1": True,
                    "reason": "pickup geometry uncertain after bounded local search",
                    "steps": self.steps - start_steps,
                }, info
        return {
            "option": "COLLECT_DROP",
            "target": target,
            "success": False,
            "status": "TIMEOUT",
            "reason": "timeout",
            "steps": self.steps - start_steps,
        }, info
