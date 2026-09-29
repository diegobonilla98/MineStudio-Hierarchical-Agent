import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "dataset_v1" / "gemini_task_bank.json"
MODEL_NAME = "gemini-3.8-flash"
TASKS_PER_CATEGORY = 8
CATEGORIES = (
    "stone_acquisition",
    "water_recovery",
    "shore_exit",
    "stone_reacquisition",
    "hole_recovery",
)
SYSTEM_PROMPT = """You write local Minecraft executor objectives for a pretrained visuomotor policy. Each instruction must be a single concrete, immediately executable task, not a plan or explanation. Preserve the requested category and goal. The instruction must remain valid across randomized worlds, so never assume a direction, visible target, biome material, shoreline shape, terrain height, lighting, or exact obstacle geometry. Recovery tasks may use only looking, walking, jumping, and swimming; never prescribe mining, attacking, placing blocks, crafting, or inventory use. Use only information an agent could infer from recent RGB frames and ordinary player state. Do not mention coordinates, commands, hidden blocks, reward, verification, datasets, or training. Vary natural wording while keeping the intended behavior unambiguous."""


class GeneratedTask(BaseModel):
    category: Literal["stone_acquisition", "water_recovery", "shore_exit", "stone_reacquisition", "hole_recovery"]
    instruction: str = Field(min_length=8, max_length=180)


class GeneratedTaskBank(BaseModel):
    tasks: list[GeneratedTask]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_instruction(category: str, instruction: str) -> bool:
    text = instruction.lower()
    common_forbidden = ("left", "right", "ahead", "in front", "sunlit", "grass", "dirt", "sand", "gravel", "mud", "two-block")
    recovery_forbidden = ("mine", "attack", "place", "craft", "inventory", "pillar", "carve", "dig")
    if any(value in text for value in common_forbidden):
        return False
    if category != "stone_acquisition" and any(value in text for value in recovery_forbidden):
        return False
    return True


def main() -> None:
    load_dotenv(PROJECT_DIRECTORY / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY is missing from {PROJECT_DIRECTORY / '.env'}")
    request = {
        "global_context": "The larger hierarchy is trying to craft a stone pickaxe.",
        "task": "Generate local objectives for a specialist that must obtain three cobblestone and recover from common terrain hazards.",
        "categories": {
            "stone_acquisition": "Search locally for natural stone, approach it, mine it with the already-equipped wooden pickaxe, and collect three cobblestone. Do not assume stone is already visible or located in any direction.",
            "water_recovery": "The player has entered water. Use swimming, looking, walking, and jumping to find an exit and return to stable land. Do not assume where the shore is or what it is made of.",
            "shore_exit": "Use swimming, looking, walking, and jumping to negotiate an unknown awkward shoreline and finish on stable dry ground. Do not assume the shore geometry or material.",
            "stone_reacquisition": "After a disruption or water exit, visually search for natural stone again and resume approaching it. Do not assume it is already visible or located in any direction.",
            "hole_recovery": "Use only looking, walking, and jumping to climb out of an unknown local depression and regain usable ground. Do not mine or place blocks.",
        },
        "instructions_per_category": TASKS_PER_CATEGORY,
        "required_total": TASKS_PER_CATEGORY * len(CATEGORIES),
    }
    client = genai.Client(api_key=api_key)
    grouped = {category: [] for category in CATEGORIES}
    response_hashes = []
    for generation_index in range(4):
        generation_request = {**request, "generation_index": generation_index, "needed": {category: TASKS_PER_CATEGORY - len(grouped[category]) for category in CATEGORIES}}
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Part.from_text(text=json.dumps(generation_request, separators=(",", ":")))],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=GeneratedTaskBank,
                max_output_tokens=4096,
                temperature=0.8,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        response_hashes.append(sha256_text(response.text))
        parsed = GeneratedTaskBank.model_validate_json(response.text)
        for task in parsed.tasks:
            instruction = " ".join(task.instruction.split())
            if valid_instruction(task.category, instruction) and instruction not in grouped[task.category]:
                grouped[task.category].append(instruction)
        if all(len(grouped[category]) >= TASKS_PER_CATEGORY for category in CATEGORIES):
            break
    counts = {category: len(grouped[category]) for category in CATEGORIES}
    if any(count < TASKS_PER_CATEGORY for count in counts.values()):
        raise RuntimeError(f"Gemini returned insufficient unique tasks: {counts}")
    grouped = {category: values[:TASKS_PER_CATEGORY] for category, values in grouped.items()}
    payload = {
        "format": "minestudio_gemini_task_bank_v1",
        "model": MODEL_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "request": request,
        "response_sha256": response_hashes,
        "tasks": grouped,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(OUTPUT_PATH)
    print(json.dumps({"status": "complete", "output": str(OUTPUT_PATH), "model": MODEL_NAME, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
