import json
import os
from collections import deque
from pathlib import Path

import cv2
from dotenv import load_dotenv
from google import genai

from evaluate_steve_hierarchy import GLOBAL_OBJECTIVE, PROJECT_DIRECTORY, SKILL_LIBRARY, encode_frame, request_decision


REFERENCE_FRAME_PATH = PROJECT_DIRECTORY / "output" / "steve_skill_evaluation" / "20260915T092103Z_paper_wood_prompt_smoke" / "mine_log_paper_002_20276753_final.jpg"
RECOVERY_FRAME_PATH = PROJECT_DIRECTORY / "output" / "steve_mine_log_benchmark" / "20260915T092936Z_mine_log_100" / "episode_003_20284672_final.jpg"


def main() -> None:
    load_dotenv(PROJECT_DIRECTORY / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    frame = cv2.imread(str(REFERENCE_FRAME_PATH))
    if frame is None:
        raise FileNotFoundError(REFERENCE_FRAME_PATH)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    state = {
        "inventory": {},
        "mine_block": {},
        "craft_item": {},
        "pickup": {},
        "place_block": {},
        "kill_entity": {},
        "player_pos": {"x": 0.0, "y": 64.0, "z": 0.0},
        "health": 20.0,
        "food_level": 20.0,
    }
    client = genai.Client(api_key=api_key)
    decision, planner_error = request_decision(client, "session_start", state, deque([encode_frame(frame)], maxlen=12), [])
    skill = SKILL_LIBRARY[decision.executor_skill]
    start_result = {
        "global_objective": decision.global_objective,
        "local_objective": decision.local_objective,
        "executor_skill": decision.executor_skill,
        "executor_prompt": skill["executor_prompt"],
        "failure_category": decision.failure_category,
        "evidence": decision.evidence,
        "forward_plan": decision.forward_plan,
        "max_steps": decision.max_steps,
        "planner_error": planner_error,
    }
    recovery_frame = cv2.imread(str(RECOVERY_FRAME_PATH))
    if recovery_frame is None:
        raise FileNotFoundError(RECOVERY_FRAME_PATH)
    recovery_frame = cv2.cvtColor(recovery_frame, cv2.COLOR_BGR2RGB)
    recovery_state = dict(state)
    recovery_state["mine_block"] = {"oak_log": 2.0}
    attempts = [{
        "decision": {"local_objective": "Collect one log.", "executor_skill": "collect_log"},
        "outcome": "timeout",
        "local_progress": 0.0,
        "steps": 1000,
        "action_counts": {"attack": 900, "forward": 30, "camera": 20},
        "final_state": recovery_state,
    }]
    recovery_decision, recovery_error = request_decision(
        client,
        "timeout",
        recovery_state,
        deque([encode_frame(recovery_frame)], maxlen=12),
        attempts,
    )
    recovery_skill = SKILL_LIBRARY[recovery_decision.executor_skill]
    recovery_result = {
        "global_objective": recovery_decision.global_objective,
        "local_objective": recovery_decision.local_objective,
        "executor_skill": recovery_decision.executor_skill,
        "executor_prompt": recovery_skill["executor_prompt"],
        "failure_category": recovery_decision.failure_category,
        "evidence": recovery_decision.evidence,
        "forward_plan": recovery_decision.forward_plan,
        "max_steps": recovery_decision.max_steps,
        "planner_error": recovery_error,
    }
    print(json.dumps({"session_start": start_result, "log_broken_but_not_acquired": recovery_result}, indent=2))
    if decision.global_objective != GLOBAL_OBJECTIVE or decision.executor_skill not in SKILL_LIBRARY or planner_error:
        raise RuntimeError("Hierarchy planner smoke failed")
    if recovery_decision.executor_skill != "pickup_log" or recovery_decision.failure_category != "execution_or_recovery" or recovery_error:
        raise RuntimeError("Hierarchy planner recovery smoke failed")


if __name__ == "__main__":
    main()
