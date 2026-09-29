import json
import pickle
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import av
import httpx
import numpy as np

import minestudio.tutorials.simulator.gemini_coach as coach
from minestudio.utils.vpt_lib.actions import ActionTransformer


class DemonstrationRecorderTest(unittest.TestCase):
    def test_open_ended_subtask_is_rejected(self) -> None:
        task = coach.CoachTask(
            instruction="Find iron ore.",
            hint="Search underground.",
            verifier=coach.VerifierSpec(kind="travel_distance", source="player_pos", target="any", threshold=8),
            max_seconds=90,
        )

        with self.assertRaises(ValueError):
            coach.validate_task(task)

    def test_exploration_requires_time_and_movement(self) -> None:
        state = {
            "inventory": {},
            "mine_block": {},
            "craft_item": {},
            "pickup": {},
            "kill_entity": {},
            "player_pos": {"x": 0.0, "y": 64.0, "z": 0.0},
        }
        verifier = coach.VerifierSpec(kind="explore_duration", source="player_pos", target="any", threshold=45)

        progress, succeeded = coach.verifier_progress(verifier, state, state, 45.0, 0.0)
        self.assertFalse(succeeded)
        self.assertEqual(progress, 0.0)

        progress, succeeded = coach.verifier_progress(verifier, state, state, 44.0, 20.0)
        self.assertFalse(succeeded)
        self.assertLess(progress, 45.0)

        progress, succeeded = coach.verifier_progress(verifier, state, state, 45.0, 9.0)
        self.assertTrue(succeeded)
        self.assertEqual(progress, 45.0)

    def test_numbered_walk_is_rejected_even_with_exploration_verifier(self) -> None:
        task = coach.CoachTask(
            instruction="Walk 12 blocks while looking for coal.",
            hint="Walk forward.",
            verifier=coach.VerifierSpec(kind="explore_duration", source="player_pos", target="any", threshold=45),
            max_seconds=60,
        )

        with self.assertRaises(ValueError):
            coach.validate_task(task)

    def test_global_goal_selection_is_seeded(self) -> None:
        first = coach.select_global_goal(987654)
        second = coach.select_global_goal(987654)

        self.assertEqual(first, second)

    def test_generic_material_group_combines_subtypes(self) -> None:
        baseline = {
            "inventory": {"oak_log": 1.0},
            "mine_block": {},
            "craft_item": {},
            "pickup": {},
            "kill_entity": {},
            "player_pos": {"x": 0.0, "y": 64.0, "z": 0.0},
        }
        current = {
            **baseline,
            "inventory": {"oak_log": 1.0, "spruce_log": 1.0, "birch_log": 1.0},
        }
        verifier = coach.VerifierSpec(kind="inventory_delta", source="inventory", target="log", threshold=2)

        progress, succeeded = coach.verifier_progress(verifier, baseline, current)

        self.assertEqual(progress, 2.0)
        self.assertTrue(succeeded)

    def test_frame_action_task_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            coach.DATASET_DIRECTORY = Path(temporary_directory)
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            response = coach.CoachResponse(
                feedback="Begin.",
                strategic_goal="Obtain basic tools.",
                plan=["Gather wood", "Craft a pickaxe"],
                context_request=coach.ContextRequest(),
                task=coach.CoachTask(
                    instruction="Walk forward.",
                    hint="Hold W.",
                    verifier=coach.VerifierSpec(kind="travel_distance", source="player_pos", target="any", threshold=1),
                    max_seconds=30,
                ),
            )
            state = {
                "inventory": {},
                "mine_block": {},
                "craft_item": {},
                "pickup": {},
                "kill_entity": {},
                "player_pos": {"x": 0.0, "y": 64.0, "z": 0.0},
                "health": 20.0,
                "food_level": 20.0,
            }
            global_goal = coach.GLOBAL_GOALS[0]
            recorder = coach.DemonstrationRecorder(frame, 123456, global_goal)
            recorder.start_task(response, state)
            for frame_id in range(3):
                next_state = json.loads(json.dumps(state))
                next_state["player_pos"]["x"] = float(frame_id + 1)
                action = {"forward": 1, "camera": np.asarray([0.0, 0.0], dtype=np.float32)}
                recorder.record_step(frame, action, state, next_state, float(frame_id), float(frame_id + 1), 0.0, False, False, float(frame_id), float(frame_id + 1))
                state = next_state
            recorder.finish_task("succeeded", state, 3.0)
            recorder.close("viewer_closed", state, 3.0, 3.0, False)

            session_directory = next(Path(temporary_directory).iterdir())
            metadata = json.loads((session_directory / "session.json").read_text(encoding="utf-8"))
            actions = json.loads((session_directory / metadata["files"]["actions"]).read_text(encoding="utf-8"))
            with (session_directory / metadata["files"]["minestudio_actions"]).open("rb") as action_pickle_file:
                action_arrays = pickle.load(action_pickle_file)
            policy_actions = ActionTransformer().env2policy(action_arrays)
            infos = json.loads((session_directory / metadata["files"]["infos"]).read_text(encoding="utf-8"))
            steps = [json.loads(line) for line in (session_directory / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
            tasks = [json.loads(line) for line in (session_directory / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
            video_container = av.open(session_directory / metadata["files"]["video"])
            decoded_frames = sum(1 for _ in video_container.decode(video=0))
            video_container.close()

            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["world_seed"], 123456)
            self.assertEqual(metadata["global_goal"]["goal_id"], global_goal.goal_id)
            self.assertEqual(metadata["global_progress"], 3.0)
            self.assertEqual(Path(metadata["files"]["video"]).stem, metadata["session_id"])
            self.assertEqual(Path(metadata["files"]["minestudio_actions"]).stem, metadata["session_id"])
            self.assertEqual(metadata["total_frames"], 3)
            self.assertEqual(len(actions), 3)
            self.assertEqual(action_arrays["camera"].shape, (3, 2))
            self.assertEqual(action_arrays["forward"].tolist(), [1, 1, 1])
            self.assertEqual(policy_actions["camera"].shape, (3, 2))
            self.assertEqual(policy_actions["buttons"].shape[0], 3)
            self.assertEqual(len(infos), 3)
            self.assertEqual(len(steps), 3)
            self.assertEqual(decoded_frames, 3)
            self.assertEqual(tasks[0]["start_frame"], 0)
            self.assertEqual(tasks[0]["end_frame"], 3)
            self.assertTrue(tasks[0]["succeeded"])
            self.assertEqual(steps[0]["action"], actions[0])
            self.assertEqual(steps[-1]["global_goal_id"], global_goal.goal_id)
            self.assertEqual(steps[-1]["global_progress_after"], 3.0)

    def test_context_request_expands_frame_delivery(self) -> None:
        class FakeModels:
            def __init__(self) -> None:
                self.calls = []

            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                request_more = len(self.calls) == 1
                payload = {
                    "feedback": "Continue.",
                    "strategic_goal": "Obtain basic tools.",
                    "plan": ["Gather wood", "Craft a pickaxe"],
                    "context_request": {
                        "more_frames": request_more,
                        "frame_count": 20,
                        "reason": "Need a wider view",
                    },
                    "task": {
                        "instruction": "Collect one log.",
                        "hint": "Break and collect a nearby tree trunk.",
                        "verifier": {
                            "kind": "inventory_delta",
                            "source": "inventory",
                            "target": "log",
                            "threshold": 1,
                        },
                        "max_seconds": 30,
                    },
                }
                return SimpleNamespace(text=json.dumps(payload))

        client = SimpleNamespace(models=FakeModels())
        state = {
            "inventory": {},
            "mine_block": {},
            "craft_item": {},
            "pickup": {},
            "kill_entity": {},
            "player_pos": {"x": 0.0, "y": 64.0, "z": 0.0},
            "health": 20.0,
            "food_level": 20.0,
        }
        frames = deque([b"frame"] * 20, maxlen=coach.MAX_BUFFERED_CONTEXT_FRAMES)
        response = coach.request_task(client, "session_start", state, frames, deque(), [], coach.GLOBAL_GOALS[0], 0.0)

        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(len(client.models.calls[0]["contents"]), coach.MAX_CONTEXT_FRAMES + 1)
        self.assertEqual(len(client.models.calls[1]["contents"]), 21)
        self.assertFalse(response.context_request.more_frames)

    def test_network_failure_uses_local_fallback(self) -> None:
        class OfflineModels:
            def generate_content(self, **kwargs):
                raise httpx.ConnectError("offline")

        state = {
            "inventory": {},
            "mine_block": {},
            "craft_item": {},
            "pickup": {},
            "kill_entity": {},
            "player_pos": {"x": 0.0, "y": 64.0, "z": 0.0},
            "health": 20.0,
            "food_level": 20.0,
        }
        client = SimpleNamespace(models=OfflineModels())
        response = coach.request_task_resilient(client, "session_start", state, deque([b"frame"]), deque(), [], coach.GLOBAL_GOALS[0], 0.0)

        self.assertEqual(response.task.verifier.kind, "explore_duration")
        self.assertEqual(response.task.verifier.threshold, 60)
        self.assertIn("temporarily unavailable", response.feedback)

    def test_malformed_response_uses_local_fallback(self) -> None:
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: SimpleNamespace(text='{"feedback":')))
        state = {
            "inventory": {},
            "mine_block": {},
            "craft_item": {},
            "pickup": {},
            "kill_entity": {},
            "player_pos": {"x": 0.0, "y": 64.0, "z": 0.0},
            "health": 20.0,
            "food_level": 20.0,
        }
        response = coach.request_task_resilient(client, "session_start", state, deque([b"frame"]), deque(), [], coach.GLOBAL_GOALS[0], 0.0)

        self.assertEqual(response.task.verifier.kind, "explore_duration")


if __name__ == "__main__":
    unittest.main()
