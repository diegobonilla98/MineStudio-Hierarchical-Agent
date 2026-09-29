import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from minestudio.simulator import MinecraftSim


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "output"
FRAME_PATH = OUTPUT_DIRECTORY / "dgx_simulator_smoke.jpg"
RESULT_PATH = OUTPUT_DIRECTORY / "dgx_simulator_smoke.json"
WORLD_SEED = 20260915
STEP_COUNT = 3


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    simulator = MinecraftSim(
        action_type="env",
        obs_size=(128, 128),
        render_size=(640, 360),
        seed=WORLD_SEED,
        num_empty_frames=5,
        callbacks=[],
    )
    started_at = time.monotonic()
    try:
        observation, info = simulator.reset()
        reset_seconds = time.monotonic() - started_at
        rewards = []
        terminated = False
        truncated = False
        for _ in range(STEP_COUNT):
            observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
            rewards.append(float(reward))
        frame = np.asarray(info["pov"])
        encoded_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(FRAME_PATH), encoded_frame):
            raise RuntimeError(f"Could not write {FRAME_PATH}")
        result = {
            "seed": WORLD_SEED,
            "reset_seconds": reset_seconds,
            "steps": STEP_COUNT,
            "observation_shape": list(observation["image"].shape),
            "frame_shape": list(frame.shape),
            "frame_mean": float(frame.mean()),
            "frame_std": float(frame.std()),
            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "rewards": rewards,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        simulator.close()


if __name__ == "__main__":
    main()
