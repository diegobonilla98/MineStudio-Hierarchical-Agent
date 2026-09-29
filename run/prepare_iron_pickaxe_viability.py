import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SUITE_DIRECTORY = Path(os.environ.get("MINESTUDIO_IRON_SUITE", PROJECT_DIRECTORY / "output" / "iron_pickaxe_viability" / "development"))
EPISODES = int(os.environ.get("MINESTUDIO_IRON_EPISODES", "24"))
BASE_SEED = int(os.environ.get("MINESTUDIO_IRON_BASE_SEED", "2026092101"))
MAX_TOTAL_STEPS = int(os.environ.get("MINESTUDIO_IRON_MAX_STEPS", "12000"))
MAX_DECISIONS = int(os.environ.get("MINESTUDIO_IRON_MAX_DECISIONS", "32"))


def main() -> None:
    SUITE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = SUITE_DIRECTORY / "manifest.json"
    generator = random.Random(BASE_SEED)
    seeds = []
    while len(seeds) < EPISODES:
        seed = generator.randint(1, 2_147_483_646)
        if seed not in seeds:
            seeds.append(seed)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paired": True,
        "objective": "Build an iron pickaxe from a fresh random survival world.",
        "base_seed": BASE_SEED,
        "episode_count": EPISODES,
        "max_total_steps": MAX_TOTAL_STEPS,
        "max_decisions": MAX_DECISIONS,
        "observation_privileges": "player-visible RGB and ordinary MineStudio player state only; no hidden world coordinates or map",
        "arms": ["full", "no_specialist", "no_system2"],
        "episodes": [{"episode_index": index, "seed": seed} for index, seed in enumerate(seeds)],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key not in {"created_at", "manifest_sha256"}}
        requested = {key: value for key, value in payload.items() if key not in {"created_at", "manifest_sha256"}}
        if comparable != requested:
            raise RuntimeError(f"Existing frozen manifest differs from requested experiment: {path}")
        print(json.dumps(existing, indent=2))
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
