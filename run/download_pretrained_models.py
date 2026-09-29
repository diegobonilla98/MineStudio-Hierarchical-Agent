import hashlib
import json
from pathlib import Path
from time import time

from huggingface_hub import snapshot_download

checkpoint_root = Path(__file__).resolve().parents[1] / "checkpoints"
manifest_path = checkpoint_root / "manifest.json"
temporary_manifest_path = checkpoint_root / "manifest.json.tmp"
max_workers = 4
repositories = {
    "steve_one_official": {
        "repo_id": "CraftJarvis/MineStudio_STEVE-1.official",
        "revision": "34e18c2fd00058deb371b6c34b183eb52de8166c",
    },
    "vpt_early_game_2x": {
        "repo_id": "CraftJarvis/MineStudio_VPT.rl_from_early_game_2x",
        "revision": "3db1bf4e2c0e121df128deed9ee1622f19c339ff",
    },
    "groot_18w_ema": {
        "repo_id": "CraftJarvis/MineStudio_GROOT.18w_EMA",
        "revision": "f721338663e9acfbdf5f4924ad3cd74e44316650",
    },
    "rocket_two_1x_22w": {
        "repo_id": "phython96/ROCKET-2-1x-22w",
        "revision": "a9ec41ed60736de310f45c4b8b14063d5d31c7d9",
    },
    "rocket_two_1_5x_17w": {
        "repo_id": "phython96/ROCKET-2-1.5x-17w",
        "revision": "d3177f39036574d676ae0f34724238f0d06e667c",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


checkpoint_root.mkdir(parents=True, exist_ok=True)
manifest = {"created_unix": int(time()), "models": {}}

for name, specification in repositories.items():
    destination = checkpoint_root / name
    snapshot_path = snapshot_download(
        repo_id=specification["repo_id"],
        revision=specification["revision"],
        local_dir=destination,
        max_workers=max_workers,
    )
    files = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files.append(
                {
                    "path": str(path.relative_to(destination)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest["models"][name] = {
        **specification,
        "snapshot_path": snapshot_path,
        "total_bytes": sum(file["bytes"] for file in files),
        "files": files,
    }

temporary_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
temporary_manifest_path.replace(manifest_path)
print(json.dumps(manifest, indent=2))
