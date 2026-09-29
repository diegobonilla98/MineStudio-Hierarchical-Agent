import gc
import os
from pathlib import Path

import torch
from minestudio.models import SteveOnePolicy
from train_stone_recovery_bc import BASE_CHECKPOINT, configure_trainable, load_trainable_state, merge_lora


OUTPUT_DIRECTORY = Path(os.environ["MINESTUDIO_BC_OUTPUT_DIRECTORY"])
VARIANT = os.environ.get("MINESTUDIO_BC_VARIANT", "upper_lora")
STEPS = tuple(int(value) for value in os.environ.get("MINESTUDIO_BC_MATERIALIZE_STEPS", "500,600,700,800").split(",") if value.strip())


def main() -> None:
    variant_directory = OUTPUT_DIRECTORY / VARIANT
    snapshot_directory = variant_directory / "snapshots"
    for step in STEPS:
        state_path = snapshot_directory / f"step_{step}_trainable.pt"
        destination = snapshot_directory / f"step_{step}_model"
        if destination.is_dir() and (destination / "model.safetensors").is_file():
            continue
        model = SteveOnePolicy.from_pretrained(BASE_CHECKPOINT).to("cuda")
        configure_trainable(model, VARIANT)
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        load_trainable_state(model, state)
        merge_lora(model)
        model.save_pretrained(destination)
        del state
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"materialized_step={step} destination={destination}", flush=True)


if __name__ == "__main__":
    main()
