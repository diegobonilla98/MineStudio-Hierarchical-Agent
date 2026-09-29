import json
from pathlib import Path

import numpy as np
import torch
from minestudio.models import SteveOnePolicy


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = PROJECT_DIRECTORY / "checkpoints" / "steve_one_official"
DEVICE = "cuda"
PROMPT = "mine a log"
CONDITION_SCALE = 6.0
RANDOM_SEED = 20260915


def tensor_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.dtype == torch.bool:
        return 0.0 if torch.equal(first, second) else 1.0
    return float((first.detach() - second.detach()).abs().max().item())


def main() -> None:
    torch.set_float32_matmul_precision("high")
    model = SteveOnePolicy.from_pretrained(CHECKPOINT_DIRECTORY).to(DEVICE).eval()
    image = np.arange(128 * 128 * 3, dtype=np.uint8).reshape(128, 128, 3)

    torch.manual_seed(RANDOM_SEED)
    official_input = {
        "image": image,
        "condition": {"cond_scale": CONDITION_SCALE, "text": PROMPT},
    }
    official_action, official_state = model.get_action(
        official_input,
        None,
        deterministic=False,
        input_shape="*",
    )
    official_logits = {name: value.detach().clone() for name, value in model.cache_latents["pi_logits"].items()}

    torch.manual_seed(RANDOM_SEED)
    cached_condition = model.prepare_condition(
        {"cond_scale": CONDITION_SCALE, "text": PROMPT},
        deterministic=False,
    )
    cached_input = {
        "image": torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE),
        "condition": cached_condition,
    }
    cached_action, cached_state = model.get_action(
        cached_input,
        None,
        deterministic=False,
        input_shape="BT*",
    )
    cached_action = {name: value[0][0] for name, value in cached_action.items()}
    cached_logits = {name: value[0][0].detach().clone() for name, value in model.cache_latents["pi_logits"].items()}

    result = {
        "prompt": PROMPT,
        "condition_scale": CONDITION_SCALE,
        "official_action": {name: int(value.detach().cpu().numpy().item()) for name, value in official_action.items()},
        "cached_action": {name: int(value.detach().cpu().numpy().item()) for name, value in cached_action.items()},
        "action_equal": all(torch.equal(official_action[name], cached_action[name]) for name in official_action),
        "max_logit_difference": max(tensor_difference(official_logits[name], cached_logits[name]) for name in official_logits),
        "max_state_difference": max(tensor_difference(first, second[0]) for first, second in zip(official_state, cached_state)),
    }
    print(json.dumps(result, indent=2))
    if not result["action_equal"] or result["max_logit_difference"] > 1e-5 or result["max_state_difference"] > 1e-5:
        raise RuntimeError("Cached and official STEVE-1 inference paths differ")


if __name__ == "__main__":
    main()
