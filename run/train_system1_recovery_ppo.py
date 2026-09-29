import json
import math
import os
import random
import socket
import time
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import benchmark_stone_acquisition as stone
import benchmark_system1_recovery_micro as micro
from closed_loop_options import ClosedLoopOptions
from minestudio.models import SteveOnePolicy
from minestudio.simulator.callbacks import CommandsCallback
from stone_task_router import LegacyStoneTaskRouter
from telegram_training_helper import TelegramBot
from train_stone_recovery_bc import PromptBank, WindowStore, batch_condition, forward_loss, inject_lora, load_trainable_state, module_parent, move_batch, normalized_policy_logits, trainable_state


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
REFERENCE_CHECKPOINT = PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "architecture_sweeps" / "20260917T213751Z" / "upper_lora_rank32" / "best_model"
RUN_ID = os.environ.get("MINESTUDIO_PPO_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
OUTPUT_DIRECTORY = Path(os.environ.get("MINESTUDIO_PPO_OUTPUT", PROJECT_DIRECTORY / "output" / "system1_recovery_ppo" / RUN_ID))
SMOKE_MODE = os.environ.get("MINESTUDIO_PPO_SMOKE", "0") == "1"
TELEGRAM_ENABLED = os.environ.get("MINESTUDIO_PPO_TELEGRAM", "1") == "1"
MAX_ITERATIONS = int(os.environ.get("MINESTUDIO_PPO_MAX_ITERATIONS", "30"))
EPISODES_PER_ITERATION = int(os.environ.get("MINESTUDIO_PPO_EPISODES_PER_ITERATION", "6"))
EVALUATION_INTERVAL = int(os.environ.get("MINESTUDIO_PPO_EVALUATION_INTERVAL", "5"))
DEVELOPMENT_MICRO_EPISODES = int(os.environ.get("MINESTUDIO_PPO_DEVELOPMENT_MICRO_EPISODES", "10"))
DEVELOPMENT_STONE_EPISODES = int(os.environ.get("MINESTUDIO_PPO_DEVELOPMENT_STONE_EPISODES", "20"))
FRAGMENT_LENGTH = 64
PPO_EPOCHS = 2
MINIBATCH_FRAGMENTS = 2
BC_BATCH_SIZE = 4
BC_UPDATES_PER_ITERATION = 1
POLICY_LEARNING_RATE = 2e-7
VALUE_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.01
GAMMA = 0.995
GAE_LAMBDA = 0.95
PPO_CLIP = 0.10
VALUE_COEFFICIENT = 0.50
KL_COEFFICIENT = 0.02
ENTROPY_COEFFICIENT = 0.001
BC_COEFFICIENT = 0.10
GRADIENT_CLIP_NORM = 1.0
MAX_APPROXIMATE_KL = 0.05
SHAPING_SCALE = 0.05
CONDITION_SCALE = 6.0
MINIMUM_ITERATIONS = 15
PLATEAU_EVALUATIONS = 3
SEED = 20261001
TRAIN_BASE_SEED = 2026101101
DEVELOPMENT_MICRO_BASE_SEED = 2026102101
DEVELOPMENT_STONE_BASE_SEED = 2026103101
SIMULATOR_EPISODE_RETRIES = 3
SIMULATOR_RETRY_SECONDS = 5
MAX_SCENARIO_RESAMPLES = 8
TASKS = (
    "RECOVER_CAMERA",
    "AVOID_DIGGING_TRAP",
    "REACQUIRE_STONE",
    "CLIMB_SHORE",
    "ESCAPE_HOLE",
    "AVOID_WATER",
)
INITIAL_WEIGHTS = {
    "RECOVER_CAMERA": 0.25,
    "AVOID_DIGGING_TRAP": 0.20,
    "REACQUIRE_STONE": 0.20,
    "CLIMB_SHORE": 0.15,
    "ESCAPE_HOLE": 0.10,
    "AVOID_WATER": 0.10,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def notify(message: str) -> None:
    if not TELEGRAM_ENABLED:
        return
    try:
        TelegramBot(poll_updates=False, enable_terminal_commands=False, drop_pending_updates=False).send_message(text=message)
    except Exception as error:
        print(f"Telegram notification failed: {error}", flush=True)


def retryable_episode_error(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError, BrokenPipeError, OSError)):
        return True
    if isinstance(error, TypeError) and str(error) == "a bytes-like object is required, not 'NoneType'":
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error)
    return (
        " initial state is not in water:" in message
        or " initial state unexpectedly contains water:" in message
        or message == "ESCAPE_HOLE initial state has no stable floor"
        or message.endswith(" life-state instrumentation missing")
        or message.startswith("Environment error for ")
    )


def invalid_initial_state_error(error: Exception) -> bool:
    if not isinstance(error, RuntimeError):
        return False
    message = str(error)
    return (
        " initial state is not in water:" in message
        or " initial state unexpectedly contains water:" in message
        or message == "ESCAPE_HOLE initial state has no stable floor"
        or message.endswith(" life-state instrumentation missing")
    )


def configure_ppo_model(model: SteveOnePolicy) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad = False
    targets = inject_lora(model, 32, 64.0, ("net.recurrent_layer.blocks.2.", "net.recurrent_layer.blocks.3."))
    for parameter in model.pi_head.parameters():
        parameter.requires_grad = True
    for parameter in model.value_head.parameters():
        parameter.requires_grad = True
    return {
        "variant": "upper_lora_rank32_ppo",
        "lora_rank": 32,
        "lora_targets": targets,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def clone_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
    return [value.detach().cpu().clone() for value in state]


def stack_states(states: list[list[torch.Tensor]], device: torch.device) -> list[torch.Tensor]:
    return [torch.cat([state[index] for state in states], dim=0).to(device, non_blocking=True) for index in range(len(states[0]))]


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def compute_gae(rewards: list[float], values: list[float], dones: list[bool]) -> tuple[list[float], list[float]]:
    advantages = [0.0] * len(rewards)
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if dones[index] else 1.0
        delta = rewards[index] + GAMMA * next_value * continuation - values[index]
        next_advantage = delta + GAMMA * GAE_LAMBDA * continuation * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def adaptive_weights(history: dict[str, deque]) -> dict[str, float]:
    values = {}
    for task in TASKS:
        success_rate = sum(history[task]) / len(history[task]) if history[task] else 0.0
        values[task] = INITIAL_WEIGHTS[task] * (0.5 + 1.5 * max(0.15, 1.0 - success_rate))
    total = sum(values.values())
    return {task: value / total for task, value in values.items()}


def training_potential(verifier: micro.MicroVerifier, state: dict) -> float:
    task = verifier.task
    if task in {"RECOVER_CAMERA", "REACQUIRE_STONE", "AVOID_DIGGING_TRAP", "AVOID_WATER"}:
        yaw_error, pitch_error = verifier.camera_errors(state)
        alignment = 1.0 - min(1.0, 0.5 * yaw_error / 180.0 + 0.5 * pitch_error / 90.0)
        if task == "AVOID_WATER":
            return alignment
        position = state["position"]
        distance = math.hypot(verifier.target_x - float(position["x"]), verifier.target_z - float(position["z"]))
        proximity = 1.0 - min(1.0, distance / 12.0)
        return 0.6 * alignment + 0.4 * proximity
    if task == "CLIMB_SHORE":
        height = np.clip(float(state["position"]["y"]) - float(verifier.baseline["position"]["y"]), 0.0, 1.0)
        return 0.5 * float(state["water"] == 0) + 0.5 * float(height)
    if task == "ESCAPE_HOLE":
        height = float(state["position"]["y"]) - float(verifier.baseline["position"]["y"])
        return float(np.clip(height, 0.0, 1.0))
    return 0.0


def action_tensor(button: int, camera: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "buttons": torch.tensor([[[button]]], dtype=torch.long, device=device),
        "camera": torch.tensor([[[camera]]], dtype=torch.long, device=device),
    }


def action_codes(action: dict) -> tuple[int, int]:
    return int(np.asarray(action["buttons"]).reshape(-1)[0]), int(np.asarray(action["camera"]).reshape(-1)[0])


def finalize_fragment(fragments: list[dict], fragment: dict | None) -> None:
    if fragment is not None and fragment["steps"]:
        fragments.append(fragment)


def collect_episode(model: SteveOnePolicy, reference: SteveOnePolicy, simulator, command_callback: CommandsCallback, config: dict, condition_embedding: torch.Tensor) -> tuple[list[dict], dict]:
    condition = {
        "cond_scale": CONDITION_SCALE,
        "mineclip_embeds": condition_embedding,
    }
    random.seed(config["seed"])
    np.random.seed(config["seed"] % (2**32))
    torch.manual_seed(config["seed"])
    command_callback.commands = micro.scenario_commands(config)
    simulator.seed = config["seed"]
    simulator.env.seed(config["seed"])
    observation, info = simulator.reset()
    terminated = False
    truncated = False
    for setup_step in range(1, micro.MAX_SETUP_SETTLE_STEPS + 1):
        observation, reward, terminated, truncated, info = simulator.step(simulator.noop_action())
        state = micro.state_snapshot(info)
        ready = "life_stats" in info and bool(micro.voxel_records(info.get("voxels")))
        if setup_step >= micro.SETUP_SETTLE_STEPS and ready and micro.has_stable_support(state):
            break
    baseline = micro.state_snapshot(info)
    micro.validate_initial_state(config, baseline)
    verifier = micro.MicroVerifier(config, baseline)
    router = LegacyStoneTaskRouter(verifier.reference_grade_y)
    controller = ClosedLoopOptions(simulator, pickup_implementation="old")
    recent_positions = deque(maxlen=40)
    current_state = None
    reference_state = None
    current_fragment = None
    fragments = []
    episode_steps = []
    active_category = None
    outcome = None
    outcome_details = {}
    environment_error = None
    blocked_controls = Counter()
    started = time.monotonic()
    while len(episode_steps) < int(config["timeout_steps"]) and not terminated and not truncated and environment_error is None and outcome is None:
        pre_state = micro.state_snapshot(info)
        recent_positions.append((pre_state["position"]["x"], pre_state["position"]["y"], pre_state["position"]["z"]))
        low_displacement = len(recent_positions) == recent_positions.maxlen and math.dist(recent_positions[0], recent_positions[-1]) < 0.6
        stone_mined = micro.grouped_mapping_value(pre_state["events"]["mine_block"], "stone")
        category = router.update(pre_state, len(episode_steps), low_displacement, stone_mined).category
        if category != active_category:
            finalize_fragment(fragments, current_fragment)
            current_fragment = None
            current_state = None
            reference_state = None
            active_category = category
        if current_state is None:
            current_state = model.initial_state(batch_size=1, condition=condition)
            reference_state = reference.initial_state(batch_size=1, condition=condition)
        if current_fragment is None:
            current_fragment = {
                "task": config["task"],
                "initial_state": clone_state(current_state),
                "reference_initial_state": clone_state(reference_state),
                "steps": [],
            }
        image = torch.from_numpy(observation["image"]).unsqueeze(0).unsqueeze(0).to(model.device)
        model_input = {"image": image, "condition": condition}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents, next_state = model(model_input, current_state)
            reference_latents, next_reference_state = reference(model_input, reference_state)
        policy_logits = normalized_policy_logits(latents["pi_logits"])
        reference_logits = normalized_policy_logits(reference_latents["pi_logits"])
        sampled_action = model.pi_head.sample(policy_logits, deterministic=False)
        raw_action = {name: value[0][0] for name, value in sampled_action.items()}
        gated_action, environment_action, blocked = controller.gate_world_control(raw_action)
        blocked_controls.update(blocked)
        button, camera = action_codes(gated_action)
        selected_action = action_tensor(button, camera, model.device)
        old_log_probability = float(model.pi_head.logprob(selected_action, policy_logits)[0, 0])
        reference_kl = float(model.pi_head.kl_divergence(policy_logits, reference_logits).clamp_min(0.0)[0, 0, 0])
        value = float(latents["vpred"].reshape(-1)[0])
        step_record = {
            "image": observation["image"].copy(),
            "button": button,
            "camera": camera,
            "old_log_probability": old_log_probability,
            "old_value": value,
            "reference_kl": reference_kl,
            "reward": 0.0,
            "done": False,
        }
        current_fragment["steps"].append(step_record)
        episode_steps.append(step_record)
        observation, reward, terminated, truncated, info = simulator.step(gated_action)
        environment_error = info.get("error")
        post_state = micro.state_snapshot(info)
        outcome, outcome_details = verifier.update(post_state, len(episode_steps))
        if outcome is None and (terminated or truncated):
            outcome = "FAILURE"
            outcome_details = {"reason": "ENVIRONMENT_TERMINATED" if terminated else "ENVIRONMENT_TRUNCATED"}
        shaped_reward = SHAPING_SCALE * float(np.clip(training_potential(verifier, post_state) - training_potential(verifier, pre_state), -1.0, 1.0))
        if outcome == "SUCCESS":
            shaped_reward += 1.0
        elif outcome == "FAILURE" or terminated or truncated:
            shaped_reward -= 1.0
        step_record["reward"] = shaped_reward
        step_record["done"] = outcome is not None or terminated or truncated
        current_state = next_state
        reference_state = next_reference_state
        if len(current_fragment["steps"]) >= FRAGMENT_LENGTH:
            finalize_fragment(fragments, current_fragment)
            current_fragment = None
    if environment_error is not None:
        raise RuntimeError(f"Environment error for {config['episode_id']}: {environment_error}")
    if not episode_steps:
        raise RuntimeError(f"No policy steps collected for {config['episode_id']}")
    if outcome is None:
        outcome = "FAILURE"
        outcome_details = {"reason": verifier.timeout_reason(micro.state_snapshot(info))}
        episode_steps[-1]["reward"] -= 1.0
        episode_steps[-1]["done"] = True
    finalize_fragment(fragments, current_fragment)
    advantages, returns = compute_gae(
        [step["reward"] for step in episode_steps],
        [step["old_value"] for step in episode_steps],
        [step["done"] for step in episode_steps],
    )
    for step, advantage, return_value in zip(episode_steps, advantages, returns):
        step["advantage"] = advantage
        step["return"] = return_value
    final = micro.state_snapshot(info)
    result = {
        "episode_id": config["episode_id"],
        "task": config["task"],
        "seed": config["seed"],
        "biome": config["biome"],
        "success": outcome == "SUCCESS",
        "failure_reason": None if outcome == "SUCCESS" else str(outcome_details.get("reason", verifier.timeout_reason(final))),
        "steps": len(episode_steps),
        "return": sum(step["reward"] for step in episode_steps),
        "mean_reference_kl": float(np.mean([step["reference_kl"] for step in episode_steps])),
        "blocked_controls": dict(blocked_controls),
        "diagnostics": verifier.diagnostics(final),
        "seconds": time.monotonic() - started,
    }
    return fragments, result


def collate_fragments(fragments: list[dict], condition_embeddings: dict[str, torch.Tensor], device: torch.device) -> dict:
    maximum_length = max(len(fragment["steps"]) for fragment in fragments)
    images = np.zeros((len(fragments), maximum_length, 128, 128, 3), dtype=np.uint8)
    buttons = np.zeros((len(fragments), maximum_length, 1), dtype=np.int64)
    cameras = np.full((len(fragments), maximum_length, 1), 60, dtype=np.int64)
    mask = np.zeros((len(fragments), maximum_length), dtype=np.float32)
    old_log_probability = np.zeros((len(fragments), maximum_length), dtype=np.float32)
    old_value = np.zeros((len(fragments), maximum_length), dtype=np.float32)
    advantages = np.zeros((len(fragments), maximum_length), dtype=np.float32)
    returns = np.zeros((len(fragments), maximum_length), dtype=np.float32)
    for row_index, fragment in enumerate(fragments):
        length = len(fragment["steps"])
        images[row_index, :length] = np.stack([step["image"] for step in fragment["steps"]])
        buttons[row_index, :length, 0] = [step["button"] for step in fragment["steps"]]
        cameras[row_index, :length, 0] = [step["camera"] for step in fragment["steps"]]
        mask[row_index, :length] = 1.0
        old_log_probability[row_index, :length] = [step["old_log_probability"] for step in fragment["steps"]]
        old_value[row_index, :length] = [step["old_value"] for step in fragment["steps"]]
        advantages[row_index, :length] = [step["advantage"] for step in fragment["steps"]]
        returns[row_index, :length] = [step["return"] for step in fragment["steps"]]
    return {
        "image": torch.from_numpy(images).to(device, non_blocking=True),
        "actions": {
            "buttons": torch.from_numpy(buttons).to(device, non_blocking=True),
            "camera": torch.from_numpy(cameras).to(device, non_blocking=True),
        },
        "mask": torch.from_numpy(mask).to(device, non_blocking=True),
        "old_log_probability": torch.from_numpy(old_log_probability).to(device, non_blocking=True),
        "old_value": torch.from_numpy(old_value).to(device, non_blocking=True),
        "advantage": torch.from_numpy(advantages).to(device, non_blocking=True),
        "return": torch.from_numpy(returns).to(device, non_blocking=True),
        "condition": {
            "cond_scale": CONDITION_SCALE,
            "mineclip_embeds": torch.cat([condition_embeddings[fragment["task"]] for fragment in fragments], dim=0),
        },
        "initial_state": stack_states([fragment["initial_state"] for fragment in fragments], device),
        "reference_initial_state": stack_states([fragment["reference_initial_state"] for fragment in fragments], device),
    }


def normalize_advantages(fragments: list[dict]) -> None:
    values = np.asarray([step["advantage"] for fragment in fragments for step in fragment["steps"]], dtype=np.float32)
    mean = float(values.mean())
    standard_deviation = float(values.std())
    for fragment in fragments:
        for step in fragment["steps"]:
            step["advantage"] = (step["advantage"] - mean) / max(standard_deviation, 1e-6)


def ppo_update(model: SteveOnePolicy, reference: SteveOnePolicy, optimizer: torch.optim.Optimizer, fragments: list[dict], condition_embeddings: dict[str, torch.Tensor], generator: random.Random) -> dict:
    normalize_advantages(fragments)
    totals = Counter()
    updates = 0
    stopped_for_kl = False
    model.train()
    model.mineclip.eval()
    model.prior.eval()
    for epoch in range(PPO_EPOCHS):
        generator.shuffle(fragments)
        for start in range(0, len(fragments), MINIBATCH_FRAGMENTS):
            selected = fragments[start:start + MINIBATCH_FRAGMENTS]
            batch = collate_fragments(selected, condition_embeddings, model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latents, state = model({"image": batch["image"], "condition": batch["condition"]}, batch["initial_state"])
                with torch.no_grad():
                    reference_latents, reference_state = reference({"image": batch["image"], "condition": batch["condition"]}, batch["reference_initial_state"])
                policy_logits = normalized_policy_logits(latents["pi_logits"])
                reference_logits = normalized_policy_logits(reference_latents["pi_logits"])
                new_log_probability = model.pi_head.logprob(batch["actions"], policy_logits)
                log_ratio = torch.clamp(new_log_probability - batch["old_log_probability"], -10.0, 10.0)
                ratio = log_ratio.exp()
                unclipped = -batch["advantage"] * ratio
                clipped = -batch["advantage"] * torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP)
                policy_loss = masked_mean(torch.maximum(unclipped, clipped), batch["mask"])
                value = latents["vpred"].reshape(batch["mask"].shape)
                value_clipped = batch["old_value"] + torch.clamp(value - batch["old_value"], -PPO_CLIP, PPO_CLIP)
                value_loss = 0.5 * masked_mean(torch.maximum((value - batch["return"]) ** 2, (value_clipped - batch["return"]) ** 2), batch["mask"])
                reference_kl = model.pi_head.kl_divergence(policy_logits, reference_logits).squeeze(-1).clamp_min(0.0)
                kl_loss = masked_mean(reference_kl, batch["mask"])
                entropy = masked_mean(model.pi_head.entropy(policy_logits), batch["mask"])
                loss = policy_loss + VALUE_COEFFICIENT * value_loss + KL_COEFFICIENT * kl_loss - ENTROPY_COEFFICIENT * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite PPO loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], GRADIENT_CLIP_NORM, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = masked_mean((ratio - 1.0) - log_ratio, batch["mask"])
                clip_fraction = masked_mean(((ratio - 1.0).abs() > PPO_CLIP).float(), batch["mask"])
            metrics = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "reference_kl": float(kl_loss.detach()),
                "approximate_kl": float(approximate_kl.detach()),
                "entropy": float(entropy.detach()),
                "clip_fraction": float(clip_fraction.detach()),
                "gradient_norm": float(gradient_norm),
            }
            totals.update(metrics)
            updates += 1
            if metrics["approximate_kl"] > MAX_APPROXIMATE_KL:
                stopped_for_kl = True
                break
        if stopped_for_kl:
            break
    model.eval()
    return {**{name: value / max(1, updates) for name, value in totals.items()}, "updates": updates, "stopped_for_kl": stopped_for_kl}


def bc_replay_update(model: SteveOnePolicy, reference: SteveOnePolicy, optimizer: torch.optim.Optimizer, store: WindowStore, prompt_embeddings: torch.Tensor) -> dict:
    totals = Counter()
    model.train()
    model.mineclip.eval()
    model.prior.eval()
    for update in range(BC_UPDATES_PER_ITERATION):
        batch = move_batch(store.batch(BC_BATCH_SIZE, random_crop=True, targeted_fraction=0.5), model.device)
        condition = batch_condition(batch, prompt_embeddings)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            metrics = forward_loss(model, reference, batch, condition, condition)
            loss = BC_COEFFICIENT * metrics["loss"]
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], GRADIENT_CLIP_NORM)
        optimizer.step()
        totals.update({name: float(value.detach()) for name, value in metrics.items()})
        totals["gradient_norm"] += float(gradient_norm)
    model.eval()
    return {name: value / BC_UPDATES_PER_ITERATION for name, value in totals.items()}


def materialize_model(model: SteveOnePolicy, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    replacements = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ != "LoRALinear":
            continue
        parent, attribute = module_parent(model, name)
        replacements.append((parent, attribute, module))
        setattr(parent, attribute, module.merged())
    try:
        model.save_pretrained(directory)
    finally:
        for parent, attribute, module in replacements:
            setattr(parent, attribute, module)


def preserve_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def run_micro_evaluation(model: SteveOnePolicy, iteration: int, episodes_per_task: int) -> dict:
    output_root = OUTPUT_DIRECTORY / "development" / f"iteration_{iteration:04d}" / "micro"
    results = {}
    saved_rng = preserve_rng_state()
    model.eval()
    for task in TASKS:
        task_directory = output_root / task.lower()
        task_directory.mkdir(parents=True, exist_ok=True)
        result_path = task_directory / "episodes.jsonl"
        existing = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()] if result_path.exists() else []
        rows = {row["result_id"]: row for row in existing}
        micro.TASK_NAME = task
        micro.BASE_SEED = DEVELOPMENT_MICRO_BASE_SEED
        configs = [micro.make_episode_config(index) for index in range(episodes_per_task)]
        simulator = None
        command_callback = None
        simulator_biome = None
        simulator_episodes = 0
        try:
            for config in configs:
                if config["episode_id"] in rows:
                    continue
                for attempt in range(1, SIMULATOR_EPISODE_RETRIES + 1):
                    try:
                        if simulator is None or simulator_biome != config["biome"] or simulator_episodes >= micro.MAX_EPISODES_PER_SIMULATOR:
                            if simulator is not None:
                                simulator.close()
                            command_callback = CommandsCallback([])
                            simulator = micro.create_simulator(config, command_callback)
                            simulator_biome = config["biome"]
                            simulator_episodes = 0
                        row = micro.run_episode(model, simulator, command_callback, config, task_directory)
                        break
                    except Exception as error:
                        if not retryable_episode_error(error):
                            raise
                        if simulator is not None:
                            simulator.close()
                        simulator = None
                        simulator_biome = None
                        simulator_episodes = 0
                        if attempt == SIMULATOR_EPISODE_RETRIES:
                            raise
                        time.sleep(SIMULATOR_RETRY_SECONDS)
                rows[row["result_id"]] = row
                simulator_episodes += 1
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        finally:
            if simulator is not None:
                simulator.close()
        ordered = [rows[config["episode_id"]] for config in configs]
        successes = sum(row["success"] for row in ordered)
        results[task] = {"episodes": len(ordered), "successes": successes, "success_rate": successes / len(ordered)}
    restore_rng_state(saved_rng)
    atomic_json(output_root / "summary.json", results)
    return results


def run_stone_evaluation(model: SteveOnePolicy, iteration: int, episodes: int) -> dict:
    output_directory = OUTPUT_DIRECTORY / "development" / f"iteration_{iteration:04d}" / "normal_stone"
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "episodes.jsonl"
    existing = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()] if result_path.exists() else []
    rows = {row["result_id"]: row for row in existing}
    saved_rng = preserve_rng_state()
    stone.EVENT_DRIVEN_TASKS = True
    stone.ROUTER_IMPLEMENTATION = "old"
    stone.PICKUP_IMPLEMENTATION = "old"
    stone.BASE_SEED = DEVELOPMENT_STONE_BASE_SEED
    stone.MANIFEST_KIND = "general_natural"
    configs = [stone.make_episode_config(index) for index in range(episodes)]
    simulator = None
    command_callback = None
    simulator_biome = None
    simulator_episodes = 0
    model.eval()
    try:
        for config in configs:
            if config["episode_id"] in rows:
                continue
            for attempt in range(1, SIMULATOR_EPISODE_RETRIES + 1):
                try:
                    if simulator is None or simulator_biome != config["biome"] or simulator_episodes >= stone.MAX_EPISODES_PER_SIMULATOR:
                        if simulator is not None:
                            simulator.close()
                        command_callback = CommandsCallback([])
                        simulator = stone.create_simulator(config, command_callback)
                        simulator_biome = config["biome"]
                        simulator_episodes = 0
                    row = stone.run_episode(model, simulator, command_callback, config, output_directory, max_steps=stone.MAX_STEPS)
                    break
                except Exception as error:
                    if not retryable_episode_error(error):
                        raise
                    if simulator is not None:
                        simulator.close()
                    simulator = None
                    simulator_biome = None
                    simulator_episodes = 0
                    if attempt == SIMULATOR_EPISODE_RETRIES:
                        raise
                    time.sleep(SIMULATOR_RETRY_SECONDS)
            rows[row["result_id"]] = row
            simulator_episodes += 1
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        if simulator is not None:
            simulator.close()
    restore_rng_state(saved_rng)
    ordered = [rows[config["episode_id"]] for config in configs]
    successes = sum(row["success"] for row in ordered)
    result = {"episodes": len(ordered), "successes": successes, "success_rate": successes / len(ordered)}
    atomic_json(output_directory / "summary_compact.json", result)
    return result


def development_evaluation(model: SteveOnePolicy, iteration: int) -> dict:
    micro_results = run_micro_evaluation(model, iteration, DEVELOPMENT_MICRO_EPISODES)
    stone_result = run_stone_evaluation(model, iteration, DEVELOPMENT_STONE_EPISODES)
    weak_mean = float(np.mean([micro_results[task]["success_rate"] for task in TASKS]))
    result = {
        "iteration": iteration,
        "micro": micro_results,
        "weak_mean": weak_mean,
        "normal_stone": stone_result,
        "created_at": utc_now(),
    }
    atomic_json(OUTPUT_DIRECTORY / "development" / f"iteration_{iteration:04d}" / "summary.json", result)
    return result


def behavioral_score(evaluation: dict, baseline: dict) -> float:
    normal_regression = max(0.0, baseline["normal_stone"]["success_rate"] - evaluation["normal_stone"]["success_rate"])
    return evaluation["weak_mean"] + 0.5 * evaluation["normal_stone"]["success_rate"] - 2.0 * normal_regression


def improvement_count(evaluation: dict, baseline: dict, threshold: float = 0.10) -> int:
    return sum(evaluation["micro"][task]["success_rate"] >= baseline["micro"][task]["success_rate"] + threshold for task in TASKS)


def checkpoint_payload(model: SteveOnePolicy, optimizer: torch.optim.Optimizer, iteration: int, episode_index: int, history: dict[str, deque], baseline_evaluation: dict, best: dict, stale_evaluations: int, degraded_evaluations: int, generator: random.Random, store: WindowStore) -> dict:
    return {
        "iteration": iteration,
        "episode_index": episode_index,
        "trainable_state": trainable_state(model),
        "optimizer_state": optimizer.state_dict(),
        "history": {task: list(values) for task, values in history.items()},
        "baseline_evaluation": baseline_evaluation,
        "best": best,
        "stale_evaluations": stale_evaluations,
        "degraded_evaluations": degraded_evaluations,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "generator_rng_state": generator.getstate(),
        "store_rng_state": store.rng.getstate(),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = SteveOnePolicy.from_pretrained(REFERENCE_CHECKPOINT).to(device)
    trainable_summary = configure_ppo_model(model)
    reference = SteveOnePolicy.from_pretrained(REFERENCE_CHECKPOINT).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    model.eval()
    generator = random.Random(SEED + 17)
    prompt_bank = PromptBank()
    store = WindowStore("train", SEED + 23, prompt_bank)
    with torch.no_grad():
        condition_embeddings = {
            task: model.prepare_condition({"cond_scale": CONDITION_SCALE, "text": micro.TASK_PROMPTS[task]}, deterministic=False)["mineclip_embeds"].detach()
            for task in TASKS
        }
        prompt_embeddings = model.prepare_condition({"cond_scale": CONDITION_SCALE, "text": prompt_bank.prompts}, deterministic=False)["mineclip_embeds"].detach()
    policy_parameters = []
    value_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("value_head."):
            value_parameters.append(parameter)
        else:
            policy_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_parameters, "lr": POLICY_LEARNING_RATE},
            {"params": value_parameters, "lr": VALUE_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
        fused=True,
    )
    resolved = {
        **trainable_summary,
        "run_id": RUN_ID,
        "output_directory": str(OUTPUT_DIRECTORY),
        "reference_checkpoint": str(REFERENCE_CHECKPOINT),
        "reference_checkpoint_sha256": sha256((REFERENCE_CHECKPOINT / "model.safetensors").read_bytes()).hexdigest(),
        "tasks": list(TASKS),
        "initial_curriculum": INITIAL_WEIGHTS,
        "max_iterations": MAX_ITERATIONS,
        "episodes_per_iteration": EPISODES_PER_ITERATION,
        "evaluation_interval": EVALUATION_INTERVAL,
        "development_micro_episodes": DEVELOPMENT_MICRO_EPISODES,
        "development_stone_episodes": DEVELOPMENT_STONE_EPISODES,
        "fragment_length": FRAGMENT_LENGTH,
        "ppo_epochs": PPO_EPOCHS,
        "minibatch_fragments": MINIBATCH_FRAGMENTS,
        "policy_learning_rate": POLICY_LEARNING_RATE,
        "value_learning_rate": VALUE_LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "ppo_clip": PPO_CLIP,
        "kl_coefficient": KL_COEFFICIENT,
        "entropy_coefficient": ENTROPY_COEFFICIENT,
        "bc_coefficient": BC_COEFFICIENT,
        "bc_updates_per_iteration": BC_UPDATES_PER_ITERATION,
        "shaping_scale": SHAPING_SCALE,
        "router": "old",
        "pickup": "old",
        "system_0_modified": False,
        "system_2_modified": False,
        "gemini_reward": False,
        "seed": SEED,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "smoke_mode": SMOKE_MODE,
    }
    atomic_json(OUTPUT_DIRECTORY / "resolved_config.json", resolved)
    last_checkpoint = OUTPUT_DIRECTORY / "last_training_state.pt"
    metrics_path = OUTPUT_DIRECTORY / "metrics.jsonl"
    run_state_path = OUTPUT_DIRECTORY / "run_state.json"
    history = {task: deque(maxlen=20) for task in TASKS}
    baseline_evaluation = None
    best = None
    stale_evaluations = 0
    degraded_evaluations = 0
    start_iteration = 1
    episode_index = 0
    resumed = last_checkpoint.exists()
    if resumed:
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        load_trainable_state(model, checkpoint["trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = {task: deque(checkpoint["history"][task], maxlen=20) for task in TASKS}
        baseline_evaluation = checkpoint["baseline_evaluation"]
        best = checkpoint["best"]
        stale_evaluations = checkpoint["stale_evaluations"]
        degraded_evaluations = checkpoint["degraded_evaluations"]
        start_iteration = checkpoint["iteration"] + 1
        episode_index = checkpoint["episode_index"]
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        generator.setstate(checkpoint["generator_rng_state"])
        store.rng.setstate(checkpoint["store_rng_state"])
    else:
        metrics_path.write_text("", encoding="utf-8")
    atomic_json(run_state_path, {"status": "running", "phase": "smoke" if SMOKE_MODE else "initial_evaluation", "pid": os.getpid(), "host": socket.gethostname(), "resumed": resumed, "start_iteration": start_iteration, "updated_at": utc_now()})
    if SMOKE_MODE:
        micro.TASK_NAME = "RECOVER_CAMERA"
        micro.BASE_SEED = TRAIN_BASE_SEED
        config = micro.make_episode_config(0)
        config["timeout_steps"] = 24
        command_callback = CommandsCallback([])
        simulator = micro.create_simulator(config, command_callback)
        try:
            fragments, rollout = collect_episode(model, reference, simulator, command_callback, config, condition_embeddings["RECOVER_CAMERA"])
        finally:
            simulator.close()
        ppo_metrics = ppo_update(model, reference, optimizer, fragments, condition_embeddings, generator)
        bc_metrics = bc_replay_update(model, reference, optimizer, store, prompt_embeddings)
        payload = checkpoint_payload(model, optimizer, 1, 1, history, {"smoke": True}, {"smoke": True}, 0, 0, generator, store)
        atomic_torch_save(payload, last_checkpoint)
        loaded = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        load_trainable_state(model, loaded["trainable_state"])
        materialize_model(model, OUTPUT_DIRECTORY / "smoke_model")
        atomic_json(OUTPUT_DIRECTORY / "summary.json", {"status": "completed", "rollout": rollout, "ppo": ppo_metrics, "bc": bc_metrics, "checkpoint_resume": True, "materialized_model": str(OUTPUT_DIRECTORY / "smoke_model")})
        atomic_json(run_state_path, {"status": "completed", "phase": "smoke", "pid": os.getpid(), "updated_at": utc_now()})
        return
    if baseline_evaluation is None:
        baseline_evaluation = development_evaluation(model, 0)
        baseline_score = behavioral_score(baseline_evaluation, baseline_evaluation)
        checkpoint_directory = OUTPUT_DIRECTORY / "candidate_models" / "iteration_0000"
        materialize_model(model, checkpoint_directory)
        best = {"iteration": 0, "score": baseline_score, "evaluation": baseline_evaluation, "checkpoint_directory": str(checkpoint_directory)}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"iteration": 0, "split": "development", **baseline_evaluation, "behavioral_score": baseline_score}) + "\n")
    notify(f"MineStudio targeted System 1 PPO started\nrun={RUN_ID}\nreference=upper_lora_r32_step6400\nrouter=old\npickup=old\noutput={OUTPUT_DIRECTORY}")
    started = time.monotonic()
    stop_reason = "max_iterations"
    completed_iteration = start_iteration - 1
    for iteration in range(start_iteration, MAX_ITERATIONS + 1):
        completed_iteration = iteration
        iteration_started = time.monotonic()
        weights = adaptive_weights(history)
        sampled_tasks = generator.choices(list(TASKS), weights=[weights[task] for task in TASKS], k=EPISODES_PER_ITERATION)
        configs = []
        for task in sampled_tasks:
            micro.TASK_NAME = task
            micro.BASE_SEED = TRAIN_BASE_SEED
            configs.append(micro.make_episode_config(episode_index))
            episode_index += 1
        fragments = []
        rollout_rows = []
        simulator = None
        command_callback = None
        simulator_biome = None
        simulator_episodes = 0
        try:
            for config in configs:
                attempt = 0
                scenario_resamples = 0
                while True:
                    try:
                        if simulator is None or simulator_biome != config["biome"] or simulator_episodes >= micro.MAX_EPISODES_PER_SIMULATOR:
                            if simulator is not None:
                                simulator.close()
                            command_callback = CommandsCallback([])
                            simulator = micro.create_simulator(config, command_callback)
                            simulator_biome = config["biome"]
                            simulator_episodes = 0
                        episode_fragments, rollout = collect_episode(model, reference, simulator, command_callback, config, condition_embeddings[config["task"]])
                        break
                    except Exception as error:
                        if not retryable_episode_error(error):
                            raise
                        attempt += 1
                        if simulator is not None:
                            simulator.close()
                        simulator = None
                        simulator_biome = None
                        simulator_episodes = 0
                        if attempt >= SIMULATOR_EPISODE_RETRIES and invalid_initial_state_error(error) and scenario_resamples < MAX_SCENARIO_RESAMPLES:
                            micro.TASK_NAME = config["task"]
                            micro.BASE_SEED = TRAIN_BASE_SEED
                            config = micro.make_episode_config(episode_index)
                            episode_index += 1
                            attempt = 0
                            scenario_resamples += 1
                        elif attempt >= SIMULATOR_EPISODE_RETRIES:
                            raise
                        time.sleep(SIMULATOR_RETRY_SECONDS)
                fragments.extend(episode_fragments)
                rollout_rows.append(rollout)
                history[config["task"]].append(bool(rollout["success"]))
                simulator_episodes += 1
        finally:
            if simulator is not None:
                simulator.close()
        ppo_metrics = ppo_update(model, reference, optimizer, fragments, condition_embeddings, generator)
        bc_metrics = bc_replay_update(model, reference, optimizer, store, prompt_embeddings)
        training_record = {
            "iteration": iteration,
            "split": "train",
            "episodes": len(rollout_rows),
            "success_rate": sum(row["success"] for row in rollout_rows) / len(rollout_rows),
            "task_counts": dict(Counter(row["task"] for row in rollout_rows)),
            "task_successes": dict(Counter(row["task"] for row in rollout_rows if row["success"])),
            "mean_episode_return": float(np.mean([row["return"] for row in rollout_rows])),
            "mean_rollout_reference_kl": float(np.mean([row["mean_reference_kl"] for row in rollout_rows])),
            "curriculum": weights,
            "ppo": ppo_metrics,
            "bc": bc_metrics,
            "iteration_seconds": time.monotonic() - iteration_started,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(training_record) + "\n")
        evaluation = None
        if iteration % EVALUATION_INTERVAL == 0 or iteration == MAX_ITERATIONS:
            evaluation = development_evaluation(model, iteration)
            score = behavioral_score(evaluation, baseline_evaluation)
            evaluation_record = {"iteration": iteration, "split": "development", **evaluation, "behavioral_score": score, "improved_tasks": improvement_count(evaluation, baseline_evaluation)}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation_record) + "\n")
            snapshot_path = OUTPUT_DIRECTORY / "snapshots" / f"iteration_{iteration:04d}_trainable.pt"
            atomic_torch_save(trainable_state(model), snapshot_path)
            normal_floor = baseline_evaluation["normal_stone"]["success_rate"] - 0.05
            if score > best["score"] and evaluation["normal_stone"]["success_rate"] >= normal_floor:
                checkpoint_directory = OUTPUT_DIRECTORY / "candidate_models" / f"iteration_{iteration:04d}"
                materialize_model(model, checkpoint_directory)
                best = {"iteration": iteration, "score": score, "evaluation": evaluation, "checkpoint_directory": str(checkpoint_directory)}
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if evaluation["normal_stone"]["success_rate"] < baseline_evaluation["normal_stone"]["success_rate"] - 0.10:
                degraded_evaluations += 1
            else:
                degraded_evaluations = 0
            clear_improvement = evaluation["weak_mean"] >= baseline_evaluation["weak_mean"] + 0.10 and improvement_count(evaluation, baseline_evaluation) >= 3 and evaluation["normal_stone"]["successes"] >= baseline_evaluation["normal_stone"]["successes"] - 1
            if iteration >= MINIMUM_ITERATIONS and clear_improvement:
                stop_reason = "clear_behavioral_improvement"
            elif iteration >= MINIMUM_ITERATIONS and degraded_evaluations >= 2:
                stop_reason = "ordinary_behavior_degrading"
            elif iteration >= MINIMUM_ITERATIONS and stale_evaluations >= PLATEAU_EVALUATIONS:
                stop_reason = "behavioral_plateau"
        payload = checkpoint_payload(model, optimizer, iteration, episode_index, history, baseline_evaluation, best, stale_evaluations, degraded_evaluations, generator, store)
        atomic_torch_save(payload, last_checkpoint)
        elapsed = time.monotonic() - started
        estimated_remaining = elapsed / max(1, iteration - start_iteration + 1) * max(0, MAX_ITERATIONS - iteration)
        atomic_json(run_state_path, {
            "status": "running",
            "phase": "training",
            "pid": os.getpid(),
            "iteration": iteration,
            "max_iterations": MAX_ITERATIONS,
            "episode_index": episode_index,
            "best_iteration": best["iteration"],
            "best_score": best["score"],
            "latest_development": evaluation,
            "stop_reason": stop_reason if stop_reason != "max_iterations" else None,
            "estimated_remaining_seconds": estimated_remaining,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "updated_at": utc_now(),
        })
        print(json.dumps({"status": "iteration_complete", **training_record, "development": evaluation, "best_iteration": best["iteration"], "stop_reason": stop_reason if stop_reason != "max_iterations" else None}), flush=True)
        if stop_reason != "max_iterations":
            break
    summary = {
        "status": "completed",
        "run_id": RUN_ID,
        "completed_iterations": completed_iteration,
        "stop_reason": stop_reason,
        "baseline_evaluation": baseline_evaluation,
        "best": best,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "reference_checkpoint": str(REFERENCE_CHECKPOINT),
        "training_curve": str(metrics_path),
    }
    atomic_json(OUTPUT_DIRECTORY / "summary.json", summary)
    atomic_json(run_state_path, {"status": "completed", "phase": "training", "iteration": completed_iteration, "stop_reason": stop_reason, "best_iteration": best["iteration"], "best_checkpoint_directory": best["checkpoint_directory"], "updated_at": utc_now()})
    notify(f"MineStudio targeted System 1 PPO training completed\nrun={RUN_ID}\niterations={completed_iteration}\nstop={stop_reason}\nbest_iteration={best['iteration']}\ncheckpoint={best['checkpoint_directory']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        atomic_json(OUTPUT_DIRECTORY / "run_state.json", {"status": "failed", "phase": "training", "pid": os.getpid(), "host": socket.gethostname(), "error": str(error), "updated_at": utc_now()})
        notify(f"MineStudio targeted System 1 PPO failed\nrun={RUN_ID}\nerror={error}\noutput={OUTPUT_DIRECTORY}")
        raise
