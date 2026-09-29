import json
import math
import os
import random
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from minestudio.models import SteveOnePolicy
from telegram_training_helper import TelegramBot


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
BASE_CHECKPOINT = PROJECT_DIRECTORY / "checkpoints" / "steve_one_official"
BASELINE_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_acquisition" / "stage0_baseline_v1"
DATASET_DIRECTORY = PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "dataset_v1"
TASK_BANK_PATH = Path(os.environ.get("MINESTUDIO_BC_TASK_BANK", DATASET_DIRECTORY / "gemini_task_bank.json"))
OUTPUT_ROOT = Path(os.environ.get("MINESTUDIO_BC_OUTPUT_DIRECTORY", PROJECT_DIRECTORY / "output" / "stone_recovery_bc" / "training_v1"))
VARIANT = os.environ.get("MINESTUDIO_BC_VARIANT", "action_head")
MAX_STEPS = int(os.environ.get("MINESTUDIO_BC_MAX_STEPS", "400"))
BATCH_SIZE = int(os.environ.get("MINESTUDIO_BC_BATCH_SIZE", "4"))
SEQUENCE_LENGTH = int(os.environ.get("MINESTUDIO_BC_SEQUENCE_LENGTH", "64"))
VALIDATION_INTERVAL = int(os.environ.get("MINESTUDIO_BC_VALIDATION_INTERVAL", "50"))
VALIDATION_BATCHES = int(os.environ.get("MINESTUDIO_BC_VALIDATION_BATCHES", "20"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("MINESTUDIO_BC_EARLY_STOPPING_PATIENCE", "4"))
TELEGRAM_ENABLED = os.environ.get("MINESTUDIO_BC_TELEGRAM", "0") == "1"
SNAPSHOT_STEPS = {int(value) for value in os.environ.get("MINESTUDIO_BC_SNAPSHOT_STEPS", "").split(",") if value.strip()}
TARGETED_FRACTION = 0.50
REFERENCE_KL_WEIGHT = 0.02
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
CONDITION_SCALE = 6.0
SEED = 20260916
LORA_DROPOUT = 0.05
LEARNING_RATES = {
    "action_head": 2e-6,
    "action_head_recurrent": 2e-7,
    "upper_lora": 2e-6,
    "upper_lora_rank32": 1e-6,
    "recurrent_upper_lora": 1e-6,
    "upper_blocks_full": 1e-7,
    "policy_no_visual_full": 5e-8,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def atomic_torch_save(value: dict, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def notify(text: str) -> None:
    if not TELEGRAM_ENABLED:
        return
    try:
        TelegramBot(poll_updates=False, enable_terminal_commands=False, drop_pending_updates=False).send_message(text=text)
    except Exception as error:
        print(f"Telegram notification failed: {error}", flush=True)


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.lora_a = torch.nn.Linear(base.in_features, rank, bias=False).to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b = torch.nn.Linear(rank, base.out_features, bias=False).to(device=base.weight.device, dtype=base.weight.dtype)
        self.scale = alpha / rank
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.lora_b(self.lora_a(self.dropout(value))) * self.scale

    def merged(self) -> torch.nn.Linear:
        result = torch.nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        result = result.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        update = self.lora_b.weight @ self.lora_a.weight
        result.weight.data.copy_(self.base.weight.data + update.to(self.base.weight.dtype) * self.scale)
        if self.base.bias is not None:
            result.bias.data.copy_(self.base.bias.data)
        return result


def module_parent(root: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def lora_configuration(variant: str) -> tuple[int, float, tuple[str, ...]]:
    if variant == "upper_lora":
        return 8, 16.0, ("net.recurrent_layer.blocks.2.", "net.recurrent_layer.blocks.3.")
    if variant == "upper_lora_rank32":
        return 32, 64.0, ("net.recurrent_layer.blocks.2.", "net.recurrent_layer.blocks.3.")
    if variant == "recurrent_upper_lora":
        return 8, 16.0, ("net.recurrent_layer.blocks.",)
    raise ValueError(f"Variant {variant} does not use LoRA")


def inject_lora(model: SteveOnePolicy, rank: int, alpha: float, prefixes: tuple[str, ...]) -> list[str]:
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.startswith(prefixes):
            continue
        if name.endswith("r_layer"):
            continue
        targets.append(name)
    for name in targets:
        parent, attribute = module_parent(model, name)
        setattr(parent, attribute, LoRALinear(getattr(parent, attribute), rank, alpha, LORA_DROPOUT))
    return targets


def merge_lora(model: SteveOnePolicy) -> None:
    targets = [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    for name in targets:
        parent, attribute = module_parent(model, name)
        setattr(parent, attribute, getattr(parent, attribute).merged())


def configure_trainable(model: SteveOnePolicy, variant: str) -> dict:
    for parameter in model.parameters():
        parameter.requires_grad = False
    lora_targets = []
    if variant == "action_head":
        for parameter in model.pi_head.parameters():
            parameter.requires_grad = True
    elif variant == "action_head_recurrent":
        for parameter in model.pi_head.parameters():
            parameter.requires_grad = True
        for parameter in model.net.recurrent_layer.parameters():
            parameter.requires_grad = True
    elif variant in {"upper_lora", "upper_lora_rank32", "recurrent_upper_lora"}:
        rank, alpha, prefixes = lora_configuration(variant)
        lora_targets = inject_lora(model, rank, alpha, prefixes)
        for parameter in model.pi_head.parameters():
            parameter.requires_grad = True
    elif variant == "upper_blocks_full":
        for block in model.net.recurrent_layer.blocks[2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in model.pi_head.parameters():
            parameter.requires_grad = True
    elif variant == "policy_no_visual_full":
        for component in (
            model.net.recurrent_layer,
            model.net.lastlayer,
            model.net.final_ln,
            model.net.mineclip_embed_linear,
            model.pi_head,
        ):
            for parameter in component.parameters():
                parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown variant: {variant}")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    lora_rank = lora_configuration(variant)[0] if "lora" in variant else None
    return {"variant": variant, "trainable_parameters": trainable, "total_parameters": total, "lora_rank": lora_rank, "lora_targets": lora_targets}


class PromptBank:
    def __init__(self):
        payload = json.loads(TASK_BANK_PATH.read_text(encoding="utf-8"))
        self.model = payload["model"]
        self.tasks = payload["tasks"]
        self.prompts = sorted({prompt for prompts in self.tasks.values() for prompt in prompts})
        self.index = {prompt: index for index, prompt in enumerate(self.prompts)}
        self.sha256 = sha256(TASK_BANK_PATH.read_bytes()).hexdigest()

    def prompt_index(self, category: str, rng: random.Random) -> int:
        prompts = self.tasks[category]
        return self.index[prompts[rng.randrange(len(prompts))]]


class WindowStore:
    def __init__(self, split: str, seed: int, prompt_bank: PromptBank):
        records = [json.loads(line) for line in (DATASET_DIRECTORY / "windows.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.targeted = [record for record in records if record["split"] == split and record["source_group"] == "targeted_recovery"]
        self.regularization = [record for record in records if record["split"] == split and record["source_group"] == "stone_regularization"]
        if not self.targeted or not self.regularization:
            raise RuntimeError(f"Missing {split} windows")
        self.rng = random.Random(seed)
        self.split = split
        self.prompt_bank = prompt_bank

    def sample_record(self, targeted: bool) -> dict:
        records = self.targeted if targeted else self.regularization
        return records[self.rng.randrange(len(records))]

    def load_sequence(self, record: dict, random_crop: bool) -> dict:
        path = BASELINE_DIRECTORY / record["trajectory_npz"]
        with np.load(path, allow_pickle=False) as arrays:
            start = record["start"]
            end = record["end"]
            available = end - start
            if available > SEQUENCE_LENGTH:
                offset = self.rng.randrange(available - SEQUENCE_LENGTH + 1) if random_crop else (available - SEQUENCE_LENGTH) // 2
                start += offset
                end = start + SEQUENCE_LENGTH
            image = arrays["rgb"][start:end].copy()
            buttons = arrays["agent_buttons"][start:end].copy()
            camera = arrays["agent_camera"][start:end].copy()
        valid = len(image)
        if valid < SEQUENCE_LENGTH:
            padding = SEQUENCE_LENGTH - valid
            image = np.pad(image, ((0, padding), (0, 0), (0, 0), (0, 0)))
            buttons = np.pad(buttons, (0, padding))
            camera = np.pad(camera, (0, padding), constant_values=60)
        mask = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
        mask[:valid] = 1.0
        return {
            "image": image,
            "buttons": buttons[:, None],
            "camera": camera[:, None],
            "mask": mask,
            "targeted": record["source_group"] == "targeted_recovery",
            "window_id": record["window_id"],
            "prompt_index": self.prompt_bank.prompt_index(record["task_category"], self.rng),
            "task_category": record["task_category"],
        }

    def batch(self, batch_size: int, random_crop: bool, targeted_fraction: float = TARGETED_FRACTION) -> dict:
        targeted_count = round(batch_size * targeted_fraction)
        targeted_flags = [True] * targeted_count + [False] * (batch_size - targeted_count)
        self.rng.shuffle(targeted_flags)
        rows = [self.load_sequence(self.sample_record(flag), random_crop) for flag in targeted_flags]
        return {
            "image": torch.from_numpy(np.stack([row["image"] for row in rows])),
            "buttons": torch.from_numpy(np.stack([row["buttons"] for row in rows])).long(),
            "camera": torch.from_numpy(np.stack([row["camera"] for row in rows])).long(),
            "mask": torch.from_numpy(np.stack([row["mask"] for row in rows])),
            "targeted": torch.tensor([row["targeted"] for row in rows], dtype=torch.bool),
            "window_ids": [row["window_id"] for row in rows],
            "prompt_indices": torch.tensor([row["prompt_index"] for row in rows], dtype=torch.long),
            "task_categories": [row["task_category"] for row in rows],
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def normalized_policy_logits(logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.log_softmax(value.float(), dim=-1) for name, value in logits.items()}


def batch_condition(batch: dict, prompt_embeddings: torch.Tensor) -> dict:
    return {
        "cond_scale": CONDITION_SCALE,
        "mineclip_embeds": prompt_embeddings.index_select(0, batch["prompt_indices"]),
    }


def forward_loss(model: SteveOnePolicy, reference: SteveOnePolicy, batch: dict, condition: dict, reference_condition: dict) -> dict:
    model_input = {"image": batch["image"], "condition": condition}
    latents, _ = model(model_input, None)
    policy_logits = normalized_policy_logits(latents["pi_logits"])
    actions = {"buttons": batch["buttons"], "camera": batch["camera"]}
    log_prob = model.pi_head.logprob(actions, policy_logits, return_dict=True)
    button_loss = -masked_mean(log_prob["buttons"], batch["mask"])
    camera_mask = batch["mask"] * (batch["camera"].squeeze(-1) != 60).float()
    camera_loss = -masked_mean(log_prob["camera"], camera_mask)
    regularization_rows = (~batch["targeted"]).float()[:, None]
    kl_mask = batch["mask"] * regularization_rows
    with torch.no_grad():
        reference_latents, _ = reference({"image": batch["image"], "condition": reference_condition}, None)
        reference_logits = normalized_policy_logits(reference_latents["pi_logits"])
    kl = model.pi_head.kl_divergence(policy_logits, reference_logits)
    while kl.ndim > kl_mask.ndim:
        kl = kl.squeeze(-1)
    kl_loss = masked_mean(kl, kl_mask)
    loss = button_loss + camera_loss + REFERENCE_KL_WEIGHT * kl_loss
    with torch.no_grad():
        button_prediction = policy_logits["buttons"].argmax(dim=-1).squeeze(-1)
        camera_prediction = policy_logits["camera"].argmax(dim=-1).squeeze(-1)
        button_target = batch["buttons"].squeeze(-1)
        camera_target = batch["camera"].squeeze(-1)
        button_accuracy = masked_mean((button_prediction == button_target).float(), batch["mask"])
        camera_accuracy = masked_mean((camera_prediction == camera_target).float(), camera_mask)
    return {
        "loss": loss,
        "button_loss": button_loss,
        "camera_loss": camera_loss,
        "reference_kl": kl_loss,
        "button_accuracy": button_accuracy,
        "camera_accuracy": camera_accuracy,
    }


def validation_metrics(model: SteveOnePolicy, reference: SteveOnePolicy, store: WindowStore, prompt_embeddings: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    store.rng.seed(SEED + 1)
    totals = {"targeted": Counter(), "regularization": Counter()}
    with torch.no_grad():
        for group, targeted_fraction in (("targeted", 1.0), ("regularization", 0.0)):
            for _ in range(VALIDATION_BATCHES):
                batch = move_batch(store.batch(BATCH_SIZE, random_crop=False, targeted_fraction=targeted_fraction), device)
                condition = batch_condition(batch, prompt_embeddings)
                reference_condition = batch_condition(batch, prompt_embeddings)
                metrics = forward_loss(model, reference, batch, condition, reference_condition)
                for name, value in metrics.items():
                    totals[group][name] += float(value)
    model.train()
    model.mineclip.eval()
    model.prior.eval()
    result = {}
    for group, values in totals.items():
        for name, value in values.items():
            result[f"{group}_{name}"] = value / VALIDATION_BATCHES
    result["loss"] = 0.5 * (result["targeted_loss"] + result["regularization_loss"])
    return result


def trainable_state(model: SteveOnePolicy) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_trainable_state(model: SteveOnePolicy, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def main() -> None:
    if VARIANT not in LEARNING_RATES:
        raise ValueError(f"Unsupported variant {VARIANT}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    output_directory = OUTPUT_ROOT / VARIANT
    output_directory.mkdir(parents=True, exist_ok=True)
    model = SteveOnePolicy.from_pretrained(BASE_CHECKPOINT).to(device)
    trainable_summary = configure_trainable(model, VARIANT)
    reference = SteveOnePolicy.from_pretrained(BASE_CHECKPOINT).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    torch.manual_seed(SEED)
    prompt_bank = PromptBank()
    with torch.no_grad():
        prompt_embeddings = model.prepare_condition(
            {"cond_scale": CONDITION_SCALE, "text": prompt_bank.prompts},
            deterministic=False,
        )["mineclip_embeds"].detach().clone()
    model.train()
    model.mineclip.eval()
    model.prior.eval()
    train_store = WindowStore("train", SEED, prompt_bank)
    validation_store = WindowStore("validation", SEED + 1, prompt_bank)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATES[VARIANT], weight_decay=WEIGHT_DECAY, fused=True)
    metrics_path = output_directory / "metrics.jsonl"
    last_checkpoint_path = output_directory / "last_training_state.pt"
    best_checkpoint_path = output_directory / "best_trainable.pt"
    run_state_path = output_directory / "run_state.json"
    resolved_configuration = {
        "variant": VARIANT,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "validation_interval": VALIDATION_INTERVAL,
        "validation_batches": VALIDATION_BATCHES,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "snapshot_steps": sorted(SNAPSHOT_STEPS),
        "targeted_fraction": TARGETED_FRACTION,
        "reference_kl_weight": REFERENCE_KL_WEIGHT,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "condition_scale": CONDITION_SCALE,
        "task_conditioning": "per-window frozen Gemini local objective",
        "task_bank": str(TASK_BANK_PATH),
        "task_bank_sha256": prompt_bank.sha256,
        "task_bank_model": prompt_bank.model,
        "task_prompt_count": len(prompt_bank.prompts),
        "seed": SEED,
        "learning_rate": LEARNING_RATES[VARIANT],
        "base_checkpoint": str(BASE_CHECKPOINT),
        "dataset_directory": str(DATASET_DIRECTORY),
        "output_directory": str(output_directory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
    }
    atomic_json(output_directory / "resolved_config.json", resolved_configuration)
    best_validation_loss = float("inf")
    best_step = 0
    best_state = None
    stale_validations = 0
    start_step = 1
    started = time.monotonic()
    resumed = last_checkpoint_path.is_file()
    if resumed:
        checkpoint = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)
        load_trainable_state(model, checkpoint["trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_validation_loss = checkpoint["best_validation_loss"]
        best_step = checkpoint["best_step"]
        stale_validations = checkpoint["stale_validations"]
        start_step = checkpoint["step"] + 1
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        train_store.rng.setstate(checkpoint["train_store_rng_state"])
        validation_store.rng.setstate(checkpoint["validation_store_rng_state"])
        initial_validation = checkpoint["initial_validation"]
        best_state = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
    else:
        metrics_path.write_text("", encoding="utf-8")
        initial_validation = validation_metrics(model, reference, validation_store, prompt_embeddings, device)
        best_validation_loss = initial_validation["loss"]
        best_state = trainable_state(model)
        atomic_torch_save(best_state, best_checkpoint_path)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": 0, "split": "validation", **initial_validation}) + "\n")
        print(json.dumps({"status": "initial_validation", "variant": VARIANT, **trainable_summary, **initial_validation}), flush=True)
    atomic_json(run_state_path, {"status": "running", "variant": VARIANT, "pid": os.getpid(), "host": socket.gethostname(), "resumed": resumed, "start_step": start_step, "updated_at": utc_now()})
    notify(f"MineStudio stone recovery BC started\nvariant={VARIANT}\nresumed={resumed}\nstart_step={start_step}\noutput={output_directory}")
    step = start_step - 1
    for step in range(start_step, MAX_STEPS + 1):
        step_started = time.monotonic()
        batch = move_batch(train_store.batch(BATCH_SIZE, random_crop=True), device)
        condition = batch_condition(batch, prompt_embeddings)
        reference_condition = batch_condition(batch, prompt_embeddings)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            metrics = forward_loss(model, reference, batch, condition, reference_condition)
        metrics["loss"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, GRADIENT_CLIP_NORM)
        optimizer.step()
        step_seconds = time.monotonic() - step_started
        train_record = {"step": step, "split": "train", "gradient_norm": float(gradient_norm), "step_seconds": step_seconds, "sequences_per_second": BATCH_SIZE / step_seconds, **{name: float(value.detach()) for name, value in metrics.items()}}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(train_record) + "\n")
        if step == 1 or step % 10 == 0:
            print(json.dumps({"status": "training", "variant": VARIANT, **train_record}), flush=True)
        if step % VALIDATION_INTERVAL == 0 or step == MAX_STEPS:
            validation = validation_metrics(model, reference, validation_store, prompt_embeddings, device)
            validation_record = {"step": step, "split": "validation", **validation}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(validation_record) + "\n")
            print(json.dumps({"status": "validation", "variant": VARIANT, **validation_record}), flush=True)
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                best_step = step
                best_state = trainable_state(model)
                stale_validations = 0
                atomic_torch_save(best_state, best_checkpoint_path)
            else:
                stale_validations += 1
            training_checkpoint = {
                "step": step,
                "trainable_state": trainable_state(model),
                "optimizer_state": optimizer.state_dict(),
                "best_validation_loss": best_validation_loss,
                "best_step": best_step,
                "stale_validations": stale_validations,
                "initial_validation": initial_validation,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "train_store_rng_state": train_store.rng.getstate(),
                "validation_store_rng_state": validation_store.rng.getstate(),
            }
            atomic_torch_save(training_checkpoint, last_checkpoint_path)
            if step in SNAPSHOT_STEPS:
                snapshot_directory = output_directory / "snapshots"
                snapshot_directory.mkdir(parents=True, exist_ok=True)
                atomic_torch_save(training_checkpoint["trainable_state"], snapshot_directory / f"step_{step}_trainable.pt")
                atomic_torch_save(training_checkpoint, snapshot_directory / f"step_{step}_training_state.pt")
            elapsed_seconds = time.monotonic() - started
            remaining_steps = max(0, MAX_STEPS - step)
            estimated_remaining_seconds = elapsed_seconds / max(1, step - start_step + 1) * remaining_steps
            atomic_json(run_state_path, {"status": "running", "variant": VARIANT, "pid": os.getpid(), "host": socket.gethostname(), "step": step, "max_steps": MAX_STEPS, "best_step": best_step, "validation_loss": validation["loss"], "best_validation_loss": best_validation_loss, "estimated_remaining_seconds": estimated_remaining_seconds, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(), "updated_at": utc_now()})
            notify(f"MineStudio stone recovery BC progress\nvariant={VARIANT}\nstep={step}/{MAX_STEPS}\nvalidation_loss={validation['loss']:.6f}\nbest_step={best_step}\nbest_loss={best_validation_loss:.6f}\neta_seconds={estimated_remaining_seconds:.0f}")
            if stale_validations >= EARLY_STOPPING_PATIENCE:
                break
    if best_state is None:
        best_state = torch.load(output_directory / "best_trainable.pt", map_location="cpu", weights_only=True)
    load_trainable_state(model, best_state)
    if "lora" in VARIANT:
        merge_lora(model)
    final_validation = validation_metrics(model, reference, validation_store, prompt_embeddings, device)
    checkpoint_directory = output_directory / "best_model"
    model.save_pretrained(checkpoint_directory)
    dataset_summary = json.loads((DATASET_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    summary = {
        **trainable_summary,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "dataset_directory": str(DATASET_DIRECTORY),
        "frozen_manifest_sha256": dataset_summary["source"]["manifest_sha256"],
        "max_steps": MAX_STEPS,
        "completed_steps": step,
        "best_step": best_step,
        "initial_validation": initial_validation,
        "best_validation_loss": best_validation_loss,
        "final_validation": final_validation,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "targeted_fraction": TARGETED_FRACTION,
        "reference_kl_weight": REFERENCE_KL_WEIGHT,
        "learning_rate": LEARNING_RATES[VARIANT],
        "task_bank_sha256": prompt_bank.sha256,
        "task_bank_model": prompt_bank.model,
        "task_prompt_count": len(prompt_bank.prompts),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_directory": str(checkpoint_directory),
    }
    atomic_json(output_directory / "summary.json", summary)
    atomic_json(run_state_path, {"status": "completed", "variant": VARIANT, "step": step, "max_steps": MAX_STEPS, "best_step": best_step, "best_validation_loss": best_validation_loss, "checkpoint_directory": str(checkpoint_directory), "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(), "updated_at": utc_now()})
    print(json.dumps({"status": "complete", **summary}, indent=2), flush=True)
    notify(f"MineStudio stone recovery BC completed\nvariant={VARIANT}\nstep={step}\nbest_step={best_step}\nbest_loss={best_validation_loss:.6f}\ncheckpoint={checkpoint_directory}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        output_directory = OUTPUT_ROOT / VARIANT
        output_directory.mkdir(parents=True, exist_ok=True)
        atomic_json(output_directory / "run_state.json", {"status": "failed", "variant": VARIANT, "pid": os.getpid(), "host": socket.gethostname(), "error": str(error), "updated_at": utc_now()})
        notify(f"MineStudio stone recovery BC failed\nvariant={VARIANT}\nerror={error}\noutput={output_directory}")
        raise
