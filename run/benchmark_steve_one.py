import json
import statistics
import time
from pathlib import Path

import torch
from minestudio.models import SteveOnePolicy

project_root = Path(__file__).resolve().parents[1]
checkpoint_path = project_root / "checkpoints" / "steve_one_official"
output_path = project_root / "output" / "steve_one_benchmark.json"
device = "cuda"
instruction = "mine log"
condition_scale = 4.0
warmup_steps = 5
measurement_steps = 30

torch.set_float32_matmul_precision("high")
torch.cuda.reset_peak_memory_stats()

load_start = time.perf_counter()
model = SteveOnePolicy.from_pretrained(checkpoint_path).to(device).eval()
torch.cuda.synchronize()
load_seconds = time.perf_counter() - load_start

condition_start = time.perf_counter()
condition = model.prepare_condition(
    {"cond_scale": condition_scale, "text": instruction},
    deterministic=True,
)
torch.cuda.synchronize()
condition_seconds = time.perf_counter() - condition_start

model_input = {
    "image": torch.zeros((1, 1, 128, 128, 3), dtype=torch.uint8, device=device),
    "condition": condition,
}
state = None

for _ in range(warmup_steps):
    action, state = model.get_action(model_input, state, deterministic=True)

latencies_ms = []
for _ in range(measurement_steps):
    start = time.perf_counter()
    action, state = model.get_action(model_input, state, deterministic=True)
    torch.cuda.synchronize()
    latencies_ms.append((time.perf_counter() - start) * 1000)

all_tensors = list(action.values()) + list(state) + [model.vpred]
finite = all(torch.isfinite(tensor).all().item() for tensor in all_tensors)
sorted_latencies = sorted(latencies_ms)
p95_index = min(len(sorted_latencies) - 1, round(0.95 * (len(sorted_latencies) - 1)))
result = {
    "checkpoint": str(checkpoint_path),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "instruction": instruction,
    "condition_scale": condition_scale,
    "load_seconds": round(load_seconds, 4),
    "condition_seconds": round(condition_seconds, 4),
    "warmup_steps": warmup_steps,
    "measurement_steps": measurement_steps,
    "mean_step_ms": round(statistics.mean(latencies_ms), 4),
    "median_step_ms": round(statistics.median(latencies_ms), 4),
    "p95_step_ms": round(sorted_latencies[p95_index], 4),
    "steps_per_second": round(1000 / statistics.mean(latencies_ms), 3),
    "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    "finite": finite,
}

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
