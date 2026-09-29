import gc
import json
import sys
import time
import unittest
from pathlib import Path

import torch
from minestudio.models import GrootPolicy, SteveOnePolicy, VPTPolicy
from model import BINARY_KEYS, CrossViewRocket

project_root = Path(__file__).resolve().parents[1]
checkpoint_root = project_root / "checkpoints"
output_path = project_root / "output" / "pretrained_model_smoke.json"
device = "cuda"
records = {}


class PretrainedModelTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def record(self, name, model, start_time, outputs, state):
        torch.cuda.synchronize()
        tensors = list(outputs.values()) + list(state)
        self.assertTrue(all(torch.isfinite(tensor).all().item() for tensor in tensors))
        records[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
            "state_shapes": [list(value.shape) for value in state],
        }
        print(json.dumps({name: records[name]}, indent=2))

    def test_01_steve_one_official(self):
        start_time = time.perf_counter()
        model = SteveOnePolicy.from_pretrained(checkpoint_root / "steve_one_official").to(device).eval()
        action, state = model.get_action(
            input={
                "image": torch.zeros((1, 1, 128, 128, 3), dtype=torch.uint8, device=device),
                "condition": {"cond_scale": 4.0, "text": "mine log"},
            },
            state_in=None,
            deterministic=True,
        )
        outputs = {**action, "value": model.vpred}
        self.record("steve_one_official", model, start_time, outputs, state)
        del model

    def test_02_vpt_early_game_2x(self):
        start_time = time.perf_counter()
        model = VPTPolicy.from_pretrained(checkpoint_root / "vpt_early_game_2x").to(device).eval()
        action, state = model.get_action(
            input={"image": torch.zeros((1, 1, 128, 128, 3), dtype=torch.uint8, device=device)},
            state_in=None,
            deterministic=True,
        )
        outputs = {**action, "value": model.vpred}
        self.record("vpt_early_game_2x", model, start_time, outputs, state)
        del model

    def test_03_groot_18w_ema(self):
        start_time = time.perf_counter()
        model = GrootPolicy.from_pretrained(checkpoint_root / "groot_18w_ema").to(device).eval()
        outputs, state = model(
            input={"image": torch.zeros((1, 2, 224, 224, 3), dtype=torch.uint8, device=device)},
            memory=None,
        )
        output_tensors = {
            "buttons": outputs["pi_logits"]["buttons"],
            "camera": outputs["pi_logits"]["camera"],
            "value": outputs["vpred"],
            "posterior": outputs["posterior_dist"]["z"],
            "prior": outputs["prior_dist"]["z"],
        }
        self.record("groot_18w_ema", model, start_time, output_tensors, state)
        del model

    def rocket_input(self):
        previous_action = {
            "camera": torch.zeros((1, 1, 2), dtype=torch.float32, device=device),
        }
        for key in BINARY_KEYS:
            previous_action[key.replace("_", ".")] = torch.zeros((1, 1), dtype=torch.long, device=device)
        return {
            "image": torch.zeros((1, 1, 224, 224, 3), dtype=torch.uint8, device=device),
            "cross_view": {
                "cross_view_image": torch.zeros((1, 1, 224, 224, 3), dtype=torch.uint8, device=device),
                "cross_view_obj_id": torch.zeros((1, 1), dtype=torch.long, device=device),
                "cross_view_obj_mask": torch.zeros((1, 1, 224, 224), dtype=torch.float32, device=device),
            },
            "env_prev_action": previous_action,
        }

    def test_04_rocket_two_1x_22w(self):
        start_time = time.perf_counter()
        model = CrossViewRocket.from_pretrained(checkpoint_root / "rocket_two_1x_22w").to(device).eval()
        outputs, state = model(input=self.rocket_input(), memory=None)
        output_tensors = {
            "buttons": outputs["pi_logits"]["buttons"],
            "camera": outputs["pi_logits"]["camera"],
            "value": outputs["vpred"],
            "exist": outputs["exist"],
            "point": outputs["point"],
            "bbox": outputs["bbox"],
        }
        self.record("rocket_two_1x_22w", model, start_time, output_tensors, state)
        del model

    def test_05_rocket_two_1_5x_17w(self):
        start_time = time.perf_counter()
        model = CrossViewRocket.from_pretrained(checkpoint_root / "rocket_two_1_5x_17w").to(device).eval()
        outputs, state = model(input=self.rocket_input(), memory=None)
        output_tensors = {
            "buttons": outputs["pi_logits"]["buttons"],
            "camera": outputs["pi_logits"]["camera"],
            "value": outputs["vpred"],
            "exist": outputs["exist"],
            "point": outputs["point"],
            "bbox": outputs["bbox"],
        }
        self.record("rocket_two_1_5x_17w", model, start_time, output_tensors, state)
        del model


torch.set_float32_matmul_precision("high")
suite = unittest.defaultTestLoader.loadTestsFromTestCase(PretrainedModelTests)
test_result = unittest.TextTestRunner(verbosity=2).run(suite)
output_path.parent.mkdir(parents=True, exist_ok=True)
result_document = {
    "successful": test_result.wasSuccessful(),
    "tests_run": test_result.testsRun,
    "failures": [{"test": str(test), "traceback": traceback} for test, traceback in test_result.failures],
    "errors": [{"test": str(test), "traceback": traceback} for test, traceback in test_result.errors],
    "records": records,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
}
output_path.write_text(json.dumps(result_document, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result_document, indent=2))
sys.exit(0 if test_result.wasSuccessful() else 1)
