import unittest

import torch

from run.train_stone_recovery_bc import LoRALinear, masked_mean, normalized_policy_logits


class RecoveryTrainingTests(unittest.TestCase):
    def test_lora_merge_preserves_output(self):
        torch.manual_seed(4)
        base = torch.nn.Linear(5, 3)
        module = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
        module.lora_b.weight.data.normal_()
        value = torch.randn(7, 5)
        expected = module(value)
        actual = module.merged()(value)
        torch.testing.assert_close(actual, expected)

    def test_masked_mean(self):
        value = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
        mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        self.assertEqual(float(masked_mean(value, mask)), 1.5)

    def test_normalized_policy_logits(self):
        logits = {"buttons": torch.tensor([[[2.0, 4.0]]])}
        normalized = normalized_policy_logits(logits)["buttons"]
        torch.testing.assert_close(torch.exp(normalized).sum(-1), torch.ones(1, 1))


if __name__ == "__main__":
    unittest.main()
