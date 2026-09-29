import unittest
from collections import deque

import torch
import train_system1_recovery_ppo as ppo


class TestSystem1RecoveryPPO(unittest.TestCase):
    def test_compute_gae_terminal(self):
        advantages, returns = ppo.compute_gae([0.0, 1.0], [0.2, 0.4], [False, True])
        self.assertAlmostEqual(advantages[1], 0.6)
        self.assertAlmostEqual(returns[1], 1.0)
        expected_first = -0.2 + ppo.GAMMA * 0.4 + ppo.GAMMA * ppo.GAE_LAMBDA * 0.6
        self.assertAlmostEqual(advantages[0], expected_first)
        self.assertAlmostEqual(returns[0], expected_first + 0.2)

    def test_adaptive_curriculum_is_normalized(self):
        history = {task: deque(maxlen=20) for task in ppo.TASKS}
        history["RECOVER_CAMERA"].extend([True] * 10)
        history["AVOID_DIGGING_TRAP"].extend([False] * 10)
        weights = ppo.adaptive_weights(history)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["AVOID_DIGGING_TRAP"], weights["RECOVER_CAMERA"])
        self.assertEqual(set(weights), set(ppo.TASKS))

    def test_behavioral_score_penalizes_normal_regression(self):
        baseline = {"weak_mean": 0.1, "normal_stone": {"success_rate": 0.7}}
        preserved = {"weak_mean": 0.2, "normal_stone": {"success_rate": 0.7}}
        regressed = {"weak_mean": 0.2, "normal_stone": {"success_rate": 0.5}}
        self.assertGreater(ppo.behavioral_score(preserved, baseline), ppo.behavioral_score(regressed, baseline))

    def test_improvement_count(self):
        baseline = {"micro": {task: {"success_rate": 0.0} for task in ppo.TASKS}}
        evaluation = {"micro": {task: {"success_rate": 0.2 if index < 3 else 0.0} for index, task in enumerate(ppo.TASKS)}}
        self.assertEqual(ppo.improvement_count(evaluation, baseline), 3)

    def test_guided_policy_scores_are_normalized(self):
        scores = {"buttons": torch.tensor([[[3.0, -2.0, 0.5]]])}
        normalized = ppo.normalized_policy_logits(scores)
        self.assertTrue(torch.allclose(torch.logsumexp(normalized["buttons"], dim=-1), torch.zeros((1, 1))))
        entropy = -(normalized["buttons"].exp() * normalized["buttons"]).sum(dim=-1)
        self.assertGreaterEqual(float(entropy.min()), 0.0)

    def test_only_transient_episode_errors_are_retryable(self):
        self.assertTrue(ppo.retryable_episode_error(TimeoutError("timed out")))
        self.assertTrue(ppo.retryable_episode_error(RuntimeError("ESCAPE_HOLE initial state has no stable floor")))
        self.assertTrue(ppo.retryable_episode_error(RuntimeError("RECOVER_CAMERA initial state unexpectedly contains water: 1")))
        self.assertFalse(ppo.retryable_episode_error(RuntimeError("Non-finite PPO loss")))
        self.assertTrue(ppo.invalid_initial_state_error(RuntimeError("ESCAPE_HOLE initial state has no stable floor")))
        self.assertFalse(ppo.invalid_initial_state_error(TimeoutError("timed out")))
        self.assertTrue(ppo.retryable_episode_error(TypeError("a bytes-like object is required, not 'NoneType'")))
        self.assertFalse(ppo.retryable_episode_error(TypeError("unrelated type failure")))


if __name__ == "__main__":
    unittest.main()
