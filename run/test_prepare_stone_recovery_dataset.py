import unittest

import numpy as np

from run.prepare_stone_recovery_dataset import hole_recoveries, stable_water_exits, steve_spans, transition_indices, water_entries


class RecoveryDatasetTests(unittest.TestCase):
    def test_water_episode_start_and_stable_exit(self):
        water = np.array([1, 1, 0] + [0] * 19, dtype=np.int8)
        self.assertEqual(water_entries(water), [0])
        self.assertEqual(stable_water_exits(water), [2])

    def test_unstable_exit_is_rejected(self):
        water = np.array([0, 1, 0, 0, 1] + [0] * 20, dtype=np.int8)
        self.assertEqual(water_entries(water), [1, 4])
        self.assertEqual(stable_water_exits(water), [5])

    def test_progress_transitions(self):
        values = np.array([0, 0, 1, 1, 3], dtype=np.float32)
        self.assertEqual(transition_indices(values).tolist(), [2, 4])

    def test_hole_recovery_requires_sustained_interval(self):
        y_values = np.array([64.0] * 10 + [62.0] * 12 + [63.6] * 10, dtype=np.float32)
        position = np.zeros((len(y_values), 5), dtype=np.float32)
        position[:, 1] = y_values
        self.assertEqual(hole_recoveries(position, 64.0), [(10, 22)])

    def test_option_phases_are_excluded(self):
        phases = ["STEVE_1/stone_acquisition"] * 30 + ["COLLECT_DROP/sweep"] * 5 + ["STEVE_1/stone_acquisition"] * 25
        self.assertEqual(steve_spans(phases, 0, len(phases)), [(0, 30), (35, 60)])


if __name__ == "__main__":
    unittest.main()
