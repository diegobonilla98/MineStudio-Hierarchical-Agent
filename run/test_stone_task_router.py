from unittest import TestCase

from stone_task_router import LegacyStoneTaskRouter, MAX_HOLE_RECOVERY_STEPS, STABLE_LAND_STEPS, StoneTaskRouter


def state(water: int = 0, y: float = 64.0) -> dict:
    return {"water": water, "position": {"y": y}}


class StoneTaskRouterTests(TestCase):
    def test_transient_dry_frame_does_not_switch_from_water_recovery(self):
        router = StoneTaskRouter(64.0)
        self.assertEqual(router.update(state(water=1), 0, False, 0).category, "water_recovery")
        self.assertEqual(router.update(state(water=0), 1, False, 0).category, "water_recovery")
        self.assertEqual(router.update(state(water=1), 2, False, 0).category, "water_recovery")
        self.assertEqual(router.diagnostics()["suppressed_switches"], 1)

    def test_stable_dry_ground_completes_water_recovery(self):
        router = StoneTaskRouter(64.0)
        router.update(state(water=1), 0, False, 0)
        decisions = [router.update(state(water=0), step, False, 0) for step in range(1, STABLE_LAND_STEPS + 22)]
        categories = [decision.category for decision in decisions]
        self.assertIn("shore_exit", categories)
        self.assertEqual(categories[-1], "stone_reacquisition")
        self.assertIsNotNone(router.stable_land_step)

    def test_hole_signal_requires_sustained_evidence_and_stable_exit(self):
        router = StoneTaskRouter(64.0)
        router.update(state(), 0, False, 0)
        for step in range(1, 8):
            self.assertNotEqual(router.update(state(y=61.0), step, True, 0).category, "hole_recovery")
        decisions = [router.update(state(y=61.0), step, True, 0) for step in range(8, 30)]
        self.assertIn("hole_recovery", [decision.category for decision in decisions])
        transient = router.update(state(y=64.0), 30, False, 0)
        self.assertEqual(transient.category, "hole_recovery")
        for step in range(31, 60):
            final = router.update(state(y=64.0), step, False, 0)
        self.assertEqual(final.category, "stone_acquisition")

    def test_water_reentry_can_interrupt_after_confirmation(self):
        router = StoneTaskRouter(64.0)
        router.update(state(), 0, False, 0)
        first = router.update(state(water=1), 1, False, 0)
        second = router.update(state(water=1), 2, False, 0)
        self.assertEqual(first.category, "stone_acquisition")
        self.assertEqual(second.category, "water_recovery")

    def test_hole_recovery_is_bounded_when_exit_predicate_never_arrives(self):
        router = StoneTaskRouter(64.0)
        router.update(state(), 0, False, 0)
        categories = []
        for step in range(1, MAX_HOLE_RECOVERY_STEPS + 40):
            categories.append(router.update(state(y=61.0), step, True, 0).category)
        self.assertIn("hole_recovery", categories)
        self.assertEqual(categories[-1], "stone_acquisition")


class LegacyStoneTaskRouterTests(TestCase):
    def test_switches_immediately_on_water_frames(self):
        router = LegacyStoneTaskRouter(64.0)
        self.assertEqual(router.update(state(), 0, False, 0).category, "stone_acquisition")
        self.assertEqual(router.update(state(water=1), 1, False, 0).category, "water_recovery")
        self.assertEqual(router.update(state(), 2, False, 0).category, "shore_exit")

    def test_uses_original_stable_land_and_reacquisition_windows(self):
        router = LegacyStoneTaskRouter(64.0)
        router.update(state(water=1), 0, False, 0)
        categories = [router.update(state(), step, False, 0).category for step in range(1, 11)]
        self.assertEqual(categories[:9], ["shore_exit"] * 9)
        self.assertEqual(categories[9], "stone_reacquisition")
        self.assertEqual(router.update(state(), 129, False, 0).category, "stone_reacquisition")
        self.assertEqual(router.update(state(), 130, False, 0).category, "stone_acquisition")

    def test_hole_recovery_has_original_immediate_trigger(self):
        router = LegacyStoneTaskRouter(64.0)
        decision = router.update(state(y=61.0), 0, True, 0)
        self.assertEqual(decision.category, "hole_recovery")
