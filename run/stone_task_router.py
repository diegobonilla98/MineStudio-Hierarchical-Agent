from dataclasses import dataclass


TASK_SWITCH_CONFIRM_STEPS = 4
TASK_SWITCH_COOLDOWN_STEPS = 20
WATER_REENTRY_CONFIRM_STEPS = 2
STABLE_LAND_STEPS = 12
HOLE_ENTRY_CONFIRM_STEPS = 8
HOLE_EXIT_CONFIRM_STEPS = 12
MAX_HOLE_RECOVERY_STEPS = 80
HOLE_REENTRY_COOLDOWN_STEPS = 80
STONE_REACQUISITION_STEPS = 160


@dataclass
class RouterDecision:
    category: str
    changed: bool
    suppressed: bool


class LegacyStoneTaskRouter:
    def __init__(self, baseline_y: float):
        self.baseline_y = baseline_y
        self.active_category = None
        self.entered_water_once = False
        self.dry_steps_after_water = 0
        self.stable_land_step = None
        self.stone_mined_at_land = None

    def update(self, state: dict, step: int, low_displacement: bool, stone_mined: float) -> RouterDecision:
        in_water = int(state["water"]) == 1
        if in_water:
            requested = "water_recovery"
            self.entered_water_once = True
            self.dry_steps_after_water = 0
            self.stable_land_step = None
            self.stone_mined_at_land = None
        else:
            if self.entered_water_once and self.stable_land_step is None:
                self.dry_steps_after_water += 1
                if self.dry_steps_after_water >= 10:
                    self.stable_land_step = step
                    self.stone_mined_at_land = stone_mined
            below_local_grade = float(state["position"]["y"]) <= self.baseline_y - 1.5
            if below_local_grade and low_displacement:
                requested = "hole_recovery"
            elif self.entered_water_once and self.stable_land_step is None:
                requested = "shore_exit"
            elif (
                self.stable_land_step is not None
                and self.stone_mined_at_land is not None
                and stone_mined <= self.stone_mined_at_land
                and step - self.stable_land_step < 120
            ):
                requested = "stone_reacquisition"
            else:
                requested = "stone_acquisition"
        changed = requested != self.active_category
        self.active_category = requested
        return RouterDecision(requested, changed, False)

    def diagnostics(self) -> dict:
        return {
            "suppressed_switches": 0,
            "entered_water_once": self.entered_water_once,
            "stable_land_step": self.stable_land_step,
            "active_category": self.active_category,
        }


class StoneTaskRouter:
    def __init__(self, baseline_y: float):
        self.baseline_y = baseline_y
        self.active_category = None
        self.candidate_category = None
        self.candidate_steps = 0
        self.last_switch_step = -TASK_SWITCH_COOLDOWN_STEPS
        self.entered_water_once = False
        self.recovery_pending = False
        self.water_steps = 0
        self.dry_steps = 0
        self.stable_land_step = None
        self.stone_mined_at_land = None
        self.hole_signal_steps = 0
        self.hole_recovery_active = False
        self.hole_clear_steps = 0
        self.hole_recovery_start_step = None
        self.hole_recovery_blocked_until = 0
        self.suppressed_switches = 0

    def update(self, state: dict, step: int, low_displacement: bool, stone_mined: float) -> RouterDecision:
        in_water = int(state["water"]) == 1
        position_y = float(state["position"]["y"])
        if in_water:
            self.entered_water_once = True
            self.recovery_pending = True
            self.water_steps += 1
            self.dry_steps = 0
            self.stable_land_step = None
            self.stone_mined_at_land = None
        elif self.recovery_pending:
            self.dry_steps += 1
            self.water_steps = 0
            if self.dry_steps >= STABLE_LAND_STEPS:
                self.recovery_pending = False
                self.stable_land_step = step
                self.stone_mined_at_land = stone_mined
        else:
            self.water_steps = 0

        below_local_grade = position_y <= self.baseline_y - 1.5
        if not self.recovery_pending and below_local_grade and low_displacement:
            self.hole_signal_steps += 1
        else:
            self.hole_signal_steps = 0
        if not self.hole_recovery_active and step >= self.hole_recovery_blocked_until and self.hole_signal_steps >= HOLE_ENTRY_CONFIRM_STEPS:
            self.hole_recovery_active = True
            self.hole_clear_steps = 0
            self.hole_recovery_start_step = step
        if self.hole_recovery_active:
            clear_signal = position_y >= self.baseline_y - 0.5 and not low_displacement and not in_water
            self.hole_clear_steps = self.hole_clear_steps + 1 if clear_signal else 0
            recovery_expired = self.hole_recovery_start_step is not None and step - self.hole_recovery_start_step >= MAX_HOLE_RECOVERY_STEPS
            if self.hole_clear_steps >= HOLE_EXIT_CONFIRM_STEPS or recovery_expired:
                self.hole_recovery_active = False
                self.hole_signal_steps = 0
                self.hole_recovery_start_step = None
                self.hole_recovery_blocked_until = step + HOLE_REENTRY_COOLDOWN_STEPS

        requested = self._requested_category(in_water, step, stone_mined)
        return self._debounce(requested, step, in_water)

    def diagnostics(self) -> dict:
        return {
            "suppressed_switches": self.suppressed_switches,
            "entered_water_once": self.entered_water_once,
            "stable_land_step": self.stable_land_step,
            "active_category": self.active_category,
        }

    def _requested_category(self, in_water: bool, step: int, stone_mined: float) -> str:
        if self.recovery_pending:
            return "water_recovery" if in_water else "shore_exit"
        if self.hole_recovery_active:
            return "hole_recovery"
        if (
            self.stable_land_step is not None
            and self.stone_mined_at_land is not None
            and stone_mined <= self.stone_mined_at_land
            and step - self.stable_land_step < STONE_REACQUISITION_STEPS
        ):
            return "stone_reacquisition"
        return "stone_acquisition"

    def _debounce(self, requested: str, step: int, in_water: bool) -> RouterDecision:
        if self.active_category is None:
            self.active_category = requested
            self.last_switch_step = step
            return RouterDecision(requested, True, False)
        if requested == self.active_category:
            self.candidate_category = None
            self.candidate_steps = 0
            return RouterDecision(self.active_category, False, False)
        if requested == self.candidate_category:
            self.candidate_steps += 1
        else:
            self.candidate_category = requested
            self.candidate_steps = 1
        required_steps = WATER_REENTRY_CONFIRM_STEPS if requested == "water_recovery" and in_water else TASK_SWITCH_CONFIRM_STEPS
        cooldown_complete = step - self.last_switch_step >= TASK_SWITCH_COOLDOWN_STEPS
        emergency_water_reentry = requested == "water_recovery" and in_water and self.candidate_steps >= WATER_REENTRY_CONFIRM_STEPS
        confirmed_shore_progress = self.active_category == "water_recovery" and requested == "shore_exit" and self.candidate_steps >= TASK_SWITCH_CONFIRM_STEPS
        if self.candidate_steps >= required_steps and (cooldown_complete or emergency_water_reentry or confirmed_shore_progress):
            self.active_category = requested
            self.last_switch_step = step
            self.candidate_category = None
            self.candidate_steps = 0
            return RouterDecision(requested, True, False)
        self.suppressed_switches += 1
        return RouterDecision(self.active_category, False, True)
