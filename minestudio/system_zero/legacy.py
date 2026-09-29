from collections.abc import Callable, Mapping
from typing import Any

from minestudio.system_zero.contracts import BackendResult, SystemZeroStatus
from minestudio.system_zero.inventory import count_item, event_count


class LegacyClosedLoopBackend:
    def __init__(self, controller: Any):
        self.controller = controller

    def execute(
        self,
        action: str,
        parameters: Mapping[str, Any],
        observation: Mapping[str, Any],
        timeout_steps: int,
        should_abort: Callable[[], bool],
    ) -> BackendResult:
        start_steps = int(self.controller.steps)
        info = dict(observation)
        if should_abort():
            return BackendResult(SystemZeroStatus.FAILURE, info, details={"reason": "interrupted before backend execution"})
        try:
            if action == "CRAFT_RECIPE":
                result, info = self.controller.craft_recipe(str(parameters["recipe"]), info)
            elif action == "EQUIP_ITEM":
                result, info = self.controller.equip_item(str(parameters["item"]), info)
            elif action == "PLACE_BLOCK":
                result, info = self.controller.place_block(str(parameters["item"]), info)
            elif action == "PLACE_STATION":
                result, info = self.controller.place_block(str(parameters["station"]), info)
            elif action == "RECLAIM_BLOCK":
                return self._reclaim_block(parameters, info, timeout_steps, should_abort, start_steps)
            elif action == "MINE_AND_COLLECT":
                return self._mine_and_collect(parameters, info, timeout_steps, should_abort, start_steps)
            elif action == "COLLECT_KNOWN_DROP":
                return self._collect_known_drop(parameters, info, timeout_steps, start_steps)
            else:
                return BackendResult(
                    SystemZeroStatus.NEEDS_SYSTEM1,
                    info,
                    details={"reason": "legacy backend does not implement operation", "operation": action},
                )
        except RuntimeError as error:
            return BackendResult(
                SystemZeroStatus.FAILURE,
                info,
                target=self._target(parameters),
                details={"reason": "environment failure", "error": str(error)},
                steps=int(self.controller.steps) - start_steps,
            )
        steps = int(self.controller.steps) - start_steps
        return BackendResult(
            self._legacy_status(result),
            info,
            target=self._target(parameters),
            details={"legacy_result": result, "reason": str(result.get("reason", ""))},
            steps=steps,
        )

    def _reclaim_block(
        self,
        parameters: Mapping[str, Any],
        info: dict[str, Any],
        timeout_steps: int,
        should_abort: Callable[[], bool],
        start_steps: int,
    ) -> BackendResult:
        item = str(parameters["item"])
        baseline_item = count_item(info, item)
        baseline_mined = event_count(info, "mine_block", item)
        collect_budget = min(80, max(20, timeout_steps // 3))
        attack_budget = max(1, timeout_steps - collect_budget - 1)
        block_broken = False
        for _ in range(attack_budget):
            if should_abort():
                observation, info = self.controller.step_values({"attack": 0}, "SYSTEM_ZERO/RECLAIM_BLOCK/release")
                return BackendResult(
                    SystemZeroStatus.FAILURE,
                    info,
                    target=item,
                    details={"reason": "interrupted while reclaiming block"},
                    steps=int(self.controller.steps) - start_steps,
                )
            observation, info = self.controller.step_values({"attack": 1}, "SYSTEM_ZERO/RECLAIM_BLOCK/attack")
            if count_item(info, item) > baseline_item:
                block_broken = True
                break
            if event_count(info, "mine_block", item) > baseline_mined:
                block_broken = True
                break
        observation, info = self.controller.step_values({"attack": 0}, "SYSTEM_ZERO/RECLAIM_BLOCK/release")
        if not block_broken:
            return BackendResult(
                SystemZeroStatus.NEEDS_SYSTEM1,
                info,
                target=item,
                details={"reason": "known station was not broken; reacquisition or alignment is required"},
                steps=int(self.controller.steps) - start_steps,
            )
        if count_item(info, item) <= baseline_item:
            remaining = max(1, timeout_steps - (int(self.controller.steps) - start_steps))
            collect_result, info = self.controller.collect_drop(item, baseline_item, info, max_steps=remaining)
            if not bool(collect_result.get("success", False)):
                status = SystemZeroStatus.NEEDS_SYSTEM1 if bool(collect_result.get("needs_system1", False)) else SystemZeroStatus.TIMEOUT
                return BackendResult(
                    status,
                    info,
                    target=item,
                    details={"reason": "station broken but known drop was not collected", "legacy_result": collect_result},
                    steps=int(self.controller.steps) - start_steps,
                )
        return BackendResult(
            SystemZeroStatus.SUCCESS,
            info,
            target=item,
            details={"reason": "station break and inventory delta observed"},
            steps=int(self.controller.steps) - start_steps,
        )

    def _collect_known_drop(
        self,
        parameters: Mapping[str, Any],
        info: dict[str, Any],
        timeout_steps: int,
        start_steps: int,
    ) -> BackendResult:
        item = str(parameters["item"])
        baseline_item = count_item(info, item)
        collect_result, info = self.controller.collect_drop(item, baseline_item, info, max_steps=timeout_steps)
        if bool(collect_result.get("success", False)):
            return BackendResult(
                SystemZeroStatus.SUCCESS,
                info,
                target=item,
                details={"reason": "known drop inventory delta observed", "legacy_result": collect_result},
                steps=int(self.controller.steps) - start_steps,
            )
        status = SystemZeroStatus.NEEDS_SYSTEM1 if bool(collect_result.get("needs_system1", False)) else SystemZeroStatus.TIMEOUT
        return BackendResult(
            status,
            info,
            target=item,
            details={"reason": str(collect_result.get("reason", "drop not collected")), "legacy_result": collect_result},
            steps=int(self.controller.steps) - start_steps,
        )

    def _mine_and_collect(
        self,
        parameters: Mapping[str, Any],
        info: dict[str, Any],
        timeout_steps: int,
        should_abort: Callable[[], bool],
        start_steps: int,
    ) -> BackendResult:
        target_type = str(parameters["target_type"])
        drop_item = str(parameters["drop_item"])
        count = int(parameters["count"])
        tool = parameters.get("tool")
        if tool is not None:
            equip_result, info = self.controller.equip_item(str(tool), info)
            if not bool(equip_result.get("success", False)):
                return BackendResult(
                    self._legacy_status(equip_result),
                    info,
                    target=target_type,
                    details={"reason": "required mining tool could not be equipped", "legacy_result": equip_result},
                    steps=int(self.controller.steps) - start_steps,
                )
        baseline_drop = count_item(info, drop_item)
        baseline_mined = event_count(info, "mine_block", target_type)
        collect_budget = min(120, max(24, timeout_steps // 3))
        attack_budget = max(1, timeout_steps - collect_budget)
        block_broken = False
        for _ in range(attack_budget):
            if should_abort():
                self.controller.step_values({"attack": 0}, "SYSTEM_ZERO/MINE_AND_COLLECT/release")
                return BackendResult(
                    SystemZeroStatus.FAILURE,
                    info,
                    target=target_type,
                    details={"reason": "interrupted while attacking"},
                    steps=int(self.controller.steps) - start_steps,
                )
            observation, info = self.controller.step_values({"attack": 1}, "SYSTEM_ZERO/MINE_AND_COLLECT/attack")
            if count_item(info, drop_item) - baseline_drop >= count:
                block_broken = True
                break
            if event_count(info, "mine_block", target_type) > baseline_mined:
                block_broken = True
                break
        observation, info = self.controller.step_values({"attack": 0}, "SYSTEM_ZERO/MINE_AND_COLLECT/release")
        if not block_broken:
            return BackendResult(
                SystemZeroStatus.NEEDS_SYSTEM1,
                info,
                target=target_type,
                details={"reason": "known target was not broken; reacquisition or alignment is required"},
                steps=int(self.controller.steps) - start_steps,
            )
        if count_item(info, drop_item) - baseline_drop < count:
            remaining = max(1, timeout_steps - (int(self.controller.steps) - start_steps))
            collect_result, info = self.controller.collect_drop(drop_item, baseline_drop, info, max_steps=remaining)
            if not bool(collect_result.get("success", False)):
                status = SystemZeroStatus.NEEDS_SYSTEM1 if bool(collect_result.get("needs_system1", False)) else SystemZeroStatus.TIMEOUT
                return BackendResult(
                    status,
                    info,
                    target=target_type,
                    details={"reason": "block broken but known drop was not collected", "legacy_result": collect_result},
                    steps=int(self.controller.steps) - start_steps,
                )
        return BackendResult(
            SystemZeroStatus.SUCCESS,
            info,
            target=target_type,
            details={"reason": "block break and inventory delta observed"},
            steps=int(self.controller.steps) - start_steps,
        )

    @staticmethod
    def _legacy_status(result: Mapping[str, Any]) -> SystemZeroStatus:
        if bool(result.get("success", False)):
            return SystemZeroStatus.SUCCESS
        reason = str(result.get("reason", "")).lower()
        if bool(result.get("needs_system1", False)) or str(result.get("status", "")).upper() == "NEEDS_SYSTEM1":
            return SystemZeroStatus.NEEDS_SYSTEM1
        if "timeout" in reason:
            return SystemZeroStatus.TIMEOUT
        if "missing" in reason:
            return SystemZeroStatus.PRECONDITION_FAILED
        if "target lost" in reason:
            return SystemZeroStatus.TARGET_LOST
        return SystemZeroStatus.FAILURE

    @staticmethod
    def _target(parameters: Mapping[str, Any]) -> str | None:
        for name in ("target_type", "recipe", "item", "station"):
            if name in parameters:
                return str(parameters[name])
        return None
