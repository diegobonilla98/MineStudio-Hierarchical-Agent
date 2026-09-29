from dataclasses import dataclass
from enum import Enum

from minestudio.system_zero.contracts import SystemZeroContext, SystemZeroRequest, SystemZeroResult, SystemZeroStatus
from minestudio.system_zero.registry import SystemZeroRegistry


class ExecutionRoute(str, Enum):
    SYSTEM_ZERO = "SYSTEM_ZERO"
    SYSTEM_ONE = "SYSTEM_ONE"


@dataclass
class RoutingResult:
    route: ExecutionRoute
    reason: str
    result: SystemZeroResult | None = None

    def to_dict(self) -> dict:
        return {
            "route": self.route.value,
            "reason": self.reason,
            "result": None if self.result is None else self.result.to_dict(),
        }


class ExecutionRouter:
    def __init__(self, registry: SystemZeroRegistry):
        self.registry = registry

    def route(self, request: SystemZeroRequest, context: SystemZeroContext) -> RoutingResult:
        preconditions = self.registry.check_preconditions(request, context.observation)
        if preconditions.status == SystemZeroStatus.NEEDS_SYSTEM1:
            return RoutingResult(ExecutionRoute.SYSTEM_ONE, str(preconditions.details.get("reason", "uncertain execution condition")))
        if preconditions.status == SystemZeroStatus.PRECONDITION_FAILED:
            result = SystemZeroResult(
                status=SystemZeroStatus.PRECONDITION_FAILED,
                action=request.skill.upper(),
                details=preconditions.details,
            )
            return RoutingResult(ExecutionRoute.SYSTEM_ZERO, "deterministic precondition failure", result)
        result = self.registry.execute(request, context)
        if result.status in {SystemZeroStatus.NEEDS_SYSTEM1, SystemZeroStatus.TARGET_LOST}:
            return RoutingResult(ExecutionRoute.SYSTEM_ONE, str(result.details.get("reason", result.status.value)), result)
        return RoutingResult(ExecutionRoute.SYSTEM_ZERO, "deterministic skill executed", result)
