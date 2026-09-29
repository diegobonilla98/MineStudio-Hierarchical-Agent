from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class SystemZeroStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    TARGET_LOST = "TARGET_LOST"
    TIMEOUT = "TIMEOUT"
    NEEDS_SYSTEM1 = "NEEDS_SYSTEM1"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type_name: str
    description: str
    required: bool = True
    default: Any = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "name": self.name,
            "type": self.type_name,
            "description": self.description,
            "required": self.required,
        }
        if not self.required:
            value["default"] = self.default
        return value


@dataclass(frozen=True)
class SystemZeroRequest:
    skill: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    timeout_steps: int | None = None


@dataclass
class SystemZeroResult:
    status: SystemZeroStatus
    action: str
    target: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status == SystemZeroStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "action": self.action,
            "target": self.target,
            "details": self.details,
            "steps": self.steps,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class PreconditionResult:
    status: SystemZeroStatus | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def satisfied(self) -> bool:
        return self.status is None


@dataclass
class BackendResult:
    status: SystemZeroStatus
    observation: dict[str, Any]
    target: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    steps: int = 0


class SystemZeroBackend(Protocol):
    def execute(
        self,
        action: str,
        parameters: Mapping[str, Any],
        observation: Mapping[str, Any],
        timeout_steps: int,
        should_abort: Callable[[], bool],
    ) -> BackendResult:
        ...


@dataclass
class SystemZeroContext:
    observation: dict[str, Any]
    backend: SystemZeroBackend
    should_abort: Callable[[], bool] = lambda: False


Precondition = Callable[[Mapping[str, Any], Mapping[str, Any]], PreconditionResult]
IdempotencyCheck = Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
Verifier = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], tuple[bool, dict[str, Any]]]


@dataclass(frozen=True)
class SystemZeroSkill:
    name: str
    description: str
    parameters: tuple[ParameterSpec, ...]
    preconditions: tuple[str, ...]
    execution: str
    success_predicate: str
    failure_predicates: tuple[str, ...]
    timeout_steps: int
    abort_conditions: tuple[str, ...]
    check_preconditions: Precondition
    verify: Verifier
    check_idempotent: IdempotencyCheck | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "preconditions": list(self.preconditions),
            "execution": self.execution,
            "success_predicate": self.success_predicate,
            "failure_predicates": list(self.failure_predicates),
            "timeout_steps": self.timeout_steps,
            "abort_conditions": list(self.abort_conditions),
        }
