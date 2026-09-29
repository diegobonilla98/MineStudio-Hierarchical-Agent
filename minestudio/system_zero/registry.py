import time
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from minestudio.system_zero.contracts import PreconditionResult, SystemZeroContext, SystemZeroRequest, SystemZeroResult, SystemZeroSkill, SystemZeroStatus


JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": Mapping,
}


class SystemZeroRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SystemZeroSkill] = {}

    def register(self, skill: SystemZeroSkill) -> None:
        name = skill.name.upper()
        if name in self._skills:
            raise ValueError(f"System Zero skill already registered: {name}")
        self._skills[name] = skill

    def has(self, name: str) -> bool:
        return name.upper() in self._skills

    def get(self, name: str) -> SystemZeroSkill:
        normalized = name.upper()
        if normalized not in self._skills:
            raise KeyError(f"Unknown System Zero skill: {name}")
        return self._skills[normalized]

    def list_skills(self) -> list[dict[str, Any]]:
        return [self._skills[name].to_dict() for name in sorted(self._skills)]

    def inspect(self, name: str) -> dict[str, Any]:
        return self.get(name).to_dict()

    def validate_request(self, request: SystemZeroRequest) -> PreconditionResult:
        if not self.has(request.skill):
            return PreconditionResult(
                status=SystemZeroStatus.NEEDS_SYSTEM1,
                details={"reason": "no deterministic skill registered", "requested_skill": request.skill},
            )
        skill = self.get(request.skill)
        declared = {parameter.name: parameter for parameter in skill.parameters}
        unknown = sorted(set(request.parameters) - set(declared))
        if unknown:
            return PreconditionResult(
                status=SystemZeroStatus.PRECONDITION_FAILED,
                details={"reason": "unknown parameters", "unknown": unknown},
            )
        missing = [parameter.name for parameter in skill.parameters if parameter.required and parameter.name not in request.parameters]
        if missing:
            return PreconditionResult(
                status=SystemZeroStatus.PRECONDITION_FAILED,
                details={"reason": "missing parameters", "missing_parameters": missing},
            )
        invalid = {}
        for name, value in request.parameters.items():
            expected = JSON_TYPES.get(declared[name].type_name)
            if expected is not None and not isinstance(value, expected):
                invalid[name] = {"expected": declared[name].type_name, "actual": type(value).__name__}
        if invalid:
            return PreconditionResult(
                status=SystemZeroStatus.PRECONDITION_FAILED,
                details={"reason": "invalid parameter types", "invalid": invalid},
            )
        return PreconditionResult()

    def check_preconditions(self, request: SystemZeroRequest, observation: Mapping[str, Any]) -> PreconditionResult:
        validation = self.validate_request(request)
        if not validation.satisfied:
            return validation
        skill = self.get(request.skill)
        parameters = self._with_defaults(skill, request.parameters)
        return skill.check_preconditions(parameters, observation)

    def execute(self, request: SystemZeroRequest, context: SystemZeroContext) -> SystemZeroResult:
        started = time.monotonic()
        preconditions = self.check_preconditions(request, context.observation)
        if not preconditions.satisfied:
            return SystemZeroResult(
                status=preconditions.status or SystemZeroStatus.FAILURE,
                action=request.skill.upper(),
                target=self._target(request.parameters),
                details=preconditions.details,
                elapsed_seconds=time.monotonic() - started,
            )
        skill = self.get(request.skill)
        parameters = self._with_defaults(skill, request.parameters)
        if context.should_abort():
            return SystemZeroResult(
                status=SystemZeroStatus.FAILURE,
                action=skill.name,
                target=self._target(parameters),
                details={"reason": "interrupted before execution"},
                elapsed_seconds=time.monotonic() - started,
            )
        if skill.check_idempotent is not None and skill.check_idempotent(parameters, context.observation):
            return SystemZeroResult(
                status=SystemZeroStatus.SUCCESS,
                action=skill.name,
                target=self._target(parameters),
                details={"reason": "already satisfied", "idempotent": True},
                elapsed_seconds=time.monotonic() - started,
            )
        timeout_steps = skill.timeout_steps if request.timeout_steps is None else min(skill.timeout_steps, request.timeout_steps)
        if timeout_steps <= 0:
            return SystemZeroResult(
                status=SystemZeroStatus.PRECONDITION_FAILED,
                action=skill.name,
                target=self._target(parameters),
                details={"reason": "timeout_steps must be positive"},
                elapsed_seconds=time.monotonic() - started,
            )
        before = deepcopy(context.observation)
        backend_result = context.backend.execute(skill.execution, parameters, before, timeout_steps, context.should_abort)
        context.observation = backend_result.observation
        elapsed = time.monotonic() - started
        if backend_result.steps > timeout_steps:
            return SystemZeroResult(
                status=SystemZeroStatus.TIMEOUT,
                action=skill.name,
                target=backend_result.target or self._target(parameters),
                details={**backend_result.details, "reason": "step budget exceeded", "timeout_steps": timeout_steps},
                steps=backend_result.steps,
                elapsed_seconds=elapsed,
            )
        if context.should_abort():
            return SystemZeroResult(
                status=SystemZeroStatus.FAILURE,
                action=skill.name,
                target=backend_result.target or self._target(parameters),
                details={**backend_result.details, "reason": "interrupted during execution"},
                steps=backend_result.steps,
                elapsed_seconds=elapsed,
            )
        if backend_result.status != SystemZeroStatus.SUCCESS:
            return SystemZeroResult(
                status=backend_result.status,
                action=skill.name,
                target=backend_result.target or self._target(parameters),
                details=backend_result.details,
                steps=backend_result.steps,
                elapsed_seconds=elapsed,
            )
        verified, verifier_details = skill.verify(parameters, before, context.observation)
        return SystemZeroResult(
            status=SystemZeroStatus.SUCCESS if verified else SystemZeroStatus.FAILURE,
            action=skill.name,
            target=backend_result.target or self._target(parameters),
            details={**backend_result.details, **verifier_details, "verified": verified},
            steps=backend_result.steps,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _with_defaults(skill: SystemZeroSkill, values: Mapping[str, Any]) -> dict[str, Any]:
        parameters = dict(values)
        for parameter in skill.parameters:
            if not parameter.required and parameter.name not in parameters:
                parameters[parameter.name] = parameter.default
        return parameters

    @staticmethod
    def _target(parameters: Mapping[str, Any]) -> str | None:
        for name in ("target", "target_type", "recipe", "item", "station"):
            value = parameters.get(name)
            if value is not None:
                return str(value)
        return None
