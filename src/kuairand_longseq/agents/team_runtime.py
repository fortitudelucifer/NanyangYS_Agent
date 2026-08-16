"""Structured team transport boundary.

The local backend is runnable today.  A future AgentTeams adapter must
implement the same dispatch contract; scientific authority stays in Harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kuairand_longseq.harness.contracts import RoleId, SkillCall, SkillResult, SkillStatus
from kuairand_longseq.harness.runtime import SkillExecutor
from kuairand_longseq.harness.storage import checkpoint_state


@dataclass(frozen=True, slots=True)
class AgentTask:
    task_id: str
    role: RoleId
    call: SkillCall
    depends_on: tuple[str, ...] = ()
    context_keys: tuple[str, ...] = ()


class TeamRuntime(Protocol):
    backend_id: str

    def dispatch(self, task: AgentTask) -> SkillResult: ...


class LocalStructuredTeamRuntime:
    """Deterministic role fixture; it is not an AgentTeams implementation."""

    backend_id = "local_structured_v1"

    def __init__(self, executor: SkillExecutor) -> None:
        self.executor = executor
        self._results: dict[str, SkillResult] = {}

    def _reject(self, task: AgentTask, reason: str, message: str) -> SkillResult:
        result = SkillResult(
            call_id=task.call.call_id,
            skill_name=task.call.skill_name,
            status=SkillStatus.REJECTED,
            reason_code=reason,
            message=message,
        )
        self.executor.context.events.append(
            "task.rejected",
            "orchestrator",
            {"task_id": task.task_id, "reason_code": reason, "message": message},
        )
        self.executor.context.state.sequence = self.executor.context.events.sequence
        checkpoint_state(self.executor.context.store, self.executor.context.state)
        return result

    def dispatch(self, task: AgentTask) -> SkillResult:
        if task.role is not task.call.actor:
            return self._reject(
                task,
                "TASK_ROLE_ACTOR_MISMATCH",
                "task role and skill-call actor differ",
            )
        if task.task_id in self._results:
            return self._reject(task, "TASK_ID_ALREADY_DISPATCHED", f"task ID was already used: {task.task_id}")
        unsatisfied = [
            dependency
            for dependency in task.depends_on
            if dependency not in self._results
            or self._results[dependency].status not in {SkillStatus.SUCCEEDED, SkillStatus.CACHED}
        ]
        if unsatisfied:
            result = self._reject(
                task,
                "TASK_DEPENDENCY_UNSATISFIED",
                f"required predecessor tasks are not successful: {', '.join(unsatisfied)}",
            )
            self._results[task.task_id] = result
            return result
        self.executor.context.events.append(
            "task.dispatched",
            "orchestrator",
            {
                "task_id": task.task_id,
                "role": task.role.value,
                "skill_name": task.call.skill_name,
                "depends_on": list(task.depends_on),
                "context_keys": list(task.context_keys),
                "backend": self.backend_id,
            },
        )
        result = self.executor.invoke(task.call)
        self._results[task.task_id] = result
        self.executor.context.events.append(
            "task.returned",
            task.role.value,
            {
                "task_id": task.task_id,
                "status": result.status.value,
                "reason_code": result.reason_code,
                "artifact_paths": [artifact.path for artifact in result.artifacts],
            },
        )
        return result

