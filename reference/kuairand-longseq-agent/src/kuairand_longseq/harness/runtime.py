"""Policy-controlled Skill runtime.

This is the run-level control plane.  Individual agents may propose a call,
but only this runtime validates, authorizes, executes, records, and checkpoints it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .contracts import (
    AgentIdentity,
    BudgetLimit,
    EVIDENCE_TIER_RANK,
    EvidenceTier,
    RoleId,
    RunPhase,
    RunState,
    SideEffect,
    SkillCall,
    SkillResult,
    SkillSpec,
    SkillStatus,
    to_jsonable,
)
from .storage import ArtifactStore, EventStore, canonical_json_bytes, checkpoint_state, sha256_bytes


class SkillHandler(Protocol):
    def __call__(self, call: SkillCall, context: "ExecutionContext") -> SkillResult: ...


@dataclass(slots=True)
class ExecutionContext:
    state: RunState
    store: ArtifactStore
    events: EventStore
    config: dict[str, Any]
    evidence_provider: Any
    evidence_admission_policy: Any


class SkillRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[SkillSpec, SkillHandler]] = {}

    def register(self, spec: SkillSpec, handler: SkillHandler) -> None:
        if spec.name in self._entries:
            raise ValueError(f"duplicate skill: {spec.name}")
        self._entries[spec.name] = (spec, handler)

    def get(self, name: str) -> tuple[SkillSpec, SkillHandler] | None:
        return self._entries.get(name)

    @property
    def specs(self) -> tuple[SkillSpec, ...]:
        return tuple(entry[0] for _, entry in sorted(self._entries.items()))


class PolicyEngine:
    def __init__(
        self,
        identities: dict[RoleId, AgentIdentity],
        *,
        sealed_access_enabled: bool = False,
        approval_adapter_enabled: bool = False,
    ) -> None:
        self.identities = identities
        self.sealed_access_enabled = sealed_access_enabled
        self.approval_adapter_enabled = approval_adapter_enabled

    @staticmethod
    def _highest_evidence_tier(state: RunState) -> EvidenceTier:
        tiers = [item.tier for item in state.evidence if item.claim_eligible]
        return max(tiers, key=EVIDENCE_TIER_RANK.__getitem__, default=EvidenceTier.NONE)

    def authorize(self, spec: SkillSpec, call: SkillCall, state: RunState) -> tuple[bool, str, str]:
        identity = self.identities.get(call.actor)
        if identity is None:
            return False, "UNKNOWN_AGENT", f"unregistered agent role: {call.actor.value}"
        if call.actor not in spec.allowed_roles or spec.name not in identity.allowed_skills:
            return False, "ROLE_NOT_AUTHORIZED", f"{call.actor.value} cannot call {spec.name}"
        if spec.side_effect not in identity.allowed_side_effects:
            return False, "SIDE_EFFECT_NOT_AUTHORIZED", f"{call.actor.value} lacks {spec.side_effect.value} permission"
        if spec.side_effect is SideEffect.SEALED_READ and not self.sealed_access_enabled:
            return False, "SEALED_ACCESS_DENIED", "sealed data access is disabled for this run"
        highest = self._highest_evidence_tier(state)
        if EVIDENCE_TIER_RANK[highest] < EVIDENCE_TIER_RANK[spec.minimum_evidence_tier]:
            return False, "EVIDENCE_TIER_TOO_LOW", (
                f"{spec.name} requires {spec.minimum_evidence_tier.value}; current claim-eligible tier is {highest.value}"
            )
        if spec.side_effect in {SideEffect.PUBLISH, SideEffect.ROLLBACK} and not self.approval_adapter_enabled:
            return False, "APPROVAL_ADAPTER_DISABLED", "external mutations are disabled until a trusted approval adapter is configured"
        if spec.side_effect is SideEffect.PUBLISH:
            if state.phase is not RunPhase.WAITING_FOR_APPROVAL:
                return False, "PUBLISH_STATE_INVALID", "publish is legal only from waiting_for_approval"
            readiness = [
                result for result in state.results.values()
                if result.skill_name == "assess_release_readiness"
                and result.facts.get("ready_for_human_approval") is True
            ]
            if not readiness:
                return False, "RELEASE_READINESS_NOT_MET", "no successful release-readiness verdict is registered"
        if spec.side_effect is SideEffect.ROLLBACK and state.phase is not RunPhase.RELEASED:
            return False, "ROLLBACK_STATE_INVALID", "rollback is legal only for a released run"
        if spec.approval_action:
            token = call.approval
            if token is None:
                return False, "APPROVAL_REQUIRED", f"approval required for {spec.approval_action}"
            if token not in state.approvals:
                return False, "APPROVAL_NOT_REGISTERED", "approval token is not present in the harness approval registry"
            if token.status != "approved" or token.action != spec.approval_action:
                return False, "APPROVAL_INVALID", "approval token does not authorize this action"
            if token.contract_sha256 != state.contract_sha256:
                return False, "APPROVAL_CONTRACT_MISMATCH", "approval was issued for a different contract"
            expected_subject = call.inputs.get("subject_sha256")
            if not expected_subject or token.subject_sha256 != expected_subject:
                return False, "APPROVAL_SUBJECT_MISMATCH", "approval does not bind the exact release/rollback subject"
        return True, "AUTHORIZED", "policy checks passed"


class BudgetManager:
    def __init__(self, limits: BudgetLimit) -> None:
        self.limits = limits

    def precheck(self, spec: SkillSpec, state: RunState) -> tuple[bool, str]:
        usage = state.budget
        if usage.steps + 1 > self.limits.max_steps:
            return False, "max_steps"
        if usage.skill_calls + 1 > self.limits.max_skill_calls:
            return False, "max_skill_calls"
        if spec.side_effect in {SideEffect.WRITE_RUN_ARTIFACT, SideEffect.PUBLISH, SideEffect.ROLLBACK}:
            if usage.write_calls + 1 > self.limits.max_write_calls:
                return False, "max_write_calls"
        if spec.side_effect is SideEffect.SEALED_READ and usage.sealed_reads + 1 > self.limits.max_sealed_reads:
            return False, "max_sealed_reads"
        if spec.side_effect is SideEffect.PUBLISH and usage.publish_calls + 1 > self.limits.max_publish_calls:
            return False, "max_publish_calls"
        return True, "within_budget"

    def consume(self, spec: SkillSpec, state: RunState) -> None:
        state.budget.steps += 1
        state.budget.skill_calls += 1
        if spec.side_effect in {SideEffect.WRITE_RUN_ARTIFACT, SideEffect.PUBLISH, SideEffect.ROLLBACK}:
            state.budget.write_calls += 1
        if spec.side_effect is SideEffect.SEALED_READ:
            state.budget.sealed_reads += 1
        if spec.side_effect is SideEffect.PUBLISH:
            state.budget.publish_calls += 1


class SkillExecutor:
    """Normalized validate -> authorize -> execute -> evidence lifecycle."""

    def __init__(
        self,
        registry: SkillRegistry,
        policy: PolicyEngine,
        budget: BudgetManager,
        context: ExecutionContext,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.budget = budget
        self.context = context

    def _rejected(self, call: SkillCall, reason: str, message: str) -> SkillResult:
        result = SkillResult(
            call_id=call.call_id,
            skill_name=call.skill_name,
            status=SkillStatus.REJECTED,
            reason_code=reason,
            message=message,
        )
        self.context.events.append("skill.rejected", call.actor.value, {"call": call, "result": result})
        self.context.state.sequence = self.context.events.sequence
        checkpoint_state(self.context.store, self.context.state)
        return result

    @staticmethod
    def _output_schema_error(spec: SkillSpec, result: SkillResult) -> str | None:
        payload = to_jsonable(result)
        if not isinstance(payload, dict):
            return "result is not an object"
        required = {str(key) for key in spec.output_schema.get("required", [])}
        missing = sorted(required.difference(payload))
        if missing:
            return f"missing output fields: {', '.join(missing)}"
        allowed_status = spec.output_schema.get("properties", {}).get("status", {}).get("enum")
        if allowed_status and payload.get("status") not in allowed_status:
            return f"invalid output status: {payload.get('status')}"
        if not isinstance(payload.get("facts"), dict):
            return "facts must be an object"
        if not isinstance(payload.get("artifacts"), list) or not isinstance(payload.get("evidence"), list):
            return "artifacts and evidence must be arrays"
        return None

    def invoke(self, call: SkillCall) -> SkillResult:
        entry = self.registry.get(call.skill_name)
        if entry is None:
            return self._rejected(call, "UNKNOWN_SKILL", f"unknown skill: {call.skill_name}")
        spec, handler = entry
        missing = [key for key in spec.required_inputs if key not in call.inputs]
        if missing:
            return self._rejected(call, "INVALID_INPUT", f"missing required inputs: {', '.join(missing)}")
        if spec.input_schema.get("additionalProperties") is False:
            allowed_inputs = set(spec.input_schema.get("properties", {}))
            unexpected = sorted(set(call.inputs).difference(allowed_inputs))
            if unexpected:
                return self._rejected(call, "INVALID_INPUT", f"unexpected inputs: {', '.join(unexpected)}")
        authorized, reason, message = self.policy.authorize(spec, call, self.context.state)
        if not authorized:
            return self._rejected(call, reason, message)
        fingerprint = sha256_bytes(
            canonical_json_bytes(
                {
                    "skill_name": call.skill_name,
                    "actor": call.actor.value,
                    "inputs": call.inputs,
                    "approval": call.approval,
                }
            )
        )
        cached = self.context.state.results.get(call.idempotency_key)
        if cached is not None:
            previous_fingerprint = self.context.state.idempotency_fingerprints.get(call.idempotency_key)
            if previous_fingerprint != fingerprint:
                return self._rejected(
                    call,
                    "IDEMPOTENCY_KEY_CONFLICT",
                    "the idempotency key is already bound to a different actor, skill, input, or approval",
                )
            result = SkillResult(
                call_id=call.call_id,
                skill_name=call.skill_name,
                status=SkillStatus.CACHED,
                reason_code="IDEMPOTENT_REPLAY",
                message="returned the previously committed result",
                facts=cached.facts,
                artifacts=cached.artifacts,
                evidence=cached.evidence,
                retryable=False,
            )
            self.context.events.append("skill.cached", call.actor.value, {"call": call, "result": result})
            self.context.state.sequence = self.context.events.sequence
            checkpoint_state(self.context.store, self.context.state)
            return result
        within_budget, budget_name = self.budget.precheck(spec, self.context.state)
        if not within_budget:
            return self._rejected(call, "BUDGET_EXHAUSTED", f"pre-execution limit reached: {budget_name}")

        self.context.events.append(
            "skill.started",
            call.actor.value,
            {"call": call, "skill_spec": spec},
        )
        self.budget.consume(spec, self.context.state)
        try:
            result = handler(call, self.context)
            if result.call_id != call.call_id or result.skill_name != call.skill_name:
                raise ValueError("handler returned a result for a different call")
        except Exception as exc:  # the boundary always normalizes tool failures
            result = SkillResult(
                call_id=call.call_id,
                skill_name=call.skill_name,
                status=SkillStatus.FAILED,
                reason_code="SKILL_EXCEPTION",
                message=f"{type(exc).__name__}: {exc}",
                retryable=False,
            )

        try:
            schema_error = self._output_schema_error(spec, result)
        except Exception as exc:
            schema_error = f"output serialization failed: {type(exc).__name__}: {exc}"
        if schema_error is not None:
            result = SkillResult(
                call_id=call.call_id,
                skill_name=call.skill_name,
                status=SkillStatus.FAILED,
                reason_code="OUTPUT_SCHEMA_VIOLATION",
                message=schema_error,
            )

        for artifact in result.artifacts:
            if not self.context.store.verify(artifact):
                result = SkillResult(
                    call_id=call.call_id,
                    skill_name=call.skill_name,
                    status=SkillStatus.FAILED,
                    reason_code="ARTIFACT_VERIFICATION_FAILED",
                    message=f"artifact failed size/hash verification: {artifact.path}",
                )
                break
        if result.status is not SkillStatus.FAILED:
            for evidence in result.evidence:
                admitted, reason, message = self.context.evidence_admission_policy.admit(evidence)
                if not admitted:
                    result = SkillResult(
                        call_id=call.call_id,
                        skill_name=call.skill_name,
                        status=SkillStatus.FAILED,
                        reason_code=reason,
                        message=message,
                    )
                    break
        self.context.state.results[call.idempotency_key] = result
        self.context.state.idempotency_fingerprints[call.idempotency_key] = fingerprint
        self.context.state.artifacts.extend(result.artifacts)
        self.context.state.evidence.extend(result.evidence)
        self.context.events.append("skill.completed", call.actor.value, {"result": result})
        self.context.state.sequence = self.context.events.sequence
        checkpoint_state(self.context.store, self.context.state)
        return result
