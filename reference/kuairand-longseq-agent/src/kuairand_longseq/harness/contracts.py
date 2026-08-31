"""Typed contracts for the CausalFeatureOps agent harness.

The harness deliberately stores facts, actions, and artifact references.  It
does not persist hidden chain-of-thought or in-memory data frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class RunMode(str, Enum):
    DESIGN_ONLY = "design_only"
    RESEARCH = "research"


class RunPhase(str, Enum):
    CREATED = "created"
    CONTRACT_REGISTERED = "contract_registered"
    INPUTS_AUDITED = "inputs_audited"
    FEATURES_PROPOSED = "features_proposed"
    EVIDENCE_EVALUATED = "evidence_evaluated"
    CLAIMS_REVIEWED = "claims_reviewed"
    WAITING_FOR_EVIDENCE = "waiting_for_evidence"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    RELEASED = "released"
    ROLLED_BACK = "rolled_back"
    BLOCKED = "blocked"
    FAILED = "failed"


TERMINAL_PHASES = {
    RunPhase.WAITING_FOR_EVIDENCE,
    RunPhase.WAITING_FOR_APPROVAL,
    RunPhase.RELEASED,
    RunPhase.ROLLED_BACK,
    RunPhase.BLOCKED,
    RunPhase.FAILED,
}


class RoleId(str, Enum):
    MANAGER = "manager"
    DATA_AUDITOR = "data_auditor"
    FEATURE_MINER = "feature_miner"
    CAUSAL_EVALUATOR = "causal_evaluator"
    SAFETY_REVIEWER = "safety_reviewer"
    FEATURE_PUBLISHER = "feature_publisher"


class SideEffect(str, Enum):
    PURE = "pure"
    READ_ONLY = "read_only"
    WRITE_RUN_ARTIFACT = "write_run_artifact"
    SEALED_READ = "sealed_read"
    PUBLISH = "publish"
    ROLLBACK = "rollback"


class SkillStatus(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    CACHED = "cached"


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    PROVISIONAL = "provisional"
    SYNTHETIC = "synthetic"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    STALE = "stale"


class EvidenceTier(str, Enum):
    NONE = "none"
    SYSTEM = "system"
    TRAIN_ONLY = "train_only"
    VALIDATION = "validation"
    SEALED_TEST = "sealed_test"


EVIDENCE_TIER_RANK = {
    EvidenceTier.NONE: 0,
    EvidenceTier.SYSTEM: 1,
    EvidenceTier.TRAIN_ONLY: 2,
    EvidenceTier.VALIDATION: 3,
    EvidenceTier.SEALED_TEST: 4,
}


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    size_bytes: int
    sha256: str
    media_type: str
    producer: str
    role: str = "supporting"


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    evidence_id: str
    evidence_kind: str
    status: EvidenceStatus
    tier: EvidenceTier
    claim_eligible: bool
    scope: Mapping[str, JsonValue]
    provenance: Mapping[str, JsonValue] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: Mapping[str, JsonValue] = field(default_factory=dict)
    gates: Mapping[str, JsonValue] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    reason_code: str | None = None
    provider_id: str = "unknown"

    def __post_init__(self) -> None:
        if self.status is not EvidenceStatus.VERIFIED and self.claim_eligible:
            raise ValueError("only verified evidence may be claim eligible")
        if self.status is EvidenceStatus.UNAVAILABLE and self.metrics:
            raise ValueError("unavailable evidence cannot contain metric values")
        if self.status is EvidenceStatus.SYNTHETIC and self.claim_eligible:
            raise ValueError("synthetic evidence cannot support scientific claims")


@dataclass(frozen=True, slots=True)
class ApprovalToken:
    token_id: str
    action: str
    contract_sha256: str
    approved_by: str
    status: str
    subject_sha256: str


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    role: RoleId
    display_name: str
    version: str
    objective: str
    allowed_skills: tuple[str, ...]
    allowed_side_effects: tuple[SideEffect, ...]
    forbidden_actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillSpec:
    name: str
    version: str
    description: str
    allowed_roles: tuple[RoleId, ...]
    side_effect: SideEffect
    required_inputs: tuple[str, ...]
    timeout_seconds: int
    max_retries: int
    idempotent: bool
    minimum_evidence_tier: EvidenceTier = EvidenceTier.NONE
    approval_action: str | None = None
    input_schema: Mapping[str, JsonValue] = field(default_factory=dict)
    output_schema: Mapping[str, JsonValue] = field(default_factory=dict)
    failure_reason_codes: tuple[str, ...] = ()
    concurrency_safe: bool = False


@dataclass(frozen=True, slots=True)
class SkillCall:
    call_id: str
    skill_name: str
    actor: RoleId
    inputs: Mapping[str, JsonValue]
    idempotency_key: str
    approval: ApprovalToken | None = None


@dataclass(frozen=True, slots=True)
class SkillResult:
    call_id: str
    skill_name: str
    status: SkillStatus
    reason_code: str
    message: str
    facts: Mapping[str, JsonValue] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    evidence: tuple[EvidenceEnvelope, ...] = ()
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class BudgetLimit:
    max_steps: int
    max_skill_calls: int
    max_write_calls: int
    max_retries: int
    max_sealed_reads: int
    max_publish_calls: int
    max_llm_calls: int = 0
    max_gpu_seconds: int = 0


@dataclass(slots=True)
class BudgetUsage:
    steps: int = 0
    skill_calls: int = 0
    write_calls: int = 0
    retries: int = 0
    sealed_reads: int = 0
    publish_calls: int = 0
    llm_calls: int = 0
    gpu_seconds: int = 0


@dataclass(slots=True)
class RunState:
    run_id: str
    system_id: str
    mode: RunMode
    objective: str
    contract_sha256: str
    phase: RunPhase = RunPhase.CREATED
    sequence: int = 0
    facts: dict[str, JsonValue] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    evidence: list[EvidenceEnvelope] = field(default_factory=list)
    results: dict[str, SkillResult] = field(default_factory=dict)
    idempotency_fingerprints: dict[str, str] = field(default_factory=dict)
    approvals: list[ApprovalToken] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    allowed_claims: list[str] = field(default_factory=list)
    budget: BudgetUsage = field(default_factory=BudgetUsage)
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunEvent:
    sequence: int
    run_id: str
    event_type: str
    actor: str
    payload: Mapping[str, JsonValue]
    previous_sha256: str
    event_sha256: str


def to_jsonable(value: Any) -> JsonValue:
    """Convert contracts to deterministic, strict JSON-compatible values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, set):
        converted = [to_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: str(item))
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
