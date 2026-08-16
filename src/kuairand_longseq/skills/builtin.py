"""Deterministic, typed Skills for the first CausalFeatureOps workflow."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from kuairand_longseq.harness.contracts import (
    EVIDENCE_TIER_RANK, EvidenceEnvelope, EvidenceStatus, EvidenceTier, RoleId, SideEffect,
    SkillCall, SkillResult, SkillSpec, SkillStatus, to_jsonable,
)
from kuairand_longseq.harness.runtime import ExecutionContext, SkillRegistry


def _result(call: SkillCall, *, message: str, status: SkillStatus = SkillStatus.SUCCEEDED,
            reason: str = "OK", facts: dict[str, Any] | None = None,
            artifacts: tuple[Any, ...] = (), evidence: tuple[Any, ...] = ()) -> SkillResult:
    return SkillResult(call.call_id, call.skill_name, status, reason, message,
                       facts or {}, artifacts, evidence, False)


def register_research_contract(call: SkillCall, context: ExecutionContext) -> SkillResult:
    payload = {
        "schema_version": "1.0", "objective": call.inputs["objective"],
        "research_boundary": call.inputs["research_boundary"],
        "task_graph": call.inputs["task_graph"], "mode": context.state.mode.value,
        "scientific_evidence": False, "checkpoint_eligible": False,
    }
    contract = context.store.write_json("contracts/run_contract.json", payload,
                                        producer=call.actor.value, role="run_contract")
    graph = context.store.write_json("task_graph.json", call.inputs["task_graph"],
                                     producer=call.actor.value, role="task_graph")
    return _result(call, message="The objective and bounded task graph were registered.",
                   facts={"task_count": len(call.inputs["task_graph"]), "scientific_evidence": False},
                   artifacts=(contract, graph))


def audit_project_boundary(call: SkillCall, context: ExecutionContext) -> SkillResult:
    policy = dict(call.inputs["data_policy"])
    expected_false = ("reclean_silver", "read_raw_data", "read_sealed_splits", "recursive_data_discovery")
    violations = [f"{key} must be false" for key in expected_false if policy.get(key) is not False]
    payload = {
        "status": "pass" if not violations else "fail", "data_policy": policy,
        "research_status": call.inputs["research_status"], "violations": violations,
        "datasets_read_by_this_skill": [],
        "claim_boundary": "Policy registration audit only; no Parquet was read or re-cleaned.",
    }
    artifact = context.store.write_json("audits/project_boundary.json", payload,
                                        producer=call.actor.value, role="policy_audit")
    if violations:
        return _result(call, status=SkillStatus.REJECTED, reason="DATA_POLICY_VIOLATION",
                       message="The declared workflow crosses the current boundary.",
                       facts={"violations": violations}, artifacts=(artifact,))
    return _result(call, message="The workflow stays within the no-reclean, no-sealed-data boundary.",
                   facts={"policy_passed": True, "datasets_read": 0}, artifacts=(artifact,))


def detect_temporal_leakage(call: SkillCall, context: ExecutionContext) -> SkillResult:
    fields = {str(x) for x in call.inputs["feature_fields"]}
    denied = {str(x) for x in call.inputs["denied_fields"]}
    overlap = sorted(fields & denied)
    payload = {"feature_fields": sorted(fields), "denied_fields_checked": sorted(denied),
               "violations": overlap,
               "strict_rule": "history_time < target_time; same-timestamp feedback is invisible"}
    artifact = context.store.write_json("audits/temporal_leakage.json", payload,
                                        producer=call.actor.value, role="leakage_audit")
    if overlap:
        return _result(call, status=SkillStatus.REJECTED, reason="TEMPORAL_LEAKAGE_FIELD",
                       message="Forbidden current/future feedback appears in the feature interface.",
                       facts={"violating_fields": overlap}, artifacts=(artifact,))
    return _result(call, message="No denylisted field is present in the feature interface.",
                   facts={"violating_fields": []}, artifacts=(artifact,))


def propose_feature_specs(call: SkillCall, context: ExecutionContext) -> SkillResult:
    gpu_train_only_complete = call.inputs["research_status"].get("gpu_experiment_completed") is True
    strict_history_status = (
        "train_only_gpu_snapshot_available_pending_agent_admission"
        if gpu_train_only_complete else "repair_evidence_required"
    )
    specs = [
        {"feature_family": "static_intrinsic", "availability": "model_contract_candidate",
         "examples": ["author_id_category", "video_duration_state", "tag_tokens"],
         "point_in_time_rule": "intrinsic metadata only"},
        {"feature_family": "strict_statistical_history", "availability": strict_history_status,
         "examples": ["prior_event_count", "prior_long_view_count", "recent_10_rate"],
         "point_in_time_rule": "completed user-time batches strictly before target_time"},
        {"feature_family": "short_sequence", "availability": "not_run",
         "examples": ["last_k_event_embeddings", "masked_time_gaps"],
         "point_in_time_rule": "strict prefix; no target feedback"},
        {"feature_family": "long_sequence", "availability": "not_run",
         "examples": ["hierarchical_sequence_encoder", "long_horizon_interest_state"],
         "point_in_time_rule": "strict prefix; same target manifest as baselines"},
    ]
    payload = {
        "schema_version": "1.0", "task": "candidate_long_view_probability_prediction",
        "target_scene": "tab=1", "history_visibility": "history_time < target_time",
        "feature_policy": call.inputs["feature_policy"], "research_status": call.inputs["research_status"],
        "feature_specs": specs,
        "promotion_status": "blocked_pending_agent_admission_validation_and_sequence_evidence",
        "scientific_result": None,
    }
    artifact = context.store.write_json("feature_specs/candidate_feature_specs.json", payload,
                                        producer=call.actor.value, role="feature_spec")
    return _result(call, message="Feature families were specified; sequence promotion remains blocked.",
                   facts={"feature_family_count": len(specs), "promotion_eligible": False},
                   artifacts=(artifact,))


def request_research_evidence(call: SkillCall, context: ExecutionContext) -> SkillResult:
    provider_admitted, provider_reason, provider_message = context.evidence_admission_policy.admit_provider(
        context.evidence_provider
    )
    if not provider_admitted:
        return _result(
            call,
            status=SkillStatus.REJECTED,
            reason=provider_reason,
            message=provider_message,
            facts={"provider_id": getattr(context.evidence_provider, "provider_id", None)},
        )
    envelope = context.evidence_provider.fetch(dict(call.inputs["request"]))
    admitted, admission_reason, admission_message = context.evidence_admission_policy.admit(envelope)
    if not admitted:
        envelope = EvidenceEnvelope(
            evidence_id=f"rejected:{envelope.evidence_id}",
            evidence_kind=envelope.evidence_kind,
            status=EvidenceStatus.INVALID,
            tier=EvidenceTier.NONE,
            claim_eligible=False,
            scope={"requested_task": call.inputs["request"].get("task")},
            provenance={"rejected_provider_id": envelope.provider_id},
            limitations=(admission_message,),
            reason_code=admission_reason,
            provider_id="harness_evidence_admission",
        )
    artifact = context.store.write_json("evidence/research_evidence_status.json", envelope,
                                        producer=call.actor.value, role="evidence_status")
    archived = []
    snapshotter = getattr(context.evidence_provider, "snapshot_bytes", None)
    if envelope.status is EvidenceStatus.VERIFIED and callable(snapshotter):
        archived.append(
            context.store.write_bytes(
                "evidence/source_manifest.yaml",
                snapshotter(),
                media_type="application/yaml",
                producer=call.actor.value,
                role="source_evidence_manifest",
            )
        )
    verified = envelope.status is EvidenceStatus.VERIFIED
    reason = "EVIDENCE_VERIFIED" if verified else (envelope.reason_code or "EVIDENCE_UNAVAILABLE")
    return _result(call, reason=reason,
                   message="Verified evidence accepted." if verified else
                           "Research evidence is unavailable or invalid; no metric claim is permitted.",
                   facts={"evidence_status": envelope.status.value, "evidence_tier": envelope.tier.value,
                          "claim_eligible": envelope.claim_eligible},
                   artifacts=(artifact, *archived), evidence=(envelope,))


def evaluate_research_evidence(call: SkillCall, context: ExecutionContext) -> SkillResult:
    evidence_id = str(call.inputs["evidence_id"])
    envelope = next((x for x in reversed(context.state.evidence) if x.evidence_id == evidence_id), None)
    required = [str(x) for x in call.inputs["required_gates"]]
    frozen_required = [str(x) for x in context.config["required_release_gates"]]
    if required != frozen_required:
        return _result(
            call,
            status=SkillStatus.REJECTED,
            reason="GATE_CONTRACT_MISMATCH",
            message="The requested gate list differs from the frozen run contract.",
            facts={"requested_gates": required, "frozen_gates": frozen_required},
        )
    if envelope is None:
        verdict = {"evidence_id": evidence_id, "release_eligible": False,
                   "reason_code": "EVIDENCE_REFERENCE_NOT_FOUND", "gate_results": {}}
    elif not envelope.claim_eligible:
        verdict = {"evidence_id": evidence_id, "release_eligible": False,
                   "reason_code": envelope.reason_code or "EVIDENCE_NOT_CLAIM_ELIGIBLE",
                   "gate_results": {}}
    else:
        gates = {name: envelope.gates.get(name, "missing") for name in required}
        passed = all(value == "pass" for value in gates.values())
        verdict = {"evidence_id": evidence_id, "release_eligible": passed,
                   "reason_code": "ALL_FROZEN_GATES_PASS" if passed else "FROZEN_GATE_FAILED_OR_MISSING",
                   "gate_results": gates, "tier": envelope.tier.value}
    artifact = context.store.write_json("evaluation/gate_verdict.json", verdict,
                                        producer=call.actor.value, role="gate_verdict")
    return _result(call, reason=str(verdict["reason_code"]),
                   message="Frozen gates were evaluated without altering evidence.",
                   facts=verdict, artifacts=(artifact,))


def review_claim_boundaries(call: SkillCall, context: ExecutionContext) -> SkillResult:
    verified = [x for x in context.state.evidence if x.claim_eligible]
    highest = max((x.tier for x in verified), key=EVIDENCE_TIER_RANK.__getitem__, default=EvidenceTier.NONE)
    allowed = [
        "The deterministic workflow enforced policy and produced hash-addressed run artifacts.",
        "Missing research evidence is explicit and blocks scientific release.",
    ]
    missing: list[str] = []
    if highest is EvidenceTier.NONE:
        missing += ["Agent-admissible model-comparison predictions", "frozen gate verdicts",
                    "complete GPU experiment provenance"]
    elif highest is EvidenceTier.TRAIN_ONLY:
        allowed.append("Verified Train-only evidence is exploratory within its registered scope.")
        missing += ["Validation evidence", "sealed-test evidence"]
    else:
        allowed.append("The offline candidate may be described only within its verified scope and gates.")
    forbidden = ["online causal lift", "arbitrary-policy off-policy evaluation",
                 "full-catalog retrieval quality", "long-sequence superiority without paired predictions",
                 "GPU speedup without a matched benchmark"]
    payload = {"evidence_tier": highest.value, "allowed_claims": allowed,
               "unsupported_claims": forbidden, "missing_evidence": missing,
               "review_scope": call.inputs["review_scope"]}
    artifact = context.store.write_json("safety/claim_review.json", payload,
                                        producer=call.actor.value, role="claim_review")
    context.state.allowed_claims, context.state.unsupported_claims = allowed, forbidden
    context.state.missing_evidence = missing
    return _result(call, reason="CLAIMS_BOUNDED", message="Claims were bound to the evidence tier.",
                   facts={"evidence_tier": highest.value, "missing_evidence_count": len(missing)},
                   artifacts=(artifact,))


def assess_release_readiness(call: SkillCall, context: ExecutionContext) -> SkillResult:
    minimum = EvidenceTier(str(call.inputs["minimum_tier"]))
    highest = max((x.tier for x in context.state.evidence if x.claim_eligible),
                  key=EVIDENCE_TIER_RANK.__getitem__, default=EvidenceTier.NONE)
    current_evidence = next((x for x in reversed(context.state.evidence) if x.claim_eligible), None)
    frozen_gates = set(str(x) for x in context.config["required_release_gates"])
    verdicts = [
        r for r in context.state.results.values()
        if r.skill_name == "evaluate_research_evidence"
        and r.status is SkillStatus.SUCCEEDED
        and current_evidence is not None
        and r.facts.get("evidence_id") == current_evidence.evidence_id
        and set(r.facts.get("gate_results", {})) == frozen_gates
        and all(value == "pass" for value in r.facts.get("gate_results", {}).values())
    ]
    gates_pass = bool(verdicts) and bool(verdicts[-1].facts.get("release_eligible"))
    ready = EVIDENCE_TIER_RANK[highest] >= EVIDENCE_TIER_RANK[minimum] and gates_pass
    payload = {"ready_for_human_approval": ready, "minimum_tier": minimum.value,
               "observed_tier": highest.value, "frozen_gates_passed": gates_pass,
               "publication_performed": False,
               "reason_code": "READY_FOR_APPROVAL" if ready else "RESEARCH_EVIDENCE_INCOMPLETE"}
    artifact = context.store.write_json("release/readiness.json", payload,
                                        producer=call.actor.value, role="release_readiness")
    return _result(call, reason=str(payload["reason_code"]),
                   message="Ready for human approval." if ready else "Publication remains disabled.",
                   facts=payload, artifacts=(artifact,))


def publish_feature_package(call: SkillCall, context: ExecutionContext) -> SkillResult:
    payload = {"package_version": call.inputs["package_version"], "approval": to_jsonable(call.approval),
               "subject_sha256": call.inputs["subject_sha256"],
               "contract_sha256": context.state.contract_sha256,
               "evidence_ids": [x.evidence_id for x in context.state.evidence if x.claim_eligible],
               "status": "released", "rollback_supported": True}
    artifact = context.store.write_json("release/release_manifest.json", payload,
                                        producer=call.actor.value, role="release_manifest")
    return _result(call, message="Approved feature package released.", facts={"released": True}, artifacts=(artifact,))


def rollback_feature_package(call: SkillCall, context: ExecutionContext) -> SkillResult:
    payload = {"release_version": call.inputs["release_version"],
               "subject_sha256": call.inputs["subject_sha256"],
               "rollback_reason": call.inputs["rollback_reason"], "approval": to_jsonable(call.approval),
               "status": "rolled_back", "history_preserved": True}
    artifact = context.store.write_json("release/rollback_manifest.json", payload,
                                        producer=call.actor.value, role="rollback_manifest")
    return _result(call, message="Release rolled back with history preserved.",
                   facts={"rolled_back": True}, artifacts=(artifact,))


def build_builtin_registry() -> SkillRegistry:
    registry = SkillRegistry()
    entries = (
        (SkillSpec("register_research_contract", "1.0.0", "Register objective and task graph.",
                   (RoleId.MANAGER,), SideEffect.WRITE_RUN_ARTIFACT,
                   ("objective", "research_boundary", "task_graph"), 30, 0, True), register_research_contract),
        (SkillSpec("audit_project_boundary", "1.0.0", "Audit data boundary.",
                   (RoleId.DATA_AUDITOR,), SideEffect.WRITE_RUN_ARTIFACT,
                   ("data_policy", "research_status"), 30, 0, True), audit_project_boundary),
        (SkillSpec("detect_temporal_leakage", "1.0.0", "Reject denylisted fields.",
                   (RoleId.DATA_AUDITOR,), SideEffect.WRITE_RUN_ARTIFACT,
                   ("feature_fields", "denied_fields"), 30, 0, True), detect_temporal_leakage),
        (SkillSpec("propose_feature_specs", "1.0.0", "Create feature specifications.",
                   (RoleId.FEATURE_MINER,), SideEffect.WRITE_RUN_ARTIFACT,
                   ("feature_policy", "research_status"), 30, 0, True), propose_feature_specs),
        (SkillSpec("request_research_evidence", "1.0.0", "Verify an evidence manifest.",
                   (RoleId.CAUSAL_EVALUATOR,), SideEffect.WRITE_RUN_ARTIFACT,
                   ("request",), 60, 0, True), request_research_evidence),
        (SkillSpec("evaluate_research_evidence", "1.0.0", "Apply frozen gates.",
                   (RoleId.CAUSAL_EVALUATOR,), SideEffect.WRITE_RUN_ARTIFACT,
                   ("evidence_id", "required_gates"), 30, 0, True), evaluate_research_evidence),
        (SkillSpec("review_claim_boundaries", "1.0.0", "Bind claims to evidence.",
                   (RoleId.SAFETY_REVIEWER,), SideEffect.WRITE_RUN_ARTIFACT,
                   ("review_scope",), 30, 0, True), review_claim_boundaries),
        (SkillSpec("assess_release_readiness", "1.0.0", "Assess without publishing.",
                   (RoleId.FEATURE_PUBLISHER,), SideEffect.WRITE_RUN_ARTIFACT,
                   ("minimum_tier",), 30, 0, True), assess_release_readiness),
        (SkillSpec("publish_feature_package", "1.0.0", "Publish approved package.",
                   (RoleId.FEATURE_PUBLISHER,), SideEffect.PUBLISH, ("package_version", "subject_sha256"), 30, 0, True,
                   minimum_evidence_tier=EvidenceTier.VALIDATION,
                   approval_action="publish_feature_package"), publish_feature_package),
        (SkillSpec("rollback_feature_package", "1.0.0", "Rollback approved release.",
                   (RoleId.FEATURE_PUBLISHER,), SideEffect.ROLLBACK,
                   ("release_version", "rollback_reason", "subject_sha256"), 30, 0, True,
                   approval_action="rollback_feature_package"), rollback_feature_package),
    )
    common_output_schema = {
        "type": "object",
        "required": ["call_id", "skill_name", "status", "reason_code", "facts", "artifacts", "evidence"],
        "properties": {
            "status": {"enum": ["succeeded", "rejected", "failed", "cached"]},
            "reason_code": {"type": "string"},
            "facts": {"type": "object"},
            "artifacts": {"type": "array"},
            "evidence": {"type": "array"},
        },
    }
    common_failure_codes = (
        "INVALID_INPUT", "ROLE_NOT_AUTHORIZED", "SIDE_EFFECT_NOT_AUTHORIZED",
        "BUDGET_EXHAUSTED", "SKILL_EXCEPTION", "ARTIFACT_VERIFICATION_FAILED",
        "OUTPUT_SCHEMA_VIOLATION", "EVIDENCE_PROVIDER_IMPLEMENTATION_NOT_TRUSTED",
    )
    for spec, handler in entries:
        typed_spec = replace(
            spec,
            input_schema={
                "type": "object",
                "required": list(spec.required_inputs),
                "properties": {key: {} for key in spec.required_inputs},
                "additionalProperties": False,
            },
            output_schema=common_output_schema,
            failure_reason_codes=common_failure_codes,
            concurrency_safe=spec.side_effect is SideEffect.PURE,
        )
        registry.register(typed_spec, handler)
    return registry
