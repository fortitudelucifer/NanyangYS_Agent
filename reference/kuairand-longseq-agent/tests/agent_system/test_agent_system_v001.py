from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from kuairand_longseq.agents.roles import build_agent_identities
from kuairand_longseq.agents.team_runtime import AgentTask
from kuairand_longseq.agents.workflow import CausalFeatureOpsWorkflow, WorkflowError, verify_run_directory
from kuairand_longseq.evidence import ManifestEvidenceProvider, NullEvidenceProvider
from kuairand_longseq.harness.contracts import (
    ApprovalToken, BudgetLimit, EvidenceEnvelope, EvidenceStatus, EvidenceTier, RoleId,
    RunMode, RunPhase, RunState, SideEffect, SkillCall, SkillResult, SkillStatus,
)
from kuairand_longseq.harness.runtime import BudgetManager
from kuairand_longseq.harness.state_machine import transition
from kuairand_longseq.harness.storage import EventStore
from scripts.run_agent_system_demo_v001 import exit_code_for


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "agent_system_v001.yaml"
EXPECTED_PROVENANCE = {
    "contract_sha256": "a" * 64,
    "code_sha256": "b" * 64,
    "input_manifest_sha256": "c" * 64,
    "model_config_sha256": "d" * 64,
    "authorization_sha256": "f" * 64,
    "target_manifest_sha256": "e" * 64,
}


def _write_config(tmp_path: Path, mutate=None) -> Path:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if mutate:
        mutate(payload)
    path = tmp_path / "agent_system.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_six_roles_have_distinct_capabilities() -> None:
    identities = build_agent_identities()
    assert len(identities) == 6
    assert len({tuple(identity.allowed_skills) for identity in identities.values()}) == 6
    assert SideEffect.PUBLISH in identities[RoleId.FEATURE_PUBLISHER].allowed_side_effects
    assert SideEffect.PUBLISH not in identities[RoleId.MANAGER].allowed_side_effects


def test_all_skills_publish_typed_contract_metadata() -> None:
    from kuairand_longseq.skills.builtin import build_builtin_registry

    specs = build_builtin_registry().specs
    assert len(specs) == 10
    for spec in specs:
        assert spec.version
        assert spec.timeout_seconds > 0
        assert spec.input_schema["type"] == "object"
        assert set(spec.input_schema["required"]) == set(spec.required_inputs)
        assert spec.output_schema["type"] == "object"
        assert "reason_code" in spec.output_schema["required"]
        assert spec.failure_reason_codes


def test_offline_demo_waits_for_evidence_and_verifies_artifacts(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(
        config_path=CONFIG,
        output_root=tmp_path,
        evidence_provider=NullEvidenceProvider(),
        run_id="offline-demo",
    )
    run_dir = workflow.run()
    assert workflow.state.phase is RunPhase.WAITING_FOR_EVIDENCE
    assert workflow.state.budget.llm_calls == 0
    assert workflow.state.budget.gpu_seconds == 0
    assert workflow.state.budget.sealed_reads == 0
    assert verify_run_directory(run_dir)["verified"] is True
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["research_result_available"] is False
    assert manifest["data_access"] == {
        "raw": 0,
        "sealed": 0,
        "silver_parquet": 0,
        "external_evidence_manifest_reads": 0,
        "external_evidence_artifact_hash_reads": 0,
        "external_evidence_bytes_read_for_verification": 0,
    }
    implementation = manifest["implementation_sha256"]
    assert implementation
    for relative_path, digest in implementation.items():
        assert hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest() == digest
    events = EventStore.read_verified(run_dir / "events.jsonl")
    started_roles = {event.actor for event in events if event.event_type == "skill.started"}
    assert {role.value for role in RoleId}.issubset(started_roles)


def test_leakage_field_blocks_before_evidence_provider(tmp_path: Path) -> None:
    config = _write_config(tmp_path, lambda p: p["feature_policy"]["demo_feature_fields"].append("long_view"))

    class ExplodingProvider:
        provider_id = "must_not_be_called"
        version = "1"

        def fetch(self, request):
            raise AssertionError("downstream evidence provider should not run after leakage rejection")

    workflow = CausalFeatureOpsWorkflow(
        config_path=config, output_root=tmp_path / "runs", evidence_provider=ExplodingProvider(), run_id="leakage"
    )
    run_dir = workflow.run()
    assert workflow.state.phase is RunPhase.BLOCKED
    assert "TEMPORAL_LEAKAGE_FIELD" in (workflow.state.terminal_reason or "")
    assert not (run_dir / "evidence" / "research_evidence_status.json").exists()


def test_verified_manifest_reaches_human_approval_not_release(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    prediction = evidence_root / "paired_predictions.json"
    prediction.write_text('{"rows": 2, "note": "test fixture"}\n', encoding="utf-8")
    digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
    required_gates = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["required_release_gates"]
    payload = {
        "schema_version": "1.0", "evidence_id": "ev-test", "evidence_kind": "paired_model_comparison",
        "status": "complete", "tier": "validation", "claim_eligible": True,
        "execution_authorized": True,
        "contract_sha256": "a" * 64, "code_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64, "model_config_sha256": "d" * 64,
        "authorization_sha256": "f" * 64,
        "models": ["static_baseline", "strict_statistical_history", "short_sequence", "long_sequence"],
        "scope": {"dataset": "fixture", "split": "validation",
                  "task": "candidate_long_view_probability_prediction", "target_manifest_sha256": "e" * 64},
        "metrics": {"average_precision_delta": 0.01},
        "gates": {name: "pass" for name in required_gates},
        "limitations": ["Fixture tests the handoff, not KuaiRand performance."],
        "artifacts": [{"path": prediction.name, "size_bytes": prediction.stat().st_size,
                       "sha256": digest, "media_type": "application/json"}],
    }
    manifest = evidence_root / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    def freeze_expectations(config):
        config["evidence_request"]["expected_dataset"] = "fixture"
        config["evidence_request"]["expected_provenance"] = EXPECTED_PROVENANCE

    config = _write_config(tmp_path, freeze_expectations)
    workflow = CausalFeatureOpsWorkflow(
        config_path=config, output_root=tmp_path / "runs",
        evidence_provider=ManifestEvidenceProvider(manifest), run_id="verified-evidence",
    )
    run_dir = workflow.run()
    assert workflow.state.phase is RunPhase.WAITING_FOR_APPROVAL
    assert not (run_dir / "release" / "release_manifest.json").exists()
    assert (run_dir / "evidence" / "source_manifest.yaml").is_file()
    assert workflow.state.evidence[-1].claim_eligible is True
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["data_access"]["external_evidence_manifest_reads"] == 1
    assert run_manifest["data_access"]["external_evidence_artifact_hash_reads"] == 1
    assert run_manifest["data_access"]["external_evidence_bytes_read_for_verification"] > 0


def test_spoofed_provider_implementation_is_rejected_before_fetch(tmp_path: Path) -> None:
    class SpoofedManifestProvider:
        provider_id = "research_evidence_manifest"
        version = "1.0.0"
        called = False

        def fetch(self, request):
            self.called = True
            raise AssertionError("an unregistered provider implementation must never execute")

    provider = SpoofedManifestProvider()
    workflow = CausalFeatureOpsWorkflow(
        config_path=CONFIG,
        output_root=tmp_path,
        evidence_provider=provider,
        run_id="spoofed-provider",
    )
    workflow.run()
    assert provider.called is False
    assert workflow.state.phase is RunPhase.BLOCKED
    assert "EVIDENCE_PROVIDER_IMPLEMENTATION_NOT_TRUSTED" in (workflow.state.terminal_reason or "")


def test_task_dependency_is_enforced_before_skill_execution(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="dependency")
    call = SkillCall(
        "call-dependency",
        "audit_project_boundary",
        RoleId.DATA_AUDITOR,
        {"data_policy": {}, "research_status": {}},
        "dependency-key",
    )
    result = workflow.team.dispatch(
        AgentTask("T02", RoleId.DATA_AUDITOR, call, depends_on=("T01",))
    )
    assert result.status is SkillStatus.REJECTED
    assert result.reason_code == "TASK_DEPENDENCY_UNSATISFIED"
    assert workflow.state.budget.skill_calls == 0


def test_task_role_actor_mismatch_is_normalized(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="role-mismatch")
    call = SkillCall(
        "call-role-mismatch",
        "audit_project_boundary",
        RoleId.DATA_AUDITOR,
        {"data_policy": {}, "research_status": {}},
        "role-mismatch-key",
    )
    result = workflow.team.dispatch(AgentTask("T-role", RoleId.MANAGER, call))
    assert result.status is SkillStatus.REJECTED
    assert result.reason_code == "TASK_ROLE_ACTOR_MISMATCH"


def test_skill_output_schema_violation_is_normalized(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="bad-output")
    spec, _ = workflow.registry.get("register_research_contract")

    def bad_handler(call, context):
        return SkillResult(
            call.call_id,
            call.skill_name,
            SkillStatus.SUCCEEDED,
            "BAD_FIXTURE",
            "malformed facts",
            facts=[],  # type: ignore[arg-type]
        )

    workflow.registry._entries["register_research_contract"] = (spec, bad_handler)
    result = workflow.executor.invoke(
        SkillCall(
            "bad-output-call",
            "register_research_contract",
            RoleId.MANAGER,
            {"objective": "o", "research_boundary": {}, "task_graph": []},
            "bad-output-key",
        )
    )
    assert result.status is SkillStatus.FAILED
    assert result.reason_code == "OUTPUT_SCHEMA_VIOLATION"


def test_unserializable_skill_output_is_normalized(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="bad-serialization")
    spec, _ = workflow.registry.get("register_research_contract")

    def bad_handler(call, context):
        return SkillResult(
            call.call_id,
            call.skill_name,
            SkillStatus.SUCCEEDED,
            "BAD_FIXTURE",
            "unserializable facts",
            facts={"bad": object()},  # type: ignore[dict-item]
        )

    workflow.registry._entries["register_research_contract"] = (spec, bad_handler)
    result = workflow.executor.invoke(
        SkillCall(
            "bad-serialization-call",
            "register_research_contract",
            RoleId.MANAGER,
            {"objective": "o", "research_boundary": {}, "task_graph": []},
            "bad-serialization-key",
        )
    )
    assert result.status is SkillStatus.FAILED
    assert result.reason_code == "OUTPUT_SCHEMA_VIOLATION"


def test_run_id_cannot_escape_output_root(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="run_id"):
        CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="../escape")


def test_cli_exit_code_distinguishes_waiting_from_failure() -> None:
    assert exit_code_for("waiting_for_evidence", True) == 0
    assert exit_code_for("waiting_for_approval", True) == 0
    assert exit_code_for("blocked", True) == 3
    assert exit_code_for("failed", True) == 3
    assert exit_code_for("waiting_for_evidence", False) == 2


def test_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "1.0", "evidence_id": "ev-tampered", "evidence_kind": "model_comparison",
        "status": "complete", "tier": "validation", "claim_eligible": True,
        "execution_authorized": True, "scope": {"dataset": "fixture"}, "metrics": {}, "gates": {}, "limitations": [],
        "contract_sha256": "a" * 64, "code_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64, "model_config_sha256": "d" * 64,
        "authorization_sha256": "f" * 64,
        "models": ["baseline"],
        "artifacts": [{"path": artifact.name, "size_bytes": artifact.stat().st_size, "sha256": "0" * 64}],
    }
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    evidence = ManifestEvidenceProvider(manifest).fetch(
        {"expected_dataset": "fixture", "expected_provenance": EXPECTED_PROVENANCE}
    )
    assert evidence.status is EvidenceStatus.INVALID
    assert evidence.claim_eligible is False
    assert evidence.reason_code == "ARTIFACT_HASH_MISMATCH"


def test_duplicate_manifest_key_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "duplicate.yaml"
    manifest.write_text("schema_version: '1.0'\nevidence_id: one\nevidence_id: two\n", encoding="utf-8")
    evidence = ManifestEvidenceProvider(manifest).fetch({})
    assert evidence.status is EvidenceStatus.INVALID
    assert evidence.reason_code == "MANIFEST_PARSE_FAILED"


def test_unknown_manifest_schema_version_is_rejected(tmp_path: Path) -> None:
    payload = {
        "schema_version": "9.9",
        "evidence_id": "ev-schema",
        "evidence_kind": "paired_model_comparison",
        "status": "complete",
        "tier": "validation",
        "claim_eligible": True,
        "execution_authorized": True,
        "scope": {},
        "metrics": {},
        "gates": {},
        "limitations": [],
        "artifacts": [],
        "contract_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "model_config_sha256": "d" * 64,
        "authorization_sha256": "f" * 64,
        "models": ["baseline"],
    }
    manifest = tmp_path / "unsupported.yaml"
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    evidence = ManifestEvidenceProvider(manifest).fetch({})
    assert evidence.status is EvidenceStatus.INVALID
    assert evidence.reason_code == "MANIFEST_SCHEMA_VERSION_UNSUPPORTED"


def test_train_only_manifest_cannot_answer_validation_request(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "1.0", "evidence_id": "ev-train", "evidence_kind": "paired_model_comparison",
        "status": "complete", "tier": "train_only", "claim_eligible": True,
        "execution_authorized": True, "contract_sha256": "a" * 64, "code_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64, "model_config_sha256": "d" * 64,
        "authorization_sha256": "f" * 64,
        "models": ["baseline", "candidate"],
        "scope": {"dataset": "fixture", "task": "candidate_long_view_probability_prediction", "split": "validation",
                  "target_manifest_sha256": "e" * 64},
        "metrics": {"average_precision_delta": 0.01}, "gates": {}, "limitations": [],
        "artifacts": [{"path": artifact.name, "size_bytes": artifact.stat().st_size,
                       "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}],
    }
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    evidence = ManifestEvidenceProvider(manifest).fetch(
        {"evidence_kind": "paired_model_comparison",
         "task": "candidate_long_view_probability_prediction", "split": "validation",
         "expected_dataset": "fixture", "expected_provenance": EXPECTED_PROVENANCE}
    )
    assert evidence.status is EvidenceStatus.INVALID
    assert evidence.reason_code == "EVIDENCE_TIER_SCOPE_MISMATCH"


def test_unavailable_evidence_cannot_contain_metrics() -> None:
    with pytest.raises(ValueError, match="cannot contain metric"):
        EvidenceEnvelope(
            evidence_id="bad", evidence_kind="metric", status=EvidenceStatus.UNAVAILABLE,
            tier=EvidenceTier.NONE, claim_eligible=False, scope={}, metrics={"ap": 0.5},
        )


def test_synthetic_evidence_is_never_claim_eligible() -> None:
    with pytest.raises(ValueError, match="only verified"):
        EvidenceEnvelope(
            evidence_id="synthetic", evidence_kind="metric", status=EvidenceStatus.SYNTHETIC,
            tier=EvidenceTier.SYSTEM, claim_eligible=True, scope={}, metrics={},
        )


def test_budget_rejects_before_next_call() -> None:
    state = RunState("r", "s", RunMode.DESIGN_ONLY, "o", "0" * 64)
    state.budget.skill_calls = 1
    state.budget.steps = 1
    limits = BudgetLimit(1, 1, 1, 0, 0, 0, 0, 0)
    manager = BudgetManager(limits)
    from kuairand_longseq.skills.builtin import build_builtin_registry

    spec, _ = build_builtin_registry().get("register_research_contract")
    allowed, reason = manager.precheck(spec, state)
    assert allowed is False
    assert reason == "max_steps"


def test_publish_is_rejected_without_evidence_and_approval(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="policy")
    call = SkillCall(
        call_id="publish-1", skill_name="publish_feature_package", actor=RoleId.FEATURE_PUBLISHER,
        inputs={"package_version": "v1", "subject_sha256": "1" * 64}, idempotency_key="publish-1",
    )
    result = workflow.executor.invoke(call)
    assert result.status is SkillStatus.REJECTED
    assert result.reason_code == "EVIDENCE_TIER_TOO_LOW"


def test_unknown_skill_has_normalized_rejection(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="unknown")
    call = SkillCall("unknown-1", "does_not_exist", RoleId.MANAGER, {}, "unknown-1")
    result = workflow.executor.invoke(call)
    assert result.status is SkillStatus.REJECTED
    assert result.reason_code == "UNKNOWN_SKILL"
    events = EventStore.read_verified(workflow.run_dir / "events.jsonl")
    assert events[-1].event_type == "skill.rejected"


def test_approval_must_bind_exact_subject(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        lambda payload: payload["permissions"].update({"approval_adapter_enabled": True}),
    )
    workflow = CausalFeatureOpsWorkflow(config_path=config, output_root=tmp_path / "runs", run_id="approval")
    # Injecting a verified tier here tests only approval policy; no publish handler runs.
    workflow.state.evidence.append(
        EvidenceEnvelope("ev", "model", EvidenceStatus.VERIFIED, EvidenceTier.VALIDATION,
                         True, {}, gates={}, metrics={})
    )
    workflow.state.phase = RunPhase.WAITING_FOR_APPROVAL
    token = ApprovalToken("a1", "publish_feature_package", workflow.state.contract_sha256,
                          "owner", "approved", "2" * 64)
    workflow.state.approvals.append(token)
    workflow.state.results["readiness"] = SkillResult(
        "ready", "assess_release_readiness", SkillStatus.SUCCEEDED, "READY_FOR_APPROVAL",
        "ready", {"ready_for_human_approval": True}, (), (), False,
    )
    call = SkillCall(
        call_id="publish-2", skill_name="publish_feature_package", actor=RoleId.FEATURE_PUBLISHER,
        inputs={"package_version": "v1", "subject_sha256": "1" * 64},
        idempotency_key="publish-2", approval=token,
    )
    result = workflow.executor.invoke(call)
    assert result.status is SkillStatus.REJECTED
    assert result.reason_code == "APPROVAL_SUBJECT_MISMATCH"


def test_mutating_state_cannot_enable_publish_when_adapter_is_disabled(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="disabled-approval")
    workflow.state.evidence.append(
        EvidenceEnvelope("ev", "model", EvidenceStatus.VERIFIED, EvidenceTier.VALIDATION,
                         True, {}, gates={}, metrics={})
    )
    workflow.state.phase = RunPhase.WAITING_FOR_APPROVAL
    workflow.state.results["readiness"] = SkillResult(
        "ready", "assess_release_readiness", SkillStatus.SUCCEEDED, "READY_FOR_APPROVAL",
        "ready", {"ready_for_human_approval": True}, (), (), False,
    )
    token = ApprovalToken("a1", "publish_feature_package", workflow.state.contract_sha256,
                          "owner", "approved", "1" * 64)
    workflow.state.approvals.append(token)
    result = workflow.executor.invoke(
        SkillCall("p", "publish_feature_package", RoleId.FEATURE_PUBLISHER,
                  {"package_version": "v1", "subject_sha256": "1" * 64}, "p", token)
    )
    assert result.status is SkillStatus.REJECTED
    assert result.reason_code == "APPROVAL_ADAPTER_DISABLED"


def test_evaluator_cannot_shrink_frozen_gate_list(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="gate-contract")
    result = workflow.executor.invoke(
        SkillCall("g", "evaluate_research_evidence", RoleId.CAUSAL_EVALUATOR,
                  {"evidence_id": "missing", "required_gates": []}, "g")
    )
    assert result.status is SkillStatus.REJECTED
    assert result.reason_code == "GATE_CONTRACT_MISMATCH"


def test_idempotency_key_cannot_be_reused_for_different_input(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="idempotency")
    first = SkillCall(
        "c1", "register_research_contract", RoleId.MANAGER,
        {"objective": "one", "research_boundary": {}, "task_graph": []}, "same-key",
    )
    second = SkillCall(
        "c2", "register_research_contract", RoleId.MANAGER,
        {"objective": "two", "research_boundary": {}, "task_graph": []}, "same-key",
    )
    assert workflow.executor.invoke(first).status is SkillStatus.SUCCEEDED
    conflict = workflow.executor.invoke(second)
    assert conflict.status is SkillStatus.REJECTED
    assert conflict.reason_code == "IDEMPOTENCY_KEY_CONFLICT"


def test_illegal_state_transition_fails_closed() -> None:
    state = RunState("r", "s", RunMode.DESIGN_ONLY, "o", "0" * 64)
    with pytest.raises(ValueError, match="illegal transition"):
        transition(state, RunPhase.FEATURES_PROPOSED)


def test_claim_review_cannot_bypass_human_approval_state() -> None:
    state = RunState("r", "s", RunMode.DESIGN_ONLY, "o", "0" * 64)
    state.phase = RunPhase.CLAIMS_REVIEWED
    with pytest.raises(ValueError, match="illegal transition"):
        transition(state, RunPhase.RELEASED)


def test_tampered_run_artifact_is_detected(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="tamper")
    run_dir = workflow.run()
    report = run_dir / "reports" / "final_report.md"
    report.write_text(report.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    verification = verify_run_directory(run_dir)
    assert verification["verified"] is False
    assert any(item.startswith("size:reports/final_report.md") for item in verification["failures"])


def test_inventory_cannot_hide_required_artifacts(tmp_path: Path) -> None:
    workflow = CausalFeatureOpsWorkflow(config_path=CONFIG, output_root=tmp_path, run_id="inventory")
    run_dir = workflow.run()
    path = run_dir / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"] = []
    payload["artifact_count"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_run_directory(run_dir)
    assert verification["verified"] is False
    assert "inventory_does_not_exactly_cover_run_files" in verification["failures"]
    assert any(item.startswith("required_missing:") for item in verification["failures"])


def test_harness_core_has_no_model_or_data_stack_imports() -> None:
    forbidden = {"duckdb", "pyarrow", "sklearn", "matplotlib", "gate2b_metrics", "gate2b_repair_v003"}
    paths = list((PROJECT_ROOT / "src" / "kuairand_longseq" / "harness").glob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported.intersection(forbidden), f"{path.name}: {imported.intersection(forbidden)}"
