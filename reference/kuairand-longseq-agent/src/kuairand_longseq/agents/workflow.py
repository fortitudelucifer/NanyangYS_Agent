"""End-to-end CausalFeatureOps workflow that can run before GPU research."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kuairand_longseq.agents.roles import build_agent_identities
from kuairand_longseq.agents.team_runtime import AgentTask, LocalStructuredTeamRuntime
from kuairand_longseq.evidence import (
    EvidenceAdmissionPolicy,
    EvidenceProvider,
    ManifestEvidenceProvider,
    NullEvidenceProvider,
)
from kuairand_longseq.harness.contracts import (
    BudgetLimit, EvidenceStatus, RoleId, RunMode, RunPhase, RunState,
    SkillCall, SkillStatus, to_jsonable,
)
from kuairand_longseq.harness.runtime import (
    BudgetManager, ExecutionContext, PolicyEngine, SkillExecutor,
)
from kuairand_longseq.harness.state_machine import transition
from kuairand_longseq.harness.storage import (
    ArtifactStore, EventStore, canonical_json_bytes, checkpoint_state, sha256_file,
)
from kuairand_longseq.harness.yaml_utils import load_yaml_unique
from kuairand_longseq.skills import build_builtin_registry


class WorkflowError(RuntimeError):
    pass


IMPLEMENTATION_RELATIVE_PATHS = (
    "scripts/run_agent_system_demo_v001.py",
    "src/kuairand_longseq/agents/roles.py",
    "src/kuairand_longseq/agents/team_runtime.py",
    "src/kuairand_longseq/agents/workflow.py",
    "src/kuairand_longseq/evidence/admission.py",
    "src/kuairand_longseq/evidence/providers.py",
    "src/kuairand_longseq/harness/contracts.py",
    "src/kuairand_longseq/harness/runtime.py",
    "src/kuairand_longseq/harness/state_machine.py",
    "src/kuairand_longseq/harness/storage.py",
    "src/kuairand_longseq/harness/yaml_utils.py",
    "src/kuairand_longseq/skills/builtin.py",
)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = load_yaml_unique(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise WorkflowError("agent-system config must be a mapping")
    return payload, hashlib.sha256(raw).hexdigest()


def _default_run_id(config_sha: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"agent-v001-{stamp}-{config_sha[:8]}"


class CausalFeatureOpsWorkflow:
    def __init__(
        self,
        *,
        config_path: Path,
        output_root: Path,
        evidence_provider: EvidenceProvider | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.config, config_sha = _load_config(self.config_path)
        self.run_id = run_id or _default_run_id(config_sha)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.run_id):
            raise WorkflowError(
                "run_id must be 1-128 characters and contain only letters, digits, '.', '_' or '-'"
            )
        output_root = output_root.resolve()
        self.run_dir = (output_root / self.run_id).resolve()
        try:
            self.run_dir.relative_to(output_root)
        except ValueError as exc:
            raise WorkflowError("run_id escapes output_root") from exc
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise WorkflowError(f"run directory already exists and is not empty: {self.run_dir}")
        self.store = ArtifactStore(self.run_dir)
        self.state = RunState(
            run_id=self.run_id,
            system_id=str(self.config["system_id"]),
            mode=RunMode(str(self.config["mode"])),
            objective=str(self.config["objective"]),
            contract_sha256=config_sha,
        )
        self.events = EventStore(self.store, self.run_id)
        self.identities = build_agent_identities()
        self.registry = build_builtin_registry()
        budget_cfg = self.config["budgets"]
        limits = BudgetLimit(**{key: int(value) for key, value in budget_cfg.items()})
        provider = evidence_provider or NullEvidenceProvider()
        self.context = ExecutionContext(
            state=self.state,
            store=self.store,
            events=self.events,
            config=self.config,
            evidence_provider=provider,
            evidence_admission_policy=EvidenceAdmissionPolicy(
                self.config["evidence_request"],
                trusted_provider_types={
                    NullEvidenceProvider.provider_id: NullEvidenceProvider,
                    ManifestEvidenceProvider.provider_id: ManifestEvidenceProvider,
                },
            ),
        )
        self.executor = SkillExecutor(
            self.registry,
            PolicyEngine(
                self.identities,
                sealed_access_enabled=bool(self.config["permissions"]["sealed_access_enabled"]),
                approval_adapter_enabled=bool(self.config["permissions"]["approval_adapter_enabled"]),
            ),
            BudgetManager(limits),
            self.context,
        )
        self.team = LocalStructuredTeamRuntime(self.executor)
        self._task_results: dict[str, Any] = {}

    def _dispatch(
        self,
        task_id: str,
        role: RoleId,
        skill_name: str,
        inputs: dict[str, Any],
        *,
        depends_on: tuple[str, ...] = (),
        context_keys: tuple[str, ...] = (),
    ):
        call = SkillCall(
            call_id=f"call-{len(self._task_results) + 1:02d}",
            skill_name=skill_name,
            actor=role,
            inputs=inputs,
            idempotency_key=f"{self.run_id}:{task_id}",
        )
        result = self.team.dispatch(AgentTask(task_id, role, call, depends_on, context_keys))
        self._task_results[task_id] = result
        return result

    def _move(self, target: RunPhase, actor: str, reason: str | None = None) -> None:
        source = self.state.phase
        transition(self.state, target, reason=reason)
        self.events.append(
            "state.transition",
            actor,
            {"from": source.value, "to": target.value, "reason": reason},
        )
        self.state.sequence = self.events.sequence
        checkpoint_state(self.store, self.state)

    def _block_if_failed(self, result, actor: str) -> bool:
        if result.status in {SkillStatus.FAILED, SkillStatus.REJECTED}:
            target = RunPhase.FAILED if result.status is SkillStatus.FAILED else RunPhase.BLOCKED
            self._move(target, actor, f"{result.reason_code}: {result.message}")
            return True
        return False

    def run(self) -> Path:
        self.events.append(
            "run.accepted", "user",
            {"objective": self.state.objective, "mode": self.state.mode.value,
             "config_path": self.config_path.as_posix()},
        )
        self.state.sequence = self.events.sequence
        checkpoint_state(self.store, self.state)
        task_graph = self.config["task_graph"]

        result = self._dispatch(
            "T01", RoleId.MANAGER, "register_research_contract",
            {"objective": self.state.objective,
             "research_boundary": self.config["research_boundary"], "task_graph": task_graph},
            context_keys=("objective", "research_boundary"),
        )
        if self._block_if_failed(result, RoleId.MANAGER.value):
            return self._finalize()
        self._move(RunPhase.CONTRACT_REGISTERED, RoleId.MANAGER.value)

        result = self._dispatch(
            "T02", RoleId.DATA_AUDITOR, "audit_project_boundary",
            {"data_policy": self.config["data_policy"], "research_status": self.config["research_status"]},
            depends_on=("T01",), context_keys=("data_policy", "research_status"),
        )
        if self._block_if_failed(result, RoleId.DATA_AUDITOR.value):
            return self._finalize()
        self._move(RunPhase.INPUTS_AUDITED, RoleId.DATA_AUDITOR.value)

        result = self._dispatch(
            "T03", RoleId.FEATURE_MINER, "propose_feature_specs",
            {"feature_policy": self.config["feature_policy"], "research_status": self.config["research_status"]},
            depends_on=("T02",), context_keys=("feature_policy",),
        )
        if self._block_if_failed(result, RoleId.FEATURE_MINER.value):
            return self._finalize()
        self._move(RunPhase.FEATURES_PROPOSED, RoleId.FEATURE_MINER.value)

        result = self._dispatch(
            "T04", RoleId.DATA_AUDITOR, "detect_temporal_leakage",
            {"feature_fields": self.config["feature_policy"]["demo_feature_fields"],
             "denied_fields": self.config["feature_policy"]["hard_deny_fields"]},
            depends_on=("T03",), context_keys=("feature_fields", "hard_deny_fields"),
        )
        if self._block_if_failed(result, RoleId.DATA_AUDITOR.value):
            return self._finalize()

        evidence_result = self._dispatch(
            "T05", RoleId.CAUSAL_EVALUATOR, "request_research_evidence",
            {"request": self.config["evidence_request"]}, depends_on=("T04",),
            context_keys=("task", "split", "evidence_kind"),
        )
        if self._block_if_failed(evidence_result, RoleId.CAUSAL_EVALUATOR.value):
            return self._finalize()
        evidence_id = evidence_result.evidence[0].evidence_id
        gate_result = self._dispatch(
            "T06", RoleId.CAUSAL_EVALUATOR, "evaluate_research_evidence",
            {"evidence_id": evidence_id, "required_gates": self.config["required_release_gates"]},
            depends_on=("T05",), context_keys=("evidence_id", "required_release_gates"),
        )
        if self._block_if_failed(gate_result, RoleId.CAUSAL_EVALUATOR.value):
            return self._finalize()
        self._move(RunPhase.EVIDENCE_EVALUATED, RoleId.CAUSAL_EVALUATOR.value)

        review = self._dispatch(
            "T07", RoleId.SAFETY_REVIEWER, "review_claim_boundaries",
            {"review_scope": self.config["claim_policy"]}, depends_on=("T06",),
            context_keys=("evidence_tier", "gate_verdict", "claim_policy"),
        )
        if self._block_if_failed(review, RoleId.SAFETY_REVIEWER.value):
            return self._finalize()
        self._move(RunPhase.CLAIMS_REVIEWED, RoleId.SAFETY_REVIEWER.value)

        readiness = self._dispatch(
            "T08", RoleId.FEATURE_PUBLISHER, "assess_release_readiness",
            {"minimum_tier": self.config["release_policy"]["minimum_evidence_tier"]},
            depends_on=("T07",), context_keys=("claims", "gates", "evidence_tier"),
        )
        if self._block_if_failed(readiness, RoleId.FEATURE_PUBLISHER.value):
            return self._finalize()
        if readiness.facts.get("ready_for_human_approval"):
            self._move(RunPhase.WAITING_FOR_APPROVAL, RoleId.FEATURE_PUBLISHER.value,
                       "Verified evidence passed frozen gates; publication still requires human approval.")
        else:
            self._move(RunPhase.WAITING_FOR_EVIDENCE, RoleId.FEATURE_PUBLISHER.value,
                       "Required verified GPU/model evidence is not available.")
        return self._finalize()

    def _report_markdown(self) -> str:
        evidence = self.state.evidence[-1] if self.state.evidence else None
        lines = [
            "# CausalFeatureOps Agent Run",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Mode: `{self.state.mode.value}`",
            f"- Terminal state: `{self.state.phase.value}`",
            f"- Scientific result available: `{'yes' if evidence and evidence.claim_eligible else 'no'}`",
            f"- LLM calls: `{self.state.budget.llm_calls}`",
            f"- GPU seconds: `{self.state.budget.gpu_seconds}`",
            "",
            "## What this run proves",
            "",
        ]
        lines += [f"- {claim}" for claim in self.state.allowed_claims] or ["- No claim review was completed."]
        lines += ["", "## Missing evidence", ""]
        lines += [f"- {item}" for item in self.state.missing_evidence] or ["- None registered."]
        lines += ["", "## Claims that remain forbidden", ""]
        lines += [f"- {item}" for item in self.state.unsupported_claims] or ["- No claim review was completed."]
        lines += ["", "## Boundary", "",
                  "This run did not clean data, train a model, access sealed splits, or fabricate experiment metrics.", ""]
        return "\n".join(lines)

    def _finalize(self) -> Path:
        self.events.append(
            "run.terminal", "orchestrator",
            {"phase": self.state.phase.value, "reason": self.state.terminal_reason,
             "skill_calls": self.state.budget.skill_calls},
        )
        self.state.sequence = self.events.sequence
        checkpoint_state(self.store, self.state)
        self.store.write_text("reports/final_report.md", self._report_markdown(),
                              producer="orchestrator", role="human_report")
        provider = self.context.evidence_provider
        project_root = Path(__file__).resolve().parents[3]
        implementation_sha256 = {
            relative_path: sha256_file(project_root / relative_path)
            for relative_path in IMPLEMENTATION_RELATIVE_PATHS
        }
        manifest = {
            "schema_version": "1.0", "run_id": self.run_id,
            "system_id": self.state.system_id, "terminal_state": self.state.phase.value,
            "terminal_reason": self.state.terminal_reason, "mode": self.state.mode.value,
            "config_path": self.config_path.as_posix(), "contract_sha256": self.state.contract_sha256,
            "implementation_root": project_root.as_posix(),
            "implementation_sha256": implementation_sha256,
            "team_backend": self.team.backend_id,
            "agent_identities": [to_jsonable(x) for x in self.identities.values()],
            "skill_specs": [to_jsonable(x) for x in self.registry.specs],
            "budget_usage": to_jsonable(self.state.budget),
            "research_result_available": any(x.claim_eligible for x in self.state.evidence),
            "checkpoint_eligible": False,
            "data_access": {
                "raw": 0,
                "silver_parquet": 0,
                "sealed": 0,
                "external_evidence_manifest_reads": int(getattr(provider, "manifest_read_count", 0)),
                "external_evidence_artifact_hash_reads": int(getattr(provider, "artifact_hash_read_count", 0)),
                "external_evidence_bytes_read_for_verification": int(getattr(provider, "bytes_read", 0)),
            },
            "artifact_inventory_file": "artifact_manifest.json",
        }
        self.store.write_json("run_manifest.json", manifest, producer="orchestrator", role="run_manifest")
        inventory = {
            "schema_version": "1.0", "run_id": self.run_id, "hash_algorithm": "sha256",
            "self_excluded": True,
            "artifacts": [to_jsonable(x) for x in self.store.produced if x.path != "artifact_manifest.json"],
        }
        inventory["artifact_count"] = len(inventory["artifacts"])
        self.store.write_json("artifact_manifest.json", inventory,
                              producer="artifact_store", role="artifact_inventory")
        return self.run_dir


def verify_run_directory(run_dir: Path) -> dict[str, Any]:
    """Independent structural and hash verification for the offline demo."""

    import json

    root = run_dir.resolve()
    manifest_path = root / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    required_top = {"schema_version", "run_id", "hash_algorithm", "self_excluded", "artifact_count", "artifacts"}
    if required_top.difference(payload):
        failures.append("inventory_schema_missing_fields")
    if payload.get("hash_algorithm") != "sha256" or payload.get("self_excluded") is not True:
        failures.append("inventory_schema_invalid")
    entries = payload.get("artifacts", [])
    if payload.get("artifact_count") != len(entries):
        failures.append("inventory_count_mismatch")
    entry_paths = [str(entry.get("path")) for entry in entries]
    if len(entry_paths) != len(set(entry_paths)):
        failures.append("inventory_duplicate_path")
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    if sorted(entry_paths) != actual_paths:
        failures.append("inventory_does_not_exactly_cover_run_files")
    required_full_run = {
        "contracts/run_contract.json", "task_graph.json", "audits/project_boundary.json",
        "audits/temporal_leakage.json", "feature_specs/candidate_feature_specs.json",
        "evidence/research_evidence_status.json", "evaluation/gate_verdict.json",
        "safety/claim_review.json", "release/readiness.json", "events.jsonl", "state.json",
        "reports/final_report.md", "run_manifest.json",
    }
    run_manifest_path = root / "run_manifest.json"
    if run_manifest_path.is_file():
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("run_id") != payload.get("run_id"):
            failures.append("run_id_mismatch")
        if run_manifest.get("terminal_state") in {"waiting_for_evidence", "waiting_for_approval"}:
            missing_required = sorted(required_full_run.difference(entry_paths))
            failures.extend(f"required_missing:{path}" for path in missing_required)
        project_root = Path(str(run_manifest.get("implementation_root", "")))
        implementation = run_manifest.get("implementation_sha256")
        if not isinstance(implementation, dict) or set(implementation) != set(IMPLEMENTATION_RELATIVE_PATHS):
            failures.append("implementation_inventory_invalid")
        else:
            for relative_path, expected_sha256 in implementation.items():
                source_path = project_root / relative_path
                if not source_path.is_file():
                    failures.append(f"implementation_missing:{relative_path}")
                elif sha256_file(source_path) != expected_sha256:
                    failures.append(f"implementation_sha256:{relative_path}")
    for entry in entries:
        candidate = (root / entry["path"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            failures.append(f"path_escape:{entry['path']}")
            continue
        if not candidate.is_file():
            failures.append(f"missing:{entry['path']}")
        elif candidate.stat().st_size != entry["size_bytes"]:
            failures.append(f"size:{entry['path']}")
        elif sha256_file(candidate) != entry["sha256"]:
            failures.append(f"sha256:{entry['path']}")
    return {"verified": not failures, "failure_count": len(failures), "failures": failures,
            "artifact_count": len(entries), "artifact_manifest_sha256": sha256_file(manifest_path)}
