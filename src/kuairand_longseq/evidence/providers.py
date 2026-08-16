"""Pluggable, fail-closed research evidence providers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Protocol

from kuairand_longseq.harness.contracts import (
    ArtifactRef,
    EvidenceEnvelope,
    EvidenceStatus,
    EvidenceTier,
)
from kuairand_longseq.harness.yaml_utils import load_yaml_unique
from kuairand_longseq.harness.storage import canonical_json_bytes


class EvidenceProvider(Protocol):
    provider_id: str
    version: str

    def fetch(self, request: dict[str, Any]) -> EvidenceEnvelope: ...


class NullEvidenceProvider:
    """Honest default used while the GPU/model evidence does not exist."""

    provider_id = "null_research_evidence"
    version = "1.0.0"
    manifest_read_count = 0
    artifact_hash_read_count = 0
    bytes_read = 0

    def fetch(self, request: dict[str, Any]) -> EvidenceEnvelope:
        requested = str(request.get("evidence_kind", "model_comparison"))
        return EvidenceEnvelope(
            evidence_id=f"unavailable:{requested}",
            evidence_kind=requested,
            status=EvidenceStatus.UNAVAILABLE,
            tier=EvidenceTier.NONE,
            claim_eligible=False,
            scope={
                "task": request.get("task", "candidate_long_view_prediction"),
                "requested_split": request.get("split", "validation"),
            },
            provenance={"provider_id": self.provider_id, "provider_version": self.version},
            metrics={},
            gates={},
            limitations=(
                "No verified experiment manifest was supplied.",
                "The system may demonstrate governance but cannot claim model improvement.",
            ),
            reason_code="GPU_RESEARCH_EVIDENCE_NOT_AVAILABLE",
            provider_id=self.provider_id,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ManifestEvidenceProvider:
    """Verify a completed evidence handoff before exposing its facts to agents.

    The provider accepts only relative artifact paths rooted beside the
    manifest.  It never discovers data recursively and never recomputes a
    scientific metric.
    """

    provider_id = "research_evidence_manifest"
    version = "1.0.0"

    REQUIRED_FIELDS = {
        "schema_version",
        "evidence_id",
        "evidence_kind",
        "status",
        "tier",
        "claim_eligible",
        "execution_authorized",
        "scope",
        "metrics",
        "gates",
        "limitations",
        "artifacts",
        "contract_sha256",
        "code_sha256",
        "input_manifest_sha256",
        "model_config_sha256",
        "authorization_sha256",
        "models",
    }

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self._verified_snapshot: bytes | None = None
        self.manifest_read_count = 0
        self.artifact_hash_read_count = 0
        self.bytes_read = 0

    def snapshot_bytes(self) -> bytes:
        """Return the exact verified manifest for run-local archival."""

        if self._verified_snapshot is None:
            raise RuntimeError("manifest has not completed verified admission")
        return self._verified_snapshot

    def _invalid(self, reason_code: str, message: str, evidence_kind: str = "model_comparison") -> EvidenceEnvelope:
        return EvidenceEnvelope(
            evidence_id=f"invalid:{self.manifest_path.name}",
            evidence_kind=evidence_kind,
            status=EvidenceStatus.INVALID,
            tier=EvidenceTier.NONE,
            claim_eligible=False,
            scope={"manifest": self.manifest_path.as_posix()},
            provenance={"provider_id": self.provider_id, "provider_version": self.version},
            metrics={},
            gates={},
            limitations=(message,),
            reason_code=reason_code,
            provider_id=self.provider_id,
        )

    def fetch(self, request: dict[str, Any]) -> EvidenceEnvelope:
        if not self.manifest_path.is_file():
            return self._invalid("MANIFEST_NOT_FOUND", "The requested evidence manifest does not exist.")
        max_manifest_reads = int(request.get("max_manifest_reads", 1))
        if max_manifest_reads <= 0 or self.manifest_read_count >= max_manifest_reads:
            return self._invalid(
                "EVIDENCE_MANIFEST_READ_BUDGET_EXCEEDED",
                "The frozen evidence-manifest read budget has been exhausted.",
            )
        try:
            manifest_bytes = self.manifest_path.read_bytes()
            self.manifest_read_count += 1
            self.bytes_read += len(manifest_bytes)
            payload = load_yaml_unique(manifest_bytes.decode("utf-8"))
        except Exception as exc:
            return self._invalid("MANIFEST_PARSE_FAILED", f"{type(exc).__name__}: {exc}")
        if not isinstance(payload, dict):
            return self._invalid("MANIFEST_TYPE_INVALID", "Evidence manifest must be a mapping.")
        missing = sorted(self.REQUIRED_FIELDS.difference(payload))
        evidence_kind = str(payload.get("evidence_kind", request.get("evidence_kind", "model_comparison")))
        if missing:
            return self._invalid("MANIFEST_FIELDS_MISSING", f"Missing fields: {', '.join(missing)}", evidence_kind)
        if payload["schema_version"] != "1.0":
            return self._invalid(
                "MANIFEST_SCHEMA_VERSION_UNSUPPORTED",
                f"Unsupported evidence-manifest schema version: {payload['schema_version']}",
                evidence_kind,
            )
        if payload["status"] != "complete" or payload["execution_authorized"] is not True:
            return self._invalid(
                "EVIDENCE_RUN_NOT_COMPLETE_OR_AUTHORIZED",
                "Only complete, explicitly authorized runs can become evidence.",
                evidence_kind,
            )
        if payload.get("placeholder") or payload.get("synthetic"):
            return self._invalid("PLACEHOLDER_EVIDENCE_FORBIDDEN", "Placeholder or synthetic results are not scientific evidence.", evidence_kind)
        try:
            tier = EvidenceTier(str(payload["tier"]))
        except ValueError:
            return self._invalid("EVIDENCE_TIER_INVALID", f"Unknown tier: {payload['tier']}", evidence_kind)
        if payload["claim_eligible"] is not True:
            return self._invalid("CLAIM_ELIGIBILITY_FALSE", "The producer did not authorize claims from this evidence.", evidence_kind)
        digest_fields = (
            "contract_sha256", "code_sha256", "input_manifest_sha256",
            "model_config_sha256", "authorization_sha256",
        )
        for field in digest_fields:
            if not re.fullmatch(r"[0-9a-f]{64}", str(payload[field])):
                return self._invalid("PROVENANCE_HASH_INVALID", f"{field} must be a lowercase SHA-256 digest.", evidence_kind)
        expected_provenance = request.get("expected_provenance")
        if not isinstance(expected_provenance, dict):
            return self._invalid(
                "EVIDENCE_EXPECTATIONS_NOT_FROZEN",
                "The consuming run has no frozen provenance-hash expectations.",
                evidence_kind,
            )
        expected_fields = (*digest_fields, "target_manifest_sha256")
        for field in expected_fields:
            expected = expected_provenance.get(field)
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                return self._invalid(
                    "EVIDENCE_EXPECTATIONS_NOT_FROZEN",
                    f"The expected {field} is not frozen in the consuming contract.",
                    evidence_kind,
                )
        for field in digest_fields:
            if str(payload[field]) != expected_provenance[field]:
                return self._invalid(
                    "PROVENANCE_HASH_MISMATCH",
                    f"Evidence {field} differs from the consuming contract.",
                    evidence_kind,
                )
        if evidence_kind != str(request.get("evidence_kind", evidence_kind)):
            return self._invalid("EVIDENCE_KIND_SCOPE_MISMATCH", "Evidence kind does not match the request.", evidence_kind)
        if not isinstance(payload["models"], list) or not payload["models"]:
            return self._invalid("MODEL_SET_INVALID", "A scientific comparison must list its model IDs.", evidence_kind)
        requested_models = {str(item) for item in request.get("required_models", [])}
        observed_models = {str(item) for item in payload["models"]}
        if not requested_models.issubset(observed_models):
            return self._invalid("MODEL_SET_SCOPE_MISMATCH", "Evidence does not cover every requested model.", evidence_kind)
        for scope_key in ("task", "split"):
            requested_value = request.get(scope_key)
            if requested_value is not None and payload["scope"].get(scope_key) != requested_value:
                return self._invalid(
                    "EVIDENCE_SCOPE_MISMATCH",
                    f"Evidence scope {scope_key} does not match the request.",
                    evidence_kind,
                )
        expected_dataset = request.get("expected_dataset")
        if not expected_dataset or payload["scope"].get("dataset") != expected_dataset:
            return self._invalid("EVIDENCE_DATASET_MISMATCH", "Evidence dataset is not the frozen dataset.", evidence_kind)
        minimum_by_split = {
            "train_only": EvidenceTier.TRAIN_ONLY,
            "validation": EvidenceTier.VALIDATION,
            "sealed_test": EvidenceTier.SEALED_TEST,
        }
        requested_minimum = minimum_by_split.get(str(request.get("split", "")), EvidenceTier.NONE)
        if requested_minimum is not EvidenceTier.NONE and tier is not requested_minimum:
            return self._invalid("EVIDENCE_TIER_SCOPE_MISMATCH", "Evidence tier must exactly match the requested split.", evidence_kind)
        target_digest = str(payload["scope"].get("target_manifest_sha256", ""))
        if evidence_kind == "paired_model_comparison" and not re.fullmatch(r"[0-9a-f]{64}", target_digest):
            return self._invalid("TARGET_MANIFEST_HASH_INVALID", "Paired comparisons require a target manifest SHA-256.", evidence_kind)
        if evidence_kind == "paired_model_comparison" and target_digest != expected_provenance["target_manifest_sha256"]:
            return self._invalid("TARGET_MANIFEST_HASH_MISMATCH", "Target rows differ from the frozen expectation.", evidence_kind)
        if evidence_kind == "paired_model_comparison" and not payload["metrics"]:
            return self._invalid("METRIC_SET_EMPTY", "Paired model evidence must contain registered metric values.", evidence_kind)
        if not isinstance(payload["artifacts"], list) or not payload["artifacts"]:
            return self._invalid("ARTIFACT_SET_EMPTY", "Verified evidence must reference at least one artifact.", evidence_kind)
        max_artifacts = int(request.get("max_artifacts", 64))
        if max_artifacts <= 0 or len(payload["artifacts"]) > max_artifacts:
            return self._invalid(
                "EVIDENCE_ARTIFACT_BUDGET_EXCEEDED",
                "Artifact count exceeds the frozen request budget.",
                evidence_kind,
            )
        try:
            declared_bytes = sum(int(item["size_bytes"]) for item in payload["artifacts"])
        except (KeyError, TypeError, ValueError):
            return self._invalid("ARTIFACT_SCHEMA_INVALID", "Artifact sizes must be integers.", evidence_kind)
        max_bytes = int(request.get("max_total_artifact_bytes", 20 * 1024**3))
        if max_bytes <= 0 or declared_bytes > max_bytes:
            return self._invalid(
                "EVIDENCE_ARTIFACT_BUDGET_EXCEEDED",
                "Artifact bytes exceed the frozen request budget.",
                evidence_kind,
            )
        try:
            canonical_json_bytes(payload["scope"])
            canonical_json_bytes(payload["metrics"])
            canonical_json_bytes(payload["gates"])
            canonical_json_bytes(payload["limitations"])
        except (TypeError, ValueError) as exc:
            return self._invalid("MANIFEST_JSON_VALUE_INVALID", str(exc), evidence_kind)

        root = self.manifest_path.parent
        artifacts: list[ArtifactRef] = []
        for index, raw in enumerate(payload["artifacts"]):
            if not isinstance(raw, dict) or not {"path", "size_bytes", "sha256"}.issubset(raw):
                return self._invalid("ARTIFACT_SCHEMA_INVALID", f"Artifact entry {index} is incomplete.", evidence_kind)
            candidate = (root / str(raw["path"])).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                return self._invalid("ARTIFACT_PATH_ESCAPE", f"Artifact {index} escapes the manifest directory.", evidence_kind)
            if not candidate.is_file():
                return self._invalid("ARTIFACT_NOT_FOUND", f"Artifact not found: {raw['path']}", evidence_kind)
            if candidate.stat().st_size != int(raw["size_bytes"]):
                return self._invalid("ARTIFACT_SIZE_MISMATCH", f"Artifact size mismatch: {raw['path']}", evidence_kind)
            self.artifact_hash_read_count += 1
            self.bytes_read += candidate.stat().st_size
            if _sha256_file(candidate) != str(raw["sha256"]):
                return self._invalid("ARTIFACT_HASH_MISMATCH", f"Artifact hash mismatch: {raw['path']}", evidence_kind)
            artifacts.append(
                ArtifactRef(
                    path=str(raw["path"]).replace("\\", "/"),
                    size_bytes=int(raw["size_bytes"]),
                    sha256=str(raw["sha256"]),
                    media_type=str(raw.get("media_type", "application/octet-stream")),
                    producer=str(raw.get("producer", payload["evidence_id"])),
                    role=str(raw.get("role", "scientific_evidence")),
                )
            )

        scope = dict(payload["scope"])
        scope["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        self._verified_snapshot = manifest_bytes
        return EvidenceEnvelope(
            evidence_id=str(payload["evidence_id"]),
            evidence_kind=evidence_kind,
            status=EvidenceStatus.VERIFIED,
            tier=tier,
            claim_eligible=True,
            scope=scope,
            provenance={
                "provider_id": self.provider_id,
                "provider_version": self.version,
                "manifest_path": self.manifest_path.as_posix(),
                **{field: str(payload[field]) for field in digest_fields},
                "models": sorted(observed_models),
            },
            artifacts=tuple(artifacts),
            metrics=dict(payload["metrics"]),
            gates=dict(payload["gates"]),
            limitations=tuple(str(item) for item in payload["limitations"]),
            reason_code=None,
            provider_id=self.provider_id,
        )

