"""Harness-owned admission policy applied to evidence from every provider."""

from __future__ import annotations

import re
from typing import Any

from kuairand_longseq.harness.contracts import EvidenceEnvelope, EvidenceStatus, EvidenceTier


class EvidenceAdmissionPolicy:
    def __init__(
        self,
        request_contract: dict[str, Any],
        *,
        trusted_provider_types: dict[str, type[Any]] | None = None,
    ) -> None:
        self.request = request_contract
        self.trusted_provider_types = trusted_provider_types or {}

    def admit_provider(self, provider: Any) -> tuple[bool, str, str]:
        provider_id = getattr(provider, "provider_id", None)
        expected_type = self.trusted_provider_types.get(str(provider_id))
        if expected_type is None or type(provider) is not expected_type:
            return (
                False,
                "EVIDENCE_PROVIDER_IMPLEMENTATION_NOT_TRUSTED",
                "the evidence provider implementation is not registered by the Harness",
            )
        return True, "EVIDENCE_PROVIDER_TRUSTED", "provider implementation is registered"

    def admit(self, envelope: EvidenceEnvelope) -> tuple[bool, str, str]:
        if envelope.status is not EvidenceStatus.VERIFIED:
            if envelope.claim_eligible:
                return False, "NONVERIFIED_EVIDENCE_CLAIM_ELIGIBLE", "non-verified evidence cannot support claims"
            return True, "NONSCIENTIFIC_EVIDENCE_RECORDED", "non-scientific status may be recorded"

        allowed_providers = {str(item) for item in self.request.get("allowed_provider_ids", [])}
        if envelope.provider_id not in allowed_providers:
            return False, "EVIDENCE_PROVIDER_NOT_ALLOWED", "provider is not frozen in the consuming contract"
        expected = self.request.get("expected_provenance")
        if not isinstance(expected, dict):
            return False, "EVIDENCE_EXPECTATIONS_NOT_FROZEN", "expected provenance is absent"
        digest_fields = (
            "contract_sha256", "code_sha256", "input_manifest_sha256",
            "model_config_sha256", "authorization_sha256",
        )
        for field in (*digest_fields, "target_manifest_sha256"):
            value = expected.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                return False, "EVIDENCE_EXPECTATIONS_NOT_FROZEN", f"expected {field} is not frozen"
        for field in digest_fields:
            if envelope.provenance.get(field) != expected[field]:
                return False, "EVIDENCE_PROVENANCE_MISMATCH", f"{field} differs from the consuming contract"
        expected_scope = {
            "dataset": self.request.get("expected_dataset"),
            "task": self.request.get("task"),
            "split": self.request.get("split"),
            "target_manifest_sha256": expected["target_manifest_sha256"],
        }
        for key, value in expected_scope.items():
            if envelope.scope.get(key) != value:
                return False, "EVIDENCE_SCOPE_MISMATCH", f"scope {key} differs from the consuming contract"
        exact_tier = {
            "train_only": EvidenceTier.TRAIN_ONLY,
            "validation": EvidenceTier.VALIDATION,
            "sealed_test": EvidenceTier.SEALED_TEST,
        }.get(str(self.request.get("split")))
        if exact_tier is None or envelope.tier is not exact_tier:
            return False, "EVIDENCE_TIER_SCOPE_MISMATCH", "evidence tier must exactly match the requested split"
        required_models = {str(item) for item in self.request.get("required_models", [])}
        observed_models = {str(item) for item in envelope.provenance.get("models", [])}
        if not required_models or not required_models.issubset(observed_models):
            return False, "MODEL_SET_SCOPE_MISMATCH", "evidence does not cover every frozen model"
        if not envelope.artifacts:
            return False, "ARTIFACT_SET_EMPTY", "verified evidence must reference artifacts"
        max_artifacts = int(self.request.get("max_artifacts", 0))
        if max_artifacts <= 0 or len(envelope.artifacts) > max_artifacts:
            return False, "EVIDENCE_ARTIFACT_BUDGET_EXCEEDED", "artifact count exceeds the frozen request budget"
        max_bytes = int(self.request.get("max_total_artifact_bytes", 0))
        total_bytes = sum(artifact.size_bytes for artifact in envelope.artifacts)
        if max_bytes <= 0 or total_bytes > max_bytes:
            return False, "EVIDENCE_ARTIFACT_BUDGET_EXCEEDED", "artifact bytes exceed the frozen request budget"
        for artifact in envelope.artifacts:
            if artifact.size_bytes <= 0 or not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256):
                return False, "EVIDENCE_ARTIFACT_REFERENCE_INVALID", "artifact reference is incomplete"
        return True, "EVIDENCE_ADMITTED", "verified evidence matches the consuming contract"

