"""Research-evidence handoff interfaces."""

from .providers import EvidenceProvider, ManifestEvidenceProvider, NullEvidenceProvider
from .admission import EvidenceAdmissionPolicy

__all__ = ["EvidenceAdmissionPolicy", "EvidenceProvider", "ManifestEvidenceProvider", "NullEvidenceProvider"]

