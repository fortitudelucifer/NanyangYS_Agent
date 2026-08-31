# Public-package sanitization record

The public kit is derived from the local evidence package but is not a byte-for-
byte mirror. The following changes are intentional and non-scientific:

- Raw, Silver, quarantine, row-level prediction, feature, bootstrap-array, and
  model-state files are omitted.
- Historical workspace handoff/index files containing machine-specific paths
  are omitted.
- machine-specific paths in public commands are replaced with repository-
  relative `python` commands.
- one `pyproject.toml` comment was changed to point to the public release lock.
- public Raw/Silver expected manifests use repository-relative paths; their
  sizes, row counts, timestamps, and SHA-256 values are copied from the formal
  evidence records and were rechecked against the local files.
- the owner-supplied supplementary v013 preflight is published as a sanitized
  relative metadata record; its machine-specific source path is omitted.

No model metric, confidence interval, seed, split, row count, artifact digest,
or scientific contract value was modified by this sanitization.
