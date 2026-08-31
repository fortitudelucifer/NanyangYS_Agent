# Compact evidence summaries

These directories retain the small, directly inspectable outputs used to
describe v007-v012: pooled metrics, paired user-cluster bootstrap confidence
intervals, daily summaries, calibration/reliability audits, run manifests,
environment records, stage decisions, and stage-access ledgers.

Deliberately omitted:

- row-level predictions and feature matrices;
- target-row manifests;
- bootstrap multiplicity arrays and per-replicate parquet files;
- serialized models, candidate states, and checkpoints.

The retained `artifact_hash_manifest.json` files record the identities of both
present and omitted formal artifacts. Missing large artifacts must be rebuilt;
they must never be replaced with fabricated values.

The scopes are not interchangeable:

- `validation_v007`: reused feature evidence on the frozen Validation rows;
- `sealed_v008`: later standard-exposure sealed window;
- `random_v010`: frozen-model post-audit random exposure evaluation;
- `calibration_v011`: monotone probability calibration on temporal random
  splits; ranking metrics remain invariant;
- `temporal_replay_v012`: internal final temporal replay, not external
  independent validation.
