# Reproducibility-kit QA

Validation date: 2026-08-31

| Check | Result |
|---|---:|
| Raw files verified against public manifest | 6 / 6 |
| Silver/quarantine outputs verified against public manifest | 10 / 10 |
| Python files parsed with Python 3.11 AST | 106 / 106 |
| JSON files parsed | 48 / 48 |
| YAML files parsed | 30 / 30 |
| Files above 10 MiB | 0 |
| Largest file | 1,300,512 bytes |
| Git LFS required | no |
| Raw/Silver/prediction/model data included | no |
| Existing Agent regression tests | 31 / 31 passed |
| Restricted absolute workspace paths | 0 |

The public manifests were checked against the existing formal files. This QA
did not rerun Silver cleaning, training, calibration, bootstrap, or any released
experiment.

The supplementary v013 runtime record is transcribed from the owner-supplied
preflight metadata and kept separate from the v010-v012 release-runtime record.
