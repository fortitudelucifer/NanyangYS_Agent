# KuaiRand-1K reproducibility kit

This directory rebuilds the governed research inputs without storing the
KuaiRand dataset, Silver tables, row-level predictions, bootstrap arrays, or
model binaries in Git.

## What is versioned

- the original research source, contracts, configs, tests, and experiment
  runners under `reference/kuairand-longseq-agent/`;
- exact Raw file sizes and SHA-256 values;
- expected Silver row counts, sizes, and SHA-256 values;
- the actual v010-v012 package versions, CUDA build, reference GPU, seeds,
  bootstrap protocol, temporal splits, and contract digests;
- compact reports and figures that explain the released evidence.

## What is deliberately not versioned

- the downloaded KuaiRand archives and CSV files;
- formal Silver and quarantine parquet files;
- feature matrices, row-level predictions, target-row manifests, bootstrap
  multiplicities/replicates, checkpoints, and model states;
- generated experiment output directories.

Those paths are ignored by Git. A reproduction must recreate them locally and
then compare them with the frozen expected manifests.

## 1. Create an environment

The published v010-v012 computations used Python 3.11.15. The exact package
versions are in `environment/requirements-release-v010-v012.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproducibility/environment/requirements-release-v010-v012.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only Silver construction and v011 calibration, PyTorch is not required.
Before a GPU run, save `python --version`, `pip freeze`, `nvidia-smi`, OS/kernel,
CPU count, RAM, GPU model, and driver version with the new run artifacts.

The owner-supplied supplementary learning-curve v013 preflight is recorded
separately in
`environment/supplementary-learning-curve-v013-runtime.json`. It includes the
exact Ubuntu/kernel, driver, CUDA, environment, optimizer, deterministic-mode,
training parameters, seeds, and contract digest. It must not be used to
retroactively fill missing fields in an older run manifest.

## 2. Download KuaiRand-1K

The default URL is the KuaiRand project's official 1K archive URL documented in
its public repository. You can instead download from an official mirror and
pass the local archive path.

```bash
python reproducibility/scripts/download_kuairand_1k.py
```

Manual archive:

```bash
python reproducibility/scripts/download_kuairand_1k.py \
  --archive /path/to/KuaiRand-1K.tar.gz
```

Archive transport is not trusted as identity. After safe extraction, the script
checks the six official data files against `manifests/raw-files.json`.

## 3. Verify Raw and build Silver

```bash
python reproducibility/scripts/verify_reproduction.py --raw
python reproducibility/scripts/run_reproduction.py --build-silver
python reproducibility/scripts/verify_reproduction.py --silver
```

The formal Silver builder intentionally excludes
`video_features_statistic_1k.csv`: it contains post-hoc aggregates and was not
joined into the released Silver layer.

The expected output hashes are strict. If hashes differ but row counts and
quality gates match, do not overwrite the reference manifest. Record the new
OS, driver, dependency lock, code revision, and resulting hashes as a separate
reproduction attempt.

## 4. Validate contracts before expensive runs

From `reference/kuairand-longseq-agent`:

```bash
python scripts/run_gate2b_baselines_v002.py --validate-only
python scripts/run_history_value_adam_validation_v007.py --validate-only
python scripts/run_history_value_adam_sealed_v008.py --validate-only
python scripts/run_history_value_adam_random_v010.py --validate-only
python experiments/bl2_target_domain_calibration_v011/run_calibration_v011.py --validate-only
python experiments/bl2_target_domain_retraining_v012/run_retraining_v012.py --validate-only
```

Release modes are intentionally not wrapped in a one-click command. Each runner
requires its own frozen contract and approval digest, and later stages consume
hash-verified predecessor artifacts. A new researcher must issue a new local
approval rather than treating the historical approval files as authorization.

## 5. Interpretation boundaries

- v010 random is a post-audit evaluation of the frozen model.
- v011 performs monotone probability calibration; it does not change ranking.
- v012 is an internal temporal replay, not external independent validation.
- the supplementary learning-curve experiment reuses Validation rows and is
  not an independent confirmation or a neural result.
- copying a producer manifest does not automatically satisfy the Agent's
  consumer Evidence contract; evidence admission remains fail-closed.

See `REPRODUCIBILITY_GAPS.md` before claiming a bitwise or independent
reproduction.
