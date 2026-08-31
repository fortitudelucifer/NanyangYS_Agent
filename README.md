# NanyangYS Agent

[English](README.md) | [中文](README_CN.md)

> [!IMPORTANT]
> **Start from `main`.** It is the authoritative, lightweight, zero-LFS branch.
> The repository intentionally does **not** distribute KuaiRand Raw data,
> governed Silver tables, row-level predictions, feature matrices, bootstrap
> arrays, checkpoints, or model states. Download KuaiRand-1K from its official
> source and use the versioned reproducibility workflow below.

## Branch policy

[`main`](https://github.com/fortitudelucifer/NanyangYS_Agent/tree/main) is the
only persistent remote branch and the sole supported starting point. Completed
feature branches are deleted after merge so the branch list stays compact;
their review history remains available through the repository's merged pull
requests and merge commits.

Local recovery branches, when needed, are never pushed or treated as release
lines. All accepted code, documentation, figures, and reproducibility material
are already present in `main`.

## Overview

NanyangYS Agent is a lightweight, auditable video-recommendation Agent loop for
research on real, openly available interaction data. Its purpose is not to add
large models for their own sake. It uses compact, explicit components to study
long-horizon user preferences and to make each research claim traceable to a
contract, evidence boundary, approval, and reproducible artifact.

The current research starts from the public **KuaiRand-1K** dataset. Future
work may add other openly available video-interaction datasets from platforms
such as Tencent, ByteDance, and YouTube to study transferability and robustness
across data sources.

## Design philosophy

The system applies **Harness** and **Loop Engineering** ideas to research. Each
iteration is constrained by role permissions, budgets, evidence gates, stop
conditions, and hash-based audit records. New user feedback can initiate a new
loop that re-audits the boundary, proposes point-in-time feature specifications,
requests and evaluates evidence, reviews claim scope, and publishes a rollback-
capable feature package only when evidence and approval are both present.

The Agent does not silently reclean Silver or train a model on the host. It
engineers the question “can this research result be trusted?” as an explicit,
fail-closed workflow.

## Research and reproducibility status

An early demonstration snapshot recorded Train-only BL0/BL1/BL2 checks on an
RTX 4090. Later v010-v012 research records used Python 3.11.15, PyTorch
2.11.0+cu128, and an RTX 5070 Ti. The repository stores the contracts,
reproduction metadata, compact evidence summaries, reports, and expected
digests, but not the large run artifacts.

The supplementary learning-curve v013 preflight additionally records Ubuntu
24.04.4 LTS, Linux 7.0.0-30-generic x86_64, NVIDIA driver 580.173.02,
driver-supported CUDA 13.0, PyTorch-compiled CUDA 12.8, deterministic GPU Adam,
and the frozen training/sampling seeds. That metadata remains scoped to the
supplementary experiment; it is not silently backfilled into older manifests.

| Component | Current status |
|---|---|
| Agent Harness | Runs and verifies offline; deterministically stops at `waiting_for_evidence` when eligible evidence is absent. |
| GPU demo snapshot | Historical Train-only BL0/BL1/BL2 numerical checks are available. |
| Research-producer evidence | v010-v012 contracts, code, seeds, environments, compact summaries, and expected hashes are versioned; large outputs are excluded. |
| Scientific evidence admission | Incomplete; producer provenance and the Agent consumer Evidence manifest remain separate contracts. |
| Agent-side Validation / sealed / random admission | Not admitted. Producer artifacts cannot directly replace the Agent Evidence contract. |
| Publication | Not authorized; the system remains fail-closed. |
| Public reproducibility kit | Available without Raw, Silver, row-level predictions, feature matrices, bootstrap arrays, or model files. |

In the historical GPU demo snapshot, preliminary BL2-minus-BL1 results were AP
`+0.041295`, event-weighted user-GAUC `+0.048758`, Log Loss `-0.022953`, and
Brier `-0.008907`; the paired user-cluster bootstrap 95% CI for AP was
`[0.037661, 0.045054]`. These values support only a **Train-only preliminary
statement**. They are not Validation, Gold, sealed-test, or release claims.

## Experiment walkthrough

This section presents the experiment as a compact, slide-like research story.
The README uses white-background PNG exports so every chart remains legible in
both light and dark GitHub themes. The original transparent exports are
preserved in
[`docs/assets/experiment-story/transparent-backup/`](docs/assets/experiment-story/transparent-backup/).
BL1 and BL2 are **sparse logistic-regression baselines**, not neural networks.
Confidence intervals below are paired 95% user-cluster bootstrap intervals
unless stated otherwise.

### 1. Define the estimand before fitting a model

The study first separates the population, exposure domain, temporal window,
and evaluable row set. This prevents a gain measured on one scope from being
silently generalized to another.

<img src="docs/assets/experiment-story/P03_nested_scope.png" width="100%" alt="Nested evaluation scope from population to eligible rows">

Feature families are then assigned a point-in-time availability rule. A field
is usable only when it would have been available at the target event; post-hoc
aggregates are not treated as historical features.

<img src="docs/assets/experiment-story/P04_field_families.png" width="100%" alt="Feature families and point-in-time availability">

**Takeaway:** the experiment contract fixes *who*, *when*, *which exposure
domain*, and *which information* before model comparison.

### 2. Audit the data before modeling

<table>
  <tr>
    <td width="50%"><img src="docs/assets/experiment-story/P05a_missingness.png" width="100%" alt="Missingness audit"></td>
    <td width="50%"><img src="docs/assets/experiment-story/P05b_duration_skew.png" width="100%" alt="Video duration distribution and skew"></td>
  </tr>
  <tr>
    <td valign="top"><b>Missingness is semantic.</b> Sentinel values and absent fields are audited by meaning rather than removed with a single global rule.</td>
    <td valign="top"><b>Duration is strongly skewed.</b> The distribution view keeps the median, the 18-second reference, and the tail behavior visually distinct.</td>
  </tr>
</table>

Raw rows are reconciled into governed analytical scopes instead of being
treated as one undifferentiated table. The diagram makes the early-standard,
late-standard, and random-exposure paths explicit.

<img src="docs/assets/experiment-story/P06_row_reconciliation.png" width="100%" alt="Raw-to-Silver row reconciliation across exposure domains">

Pooled discrimination and user-level discrimination answer different
questions. Event-weighted user-gAUC is therefore reported alongside pooled
metrics instead of being replaced by them.

<img src="docs/assets/experiment-story/P07_pooled_vs_usergauc.png" width="100%" alt="Pooled metric versus event-weighted user-gAUC">

**Takeaway:** missingness, heavy tails, row reconciliation, and metric
aggregation are part of the scientific design—not cleanup details.

### 3. Build leakage-safe features and diagnose optimization

BL1 and BL2 are nested sparse logistic regressions: BL2 adds the audited
long-sequence feature block while preserving the shared evaluation contract.
This makes the BL2-minus-BL1 contrast interpretable.

<img src="docs/assets/experiment-story/P08_nested_feature_blocks.png" width="100%" alt="Nested BL1 and BL2 feature blocks">

<table>
  <tr>
    <td width="50%"><img src="docs/assets/experiment-story/P09a_sgd_prediction_shape.png" width="100%" alt="Ranking and prediction shape under SGD"></td>
    <td width="50%"><img src="docs/assets/experiment-story/P09b_optimizer_adequacy.png" width="100%" alt="Optimizer adequacy comparison for SGD and Adam"></td>
  </tr>
  <tr>
    <td valign="top"><b>Prediction shape is a diagnostic.</b> Ranking behavior alone can hide an optimization problem in the probability scale.</td>
    <td valign="top"><b>Optimizer adequacy is tested, not assumed.</b> Repeated Adam fits and the recorded SGD runs are shown against the pre-specified gate.</td>
  </tr>
</table>

The frozen release configuration uses GPU Adam with the same fitting settings
for BL1 and BL2; it does not change the model family.

### 4. Freeze chronology, then compare domains

The temporal audit separates the standard and random exposure domains and
records exactly which slices were consumed at each stage.

<img src="docs/assets/experiment-story/P10_data_timeline.png" width="100%" alt="Audited timeline for standard and random exposure data">

The v010 random slice contains **43,027 rows**. The v011 and v012 ledgers each
reconcile to the same 43,027-row, hash-pinned canonical slice. Their use is a
**post-audit temporal replay**, not a new independent data confirmation.

The primary comparison uses common 0–1 precision-recall axes and a separate
forest panel for the paired user-cluster bootstrap effect estimates.

<img src="docs/assets/experiment-story/P11_main_result.png" width="100%" alt="BL1 versus BL2 precision-recall curves and delta AP forest plot">

| Evaluation domain | Rows | BL1 AP | BL2 AP | BL2 − BL1 ΔAP (95% CI) |
|---|---:|---:|---:|---:|
| Validation | 886,452 | 0.549387 | 0.585013 | +0.035626 [0.031302, 0.040028] |
| Sealed | 4,431,299 | 0.537281 | 0.578308 | +0.040651 [0.036937, 0.044402] |
| Random | 43,027 | 0.169530 | 0.196344 | +0.026188 [0.018296, 0.034667] |

All three paired 95% intervals exclude zero. The precision-recall panels and
the forest estimates use their explicitly declared row sets and should not be
treated as interchangeable summaries.

<img src="docs/assets/experiment-story/P11_robustness_strip_candidate.png" width="100%" alt="Training-user-fraction robustness strip with three seeds per fraction">

The supplementary sensitivity check uses five discrete training-user
fractions and three seeds per fraction. All 15 lower confidence bounds exceed
zero. It reuses Validation rows, so it is **not independent confirmation** and
does not introduce a neural-network result.

### 5. Repair probability quality without changing ranking

Monotone calibration changes the probability scale while preserving ranking.
The held-out calibration evaluation contains **23,752 rows**, with a true
event rate of **0.086856**.

<img src="docs/assets/experiment-story/P12_calibration.png" width="100%" alt="Reliability diagram, probability shift, and calibration metrics">

| Metric | Before | After | Interpretation |
|---|---:|---:|---|
| Log Loss | 0.512076 | 0.268939 | Δ = −0.243137 [−0.258633, −0.226836] |
| Brier score | 0.169978 | 0.074827 | Δ = −0.095151 [−0.102219, −0.087959] |
| ECE (20 bins) | 0.281335 | 0.006724 | Descriptive; no bootstrap CI claimed |
| AP | 0.197730 | 0.197730 | Ranking invariant |
| ROC-AUC | 0.727128 | 0.727128 | Ranking invariant |
| event-gAUC | 0.623640 | 0.623640 | Ranking invariant |

**Takeaway:** calibration substantially improves probability quality without
being misreported as a ranking improvement.

### 6. Replay the final model on a later internal window

The final replay contains **12,399 rows from 857 users**. It is an internal
temporal replay, not external independent validation.

<img src="docs/assets/experiment-story/P13_replay_ranking.png" width="100%" alt="Temporal replay ranking results">

The replay AP values are 0.187896 for the incumbent, 0.176476 for retrained
BL1, and 0.204928 for retrained BL2. Against the incumbent, BL2 improves AP by
**+0.017032 [0.005265, 0.031319]** and event-gAUC by
**+0.024183 [0.007412, 0.041419]**. Against retrained BL1, the corresponding
gains are **+0.028452 [0.015166, 0.041826]** and
**+0.031612 [0.011644, 0.050783]**.

<img src="docs/assets/experiment-story/P14_replay_probability.png" width="100%" alt="Temporal replay probability-quality improvements">

Probability improvements are plotted rightward after sign-normalizing loss
reductions. Log Loss is reduced by **0.003307 [0.001725, 0.005145]** versus the
incumbent and **0.008684 [0.006119, 0.011364]** versus retrained BL1. Brier
score is reduced by **0.000752 [0.000333, 0.001243]** and
**0.001515 [0.000970, 0.002099]**, respectively.

**Takeaway:** ranking and probability quality both improve on the internal
replay, with paired intervals above zero after orienting every estimate so that
improvement is to the right.

### 7. Turn the result into an auditable Agent workflow

The system keeps ranking and calibration as two coordinated but separately
evaluated tracks.

<img src="docs/assets/experiment-story/P15_rank_calibrate.png" width="100%" alt="Separate ranking and calibration tracks">

The evidence boundary prevents a producer artifact from becoming a release
claim merely because the file exists. Admission, claim review, and approval
remain explicit gates.

<img src="docs/assets/experiment-story/P16_evidence_boundary.png" width="100%" alt="Supported and blocked claims at the evidence boundary">

The contribution is therefore a chain: governed data, point-in-time features,
paired evaluation, probability repair, and a fail-closed Agent Harness.

<img src="docs/assets/experiment-story/P17_contribution_chain.png" width="100%" alt="Experiment and Agent contribution chain">

Future work must pass new gates rather than extend the present claim by prose.
The next validation step is evaluation on other suitable public datasets, plus
the additional release checks shown below; these are planned, not completed.

<img src="docs/assets/experiment-story/P18_future_gates.png" width="100%" alt="Future evidence and release gates">

**Overall conclusion:** BL2 provides a consistent ranking gain over BL1 across
the audited evaluation domains, monotone calibration repairs probability
quality, and the later window supports an internal temporal-replay result. The
repository deliberately stops short of claiming external validation, online
causal lift, or release approval.

## Run the Agent demo

From the repository root on PowerShell:

```powershell
python scripts/run_agent_system_demo_v001.py `
  --config configs/agent_system_v001.yaml `
  --output-root artifacts/agent_runs `
  --verify
```

Expected key output:

```text
terminal_state = waiting_for_evidence
verification.verified = true
```

`waiting_for_evidence` is the correct safe terminal state. The default provider
does not supply scientific evidence, so the Agent cannot promote the GPU demo
snapshot into a release claim.

Install and run the Agent regression tests:

```powershell
python -m pip install -e ".[test]"
python -m pytest tests/agent_system -q
```

The three-minute demo sequence is documented in
[`demo/DEMO_GUIDE.md`](demo/DEMO_GUIDE.md).

## Reproduce from the official KuaiRand data

This repository uses no Git LFS and does not mirror the KuaiRand dataset. The
shortest supported path is:

```powershell
python reproducibility/scripts/download_kuairand_1k.py
python reproducibility/scripts/verify_reproduction.py --raw
python -m pip install -r reproducibility/environment/requirements-release-v010-v012.txt
python reproducibility/scripts/run_reproduction.py --build-silver
python reproducibility/scripts/verify_reproduction.py --silver
```

The downloader accepts the official default URL or a mirror explicitly chosen
by the user. The archive transport is not treated as data identity: after safe
extraction, every expected CSV must match the frozen file size and SHA-256 in
[`reproducibility/manifests/raw-files.json`](reproducibility/manifests/raw-files.json).

Key reproducibility records:

- [complete workflow](reproducibility/README.md);
- [release environment pins](reproducibility/environment/reference-runtime.json);
- [supplementary v013 runtime](reproducibility/environment/supplementary-learning-curve-v013-runtime.json);
- [seeds, temporal splits, bootstrap protocol, and contract hashes](reproducibility/seeds-and-splits.yaml);
- [expected Silver identities](reproducibility/manifests/silver-expected.json);
- [known reproducibility gaps](reproducibility/REPRODUCIBILITY_GAPS.md);
- [QA report](reproducibility/QA_REPORT.md).

The formal Silver builder intentionally excludes
`video_features_statistic_1k.csv` because it contains post-hoc aggregates and
was not joined into the released Silver layer.

## Agent system

Six roles collaborate under least-privilege rules:

| Role | Responsibility |
|---|---|
| Manager | Freeze the research contract and task boundary. |
| Data Auditor | Audit data scope and temporal leakage without recleaning data. |
| Feature Miner | Produce point-in-time feature specifications. |
| Causal Evaluator | Request, verify, and evaluate research evidence. |
| Safety Reviewer | Constrain claim scope and stop unsupported statements. |
| Feature Publisher | Publish only when evidence and approval are both satisfied; support rollback. |

Ten typed skills are available:

`register_research_contract` · `audit_project_boundary` ·
`propose_feature_specs` · `detect_temporal_leakage` ·
`request_research_evidence` · `evaluate_research_evidence` ·
`review_claim_boundaries` · `assess_release_readiness` ·
`publish_feature_package` · `rollback_feature_package`

Default workflow:

```text
research contract -> boundary audit -> feature specification -> leakage audit
                  -> evidence request -> gate evaluation -> claim review
                  -> release-readiness assessment
```

Every skill call checks role permission, budget, idempotency key, and output
schema. Events are appended to a hash chain. Missing or mismatched evidence
stops the workflow; the Agent neither fabricates placeholder results nor
publishes automatically.

## Repository layout

```text
NanyangYS_Agent/
├─ README.md                         # English entry point and branch guide
├─ README_CN.md                      # Chinese documentation
├─ pyproject.toml
├─ configs/
│  └─ agent_system_v001.yaml
├─ scripts/run_agent_system_demo_v001.py
├─ src/kuairand_longseq/
│  ├─ agents/
│  ├─ evidence/
│  ├─ harness/
│  └─ skills/
├─ tests/agent_system/test_agent_system_v001.py
├─ demo/
│  ├─ DEMO_GUIDE.md
│  └─ gpu_train_only_snapshot/       # compact historical summaries only
├─ docs/assets/experiment-story/     # white-background README figures
│  └─ transparent-backup/            # original transparent PNG exports
├─ reproducibility/                  # download, verification, environment, QA
├─ reference/
│  ├─ evidence-summaries/            # compact v007-v012 direct summaries
│  ├─ figures/
│  ├─ reports/
│  └─ kuairand-longseq-agent/        # source, contracts, configs, tests
└─ artifacts/agent_runs/             # generated locally and rebuildable
```

`demo/gpu_train_only_snapshot` contains only the small summaries required by
the demo. It does not include the original data, training environment, or the
54 MB row-level prediction parquet.

## Unsupported claims

- Do not describe the GPU demo snapshot as Validation, Gold, sealed-test, or a
  formal release result.
- Do not describe BL2 as a validated neural or long-sequence model; the demo
  snapshot is a Train-only BL0/BL1/BL2 comparison.
- Do not pass a producer `run_manifest.json` or
  `evidence_bundle_manifest.yaml` directly to `--evidence-manifest`; those are
  not the consumer Evidence manifest required by the Agent contract.
- Do not claim online causal lift, full-catalog retrieval quality, or arbitrary
  policy evaluation.
- Do not claim that the Agent trained a model, invoked an LLM/GPU, or received
  release approval.
- Do not describe v012 temporal replay as external independent validation.
- Do not describe the supplementary learning curve as independent
  confirmation or as a neural-network result; it reuses Validation rows.

The runtime contract is defined by `configs/agent_system_v001.yaml`,
`src/kuairand_longseq/`, and the safety regression tests.

## License and dataset attribution

### Code

Project code is licensed under the **Apache License 2.0**. See
[`LICENSE`](LICENSE).

### Dataset

| Dataset | Source | Dataset license | Citation |
|---|---|---|---|
| **KuaiRand-1K** | Kuaishou | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Gao et al., CIKM 2022 |
| Tencent video data | Tencent | Planned; not included | To be determined |
| ByteDance video data | ByteDance | Planned; not included | To be determined |
| YouTube data | YouTube | Planned; not included | To be determined |

KuaiRand was released by Chongming Gao, Shijun Li, Yuan Zhang, Jiawei Chen,
Biao Li, Wenqiang Lei, Peng Jiang, Xiangnan He, and collaborators:

> Gao, C., Li, S., Zhang, Y., Chen, J., Li, B., Lei, W., Jiang, P., & He, X.
> (2022). *KuaiRand: An Unbiased Sequential Recommendation Dataset with
> Randomly Exposed Videos*. CIKM '22, Atlanta, GA, USA.
> [DOI: 10.1145/3511808.3557624](https://doi.org/10.1145/3511808.3557624)

- Dataset website: <https://kuairand.com>
- Official repository: <https://github.com/chongminggao/KuaiRand>
- Dataset license: **CC BY-SA 4.0**; attribution and share-alike obligations
  apply to dataset-derived distributions.

This repository does **not** distribute the KuaiRand dataset. Each user must
obtain it from an official source and comply with its license.
