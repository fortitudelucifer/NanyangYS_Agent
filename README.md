# NanyangYS Agent

[English](README.md) | [中文](README_CN.md)

> [!IMPORTANT]
> **Start from `main`.** It is the authoritative, lightweight, zero-LFS branch.
> The repository intentionally does **not** distribute KuaiRand Raw data,
> governed Silver tables, row-level predictions, feature matrices, bootstrap
> arrays, checkpoints, or model states. Download KuaiRand-1K from its official
> source and use the versioned reproducibility workflow below.

## Branch guide

| Branch | Status | Guidance |
|---|---|---|
| [`main`](https://github.com/fortitudelucifer/NanyangYS_Agent/tree/main) | Authoritative and supported | Clone or branch from here. It contains the Agent, compact evidence summaries, research code, reports, figures, and the zero-LFS reproducibility kit. |
| [`jkrec-reproducibility-kit`](https://github.com/fortitudelucifer/NanyangYS_Agent/tree/jkrec-reproducibility-kit) | Merged through [PR #1](https://github.com/fortitudelucifer/NanyangYS_Agent/pull/1) | Historical source branch for the reproducibility-kit import. Do not treat it as newer than `main`. |
| [`jkrec-v013-runtime-metadata`](https://github.com/fortitudelucifer/NanyangYS_Agent/tree/jkrec-v013-runtime-metadata) | Merged through [PR #2](https://github.com/fortitudelucifer/NanyangYS_Agent/pull/2) | Historical source branch for the supplementary v013 runtime metadata. Do not treat it as an independent release line. |
| `jkrec-full-data-local-backup` | Local-only; never pushed | A private recovery branch used during import preparation. It contains large-data references and is not part of the GitHub repository. Never push or merge it. |

The two remote feature branches are retained only for auditability. All of
their accepted content is already in `main`.

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
