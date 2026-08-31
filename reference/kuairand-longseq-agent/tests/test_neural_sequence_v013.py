from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import yaml

from kuairand_longseq.models.neural_sequence_v013 import (
    HISTORY_LENGTH_BY_VARIANT,
    SCIENTIFIC_VARIANTS,
    SyntheticV013Config,
    SyntheticV013StressModel,
    make_synthetic_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "neural_sequence_candidate_model_v013"
    / "contract_v013.yaml"
)
PREFLIGHT_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "experiments"
    / "v013-neural-sequence"
    / "gpu_preflight_v013.py"
)


@pytest.fixture
def tiny_config() -> SyntheticV013Config:
    return SyntheticV013Config(
        author_vocab_size=31,
        music_vocab_size=37,
        tag_vocab_size=41,
        upload_type_vocab_size=7,
        scene_vocab_size=5,
        dropout=0.0,
    )


def test_synthetic_registry_matches_contract() -> None:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    registered = tuple(
        row["model_id"]
        for row in contract["model_registry"]
        if row["model_id"] in SCIENTIFIC_VARIANTS
    )
    assert registered == SCIENTIFIC_VARIANTS
    assert HISTORY_LENGTH_BY_VARIANT == {
        "MLP_STATIC": 0,
        "MLP_H2": 0,
        "DIN10": 10,
        "DIN50": 50,
        "DIN200": 200,
        "HIER500": 500,
    }


@pytest.mark.parametrize("variant", SCIENTIFIC_VARIANTS)
def test_all_registered_variants_produce_finite_logits(
    variant: str, tiny_config: SyntheticV013Config
) -> None:
    torch.manual_seed(20260824)
    device = torch.device("cpu")
    model = SyntheticV013StressModel(variant, tiny_config).eval()
    batch = make_synthetic_batch(
        variant, 3, tiny_config, device=device, seed=20260824
    )
    with torch.inference_mode():
        logits = model(batch)
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_masked_padding_cannot_change_din_prediction(
    tiny_config: SyntheticV013Config,
) -> None:
    torch.manual_seed(20260824)
    model = SyntheticV013StressModel("DIN10", tiny_config).eval()
    batch = make_synthetic_batch(
        "DIN10", 2, tiny_config, device=torch.device("cpu"), seed=20260824
    )
    batch["history_mask"][:, -2:] = False
    with torch.inference_mode():
        expected = model(batch)
    changed = {name: value.clone() for name, value in batch.items()}
    for name in (
        "history_author_id",
        "history_music_id",
        "history_tag_id",
        "history_upload_type_id",
    ):
        changed[name][:, -2:] = 1
    changed["history_static_numeric"][:, -2:] = 123.0
    changed["history_numeric"][:, -2:] = -456.0
    with torch.inference_mode():
        observed = model(changed)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_hier500_uses_recent_200_and_masked_older_summary(
    tiny_config: SyntheticV013Config,
) -> None:
    torch.manual_seed(20260824)
    model = SyntheticV013StressModel("HIER500", tiny_config).eval()
    batch = make_synthetic_batch(
        "HIER500", 2, tiny_config, device=torch.device("cpu"), seed=20260824
    )
    assert batch["history_mask"].shape == (2, 500)
    with torch.inference_mode():
        logits = model(batch)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_synthetic_batch_is_reproducible(tiny_config: SyntheticV013Config) -> None:
    first = make_synthetic_batch(
        "DIN50", 2, tiny_config, device=torch.device("cpu"), seed=20260824
    )
    second = make_synthetic_batch(
        "DIN50", 2, tiny_config, device=torch.device("cpu"), seed=20260824
    )
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0.0, atol=0.0)


def test_preflight_validate_only_preserves_fail_closed_contract() -> None:
    completed = subprocess.run(
        [sys.executable, str(PREFLIGHT_SCRIPT), "--validate-only"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)
    assert payload["formal_execution_authorized"] is False
    assert payload["synthetic_gpu_preflight_authorized"] is True
    assert payload["evidence_level"] == "engineering_synthetic_only"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA preflight requires CUDA")
def test_one_synthetic_hier500_cuda_optimizer_step(
    tiny_config: SyntheticV013Config,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
    model = SyntheticV013StressModel("HIER500", tiny_config).to(device)
    batch = make_synthetic_batch(
        "HIER500", 4, tiny_config, device=device, seed=20260824
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = F.binary_cross_entropy_with_logits(model(batch), batch["target"])
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

