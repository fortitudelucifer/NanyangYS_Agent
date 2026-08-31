"""Synthetic-only neural geometry for the v013 CUDA engineering preflight.

This module mirrors the registered v013 tensor dimensions and history windows,
but it is not the release model.  It deliberately consumes generated tensors
only; the Gold feature contract and production encoders remain to be frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import nn


SCIENTIFIC_VARIANTS: Final[tuple[str, ...]] = (
    "MLP_STATIC",
    "MLP_H2",
    "DIN10",
    "DIN50",
    "DIN200",
    "HIER500",
)
HISTORY_LENGTH_BY_VARIANT: Final[dict[str, int]] = {
    "MLP_STATIC": 0,
    "MLP_H2": 0,
    "DIN10": 10,
    "DIN50": 50,
    "DIN200": 200,
    "HIER500": 500,
}


@dataclass(frozen=True)
class SyntheticV013Config:
    """Provisional cardinalities plus contract-registered neural dimensions."""

    author_vocab_size: int = 500_000
    music_vocab_size: int = 1_000_000
    tag_vocab_size: int = 100_000
    upload_type_vocab_size: int = 128
    scene_vocab_size: int = 32
    categorical_embedding_dim: int = 32
    event_embedding_dim: int = 64
    attention_hidden_dim: int = 64
    static_numeric_dim: int = 6
    history_numeric_dim: int = 8
    user_context_dim: int = 16
    h2_dim: int = 20
    fusion_hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.10

    def validate(self) -> None:
        vocabularies = (
            self.author_vocab_size,
            self.music_vocab_size,
            self.tag_vocab_size,
            self.upload_type_vocab_size,
            self.scene_vocab_size,
        )
        if any(value < 2 for value in vocabularies):
            raise ValueError("synthetic vocabularies must reserve index 0 for padding")
        dimensions = (
            self.categorical_embedding_dim,
            self.event_embedding_dim,
            self.attention_hidden_dim,
            self.static_numeric_dim,
            self.history_numeric_dim,
            self.user_context_dim,
            self.h2_dim,
            *self.fusion_hidden_dims,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all neural dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class ContentEmbeddingEncoder(nn.Module):
    """Shared synthetic content encoder for candidates and history events."""

    def __init__(self, config: SyntheticV013Config) -> None:
        super().__init__()
        dim = config.categorical_embedding_dim
        self.author = nn.Embedding(config.author_vocab_size, dim, padding_idx=0)
        self.music = nn.Embedding(config.music_vocab_size, dim, padding_idx=0)
        self.tag = nn.Embedding(config.tag_vocab_size, dim, padding_idx=0)
        self.upload_type = nn.Embedding(
            config.upload_type_vocab_size, dim, padding_idx=0
        )
        self.projection = nn.Sequential(
            nn.Linear(4 * dim + config.static_numeric_dim, config.event_embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        author_id: torch.Tensor,
        music_id: torch.Tensor,
        tag_id: torch.Tensor,
        upload_type_id: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            (
                self.author(author_id),
                self.music(music_id),
                self.tag(tag_id),
                self.upload_type(upload_type_id),
                numeric,
            ),
            dim=-1,
        )
        return self.projection(combined)


class CandidateAwareAttention(nn.Module):
    def __init__(self, event_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(4 * event_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        candidate: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if history.ndim != 3 or mask.shape != history.shape[:2]:
            raise ValueError("history and mask shapes do not align")
        if history.shape[1] == 0:
            return torch.zeros_like(candidate)
        query = candidate.unsqueeze(1).expand_as(history)
        features = torch.cat(
            (history, query, history - query, history * query), dim=-1
        )
        scores = self.score(features).squeeze(-1)
        safe_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(safe_scores, dim=1) * mask.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.sum(weights.unsqueeze(-1) * history, dim=1)


class SyntheticV013StressModel(nn.Module):
    """Synthetic stress surrogate for one registered neural candidate."""

    def __init__(self, variant: str, config: SyntheticV013Config) -> None:
        super().__init__()
        config.validate()
        if variant not in HISTORY_LENGTH_BY_VARIANT:
            raise ValueError(f"unknown v013 neural variant: {variant}")
        self.variant = variant
        self.history_length = HISTORY_LENGTH_BY_VARIANT[variant]
        self.uses_h2 = variant != "MLP_STATIC"
        self.uses_sequence = variant.startswith("DIN") or variant == "HIER500"
        self.config = config

        dim = config.event_embedding_dim
        self.content_encoder = ContentEmbeddingEncoder(config)
        self.context_encoder = nn.Sequential(
            nn.Embedding(config.scene_vocab_size, config.categorical_embedding_dim),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(
                config.categorical_embedding_dim + config.user_context_dim, dim
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        if self.uses_h2:
            self.h2_projection = nn.Sequential(
                nn.Linear(config.h2_dim, dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        if self.uses_sequence:
            self.history_projection = nn.Sequential(
                nn.Linear(dim + config.history_numeric_dim, dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.attention = CandidateAwareAttention(dim, config.attention_hidden_dim)
        if variant == "HIER500":
            self.older_projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
            self.older_gate = nn.Linear(3 * dim, dim)

        fusion_width = 2 * dim
        if self.uses_h2:
            fusion_width += dim
        if self.uses_sequence:
            fusion_width += dim
        layers: list[nn.Module] = []
        width = fusion_width
        for hidden in config.fusion_hidden_dims:
            layers.extend(
                (nn.Linear(width, hidden), nn.GELU(), nn.Dropout(config.dropout))
            )
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.fusion_head = nn.Sequential(*layers)

    def _encode_history(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        content = self.content_encoder(
            batch["history_author_id"],
            batch["history_music_id"],
            batch["history_tag_id"],
            batch["history_upload_type_id"],
            batch["history_static_numeric"],
        )
        return self.history_projection(
            torch.cat((content, batch["history_numeric"]), dim=-1)
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        total = torch.sum(values * weights, dim=1)
        count = torch.sum(weights, dim=1).clamp_min(1.0)
        return total / count

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate = self.content_encoder(
            batch["candidate_author_id"],
            batch["candidate_music_id"],
            batch["candidate_tag_id"],
            batch["candidate_upload_type_id"],
            batch["candidate_static_numeric"],
        )
        scene = self.context_encoder[0](batch["scene_id"])
        context = self.context_projection(
            torch.cat((scene, batch["user_context"]), dim=-1)
        )
        parts = [candidate, context]
        if self.uses_h2:
            parts.append(self.h2_projection(batch["h2_features"]))
        if self.uses_sequence:
            history = self._encode_history(batch)
            mask = batch["history_mask"]
            if self.variant == "HIER500":
                recent = self.attention(history[:, :200], candidate, mask[:, :200])
                older = self._masked_mean(history[:, 200:], mask[:, 200:])
                older = self.older_projection(older)
                gate = torch.sigmoid(
                    self.older_gate(torch.cat((recent, older, candidate), dim=-1))
                )
                sequence = recent + gate * older
            else:
                sequence = self.attention(history, candidate, mask)
            parts.append(sequence)
        return self.fusion_head(torch.cat(parts, dim=-1)).squeeze(-1)


def _categorical_tensor(
    shape: tuple[int, ...],
    vocabulary_size: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        1, vocabulary_size, shape, device=device, generator=generator
    )


def make_synthetic_batch(
    variant: str,
    batch_size: int,
    config: SyntheticV013Config,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Create a generated-only batch with padded strict-history-shaped tensors."""

    if variant not in HISTORY_LENGTH_BY_VARIANT:
        raise ValueError(f"unknown v013 neural variant: {variant}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config.validate()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    history_length = HISTORY_LENGTH_BY_VARIANT[variant]
    result = {
        "candidate_author_id": _categorical_tensor(
            (batch_size,), config.author_vocab_size,
            device=device, generator=generator,
        ),
        "candidate_music_id": _categorical_tensor(
            (batch_size,), config.music_vocab_size,
            device=device, generator=generator,
        ),
        "candidate_tag_id": _categorical_tensor(
            (batch_size,), config.tag_vocab_size,
            device=device, generator=generator,
        ),
        "candidate_upload_type_id": _categorical_tensor(
            (batch_size,), config.upload_type_vocab_size,
            device=device, generator=generator,
        ),
        "candidate_static_numeric": torch.randn(
            (batch_size, config.static_numeric_dim),
            device=device, generator=generator,
        ),
        "scene_id": _categorical_tensor(
            (batch_size,), config.scene_vocab_size,
            device=device, generator=generator,
        ),
        "user_context": torch.randn(
            (batch_size, config.user_context_dim),
            device=device, generator=generator,
        ),
        "h2_features": torch.randn(
            (batch_size, config.h2_dim), device=device, generator=generator
        ),
        "target": torch.randint(
            0, 2, (batch_size,), device=device, generator=generator
        ).to(torch.float32),
    }

    history_shape = (batch_size, history_length)
    if history_length:
        minimum = max(1, history_length // 4)
        lengths = torch.randint(
            minimum,
            history_length + 1,
            (batch_size,),
            device=device,
            generator=generator,
        )
        positions = torch.arange(history_length, device=device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
    else:
        mask = torch.zeros(history_shape, dtype=torch.bool, device=device)
    result["history_mask"] = mask

    for name, vocabulary_size in (
        ("history_author_id", config.author_vocab_size),
        ("history_music_id", config.music_vocab_size),
        ("history_tag_id", config.tag_vocab_size),
        ("history_upload_type_id", config.upload_type_vocab_size),
    ):
        values = _categorical_tensor(
            history_shape, vocabulary_size, device=device, generator=generator
        )
        result[name] = torch.where(mask, values, torch.zeros_like(values))
    history_static = torch.randn(
        (*history_shape, config.static_numeric_dim),
        device=device,
        generator=generator,
    )
    history_numeric = torch.randn(
        (*history_shape, config.history_numeric_dim),
        device=device,
        generator=generator,
    )
    result["history_static_numeric"] = history_static * mask.unsqueeze(-1)
    result["history_numeric"] = history_numeric * mask.unsqueeze(-1)
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

