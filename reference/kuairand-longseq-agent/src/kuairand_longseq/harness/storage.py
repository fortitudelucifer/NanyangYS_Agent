"""Artifact and append-only event storage for agent runs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import ArtifactRef, RunEvent, RunState, to_jsonable


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactStore:
    """Only writes inside one run directory and always uses atomic replacement."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._produced: dict[str, ArtifactRef] = {}

    @property
    def produced(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._produced[key] for key in sorted(self._produced))

    def resolve(self, relative_path: str) -> Path:
        candidate = (self.run_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.run_dir)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes run directory: {relative_path}") from exc
        return candidate

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def write_bytes(
        self,
        relative_path: str,
        payload: bytes,
        *,
        media_type: str,
        producer: str,
        role: str = "supporting",
    ) -> ArtifactRef:
        path = self.resolve(relative_path)
        self._atomic_write(path, payload)
        reference = ArtifactRef(
            path=relative_path.replace("\\", "/"),
            size_bytes=len(payload),
            sha256=sha256_bytes(payload),
            media_type=media_type,
            producer=producer,
            role=role,
        )
        self._produced[reference.path] = reference
        return reference

    def write_json(self, relative_path: str, value: Any, *, producer: str, role: str = "supporting") -> ArtifactRef:
        payload = canonical_json_bytes(value) + b"\n"
        return self.write_bytes(
            relative_path,
            payload,
            media_type="application/json",
            producer=producer,
            role=role,
        )

    def write_text(self, relative_path: str, text: str, *, producer: str, role: str = "supporting") -> ArtifactRef:
        return self.write_bytes(
            relative_path,
            text.encode("utf-8"),
            media_type="text/markdown; charset=utf-8" if relative_path.endswith(".md") else "text/plain; charset=utf-8",
            producer=producer,
            role=role,
        )

    def verify(self, artifact: ArtifactRef) -> bool:
        path = self.resolve(artifact.path)
        return (
            path.is_file()
            and path.stat().st_size == artifact.size_bytes
            and sha256_file(path) == artifact.sha256
        )


class EventStore:
    """Hash-chained JSONL event log with replay verification."""

    def __init__(self, artifact_store: ArtifactStore, run_id: str) -> None:
        self.artifact_store = artifact_store
        self.run_id = run_id
        self.path = artifact_store.resolve("events.jsonl")
        self.sequence = 0
        self.previous_sha256 = "0" * 64
        if self.path.exists() and self.path.stat().st_size:
            events = self.read_verified(self.path)
            if events:
                self.sequence = events[-1].sequence
                self.previous_sha256 = events[-1].event_sha256

    def append(self, event_type: str, actor: str, payload: dict[str, Any]) -> RunEvent:
        sequence = self.sequence + 1
        core = {
            "sequence": sequence,
            "run_id": self.run_id,
            "event_type": event_type,
            "actor": actor,
            "payload": to_jsonable(payload),
            "previous_sha256": self.previous_sha256,
        }
        event_sha256 = sha256_bytes(canonical_json_bytes(core))
        event = RunEvent(event_sha256=event_sha256, **core)
        line = canonical_json_bytes(event) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self.sequence = sequence
        self.previous_sha256 = event_sha256
        self.artifact_store._produced["events.jsonl"] = ArtifactRef(
            path="events.jsonl",
            size_bytes=self.path.stat().st_size,
            sha256=sha256_file(self.path),
            media_type="application/x-ndjson",
            producer="event_store",
            role="audit_log",
        )
        return event

    @staticmethod
    def read_verified(path: Path) -> list[RunEvent]:
        events: list[RunEvent] = []
        previous = "0" * 64
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                raw = json.loads(line)
                observed_hash = raw.pop("event_sha256")
                if raw["previous_sha256"] != previous:
                    raise ValueError(f"event hash chain broken at line {index}")
                expected_hash = sha256_bytes(canonical_json_bytes(raw))
                if observed_hash != expected_hash:
                    raise ValueError(f"event hash mismatch at line {index}")
                event = RunEvent(event_sha256=observed_hash, **raw)
                events.append(event)
                previous = observed_hash
        return events


def checkpoint_state(store: ArtifactStore, state: RunState) -> ArtifactRef:
    """Write the latest state without adding itself to the scientific evidence."""

    return store.write_json("state.json", state, producer="harness", role="checkpoint")

