"""Fail-closed state transitions for the research agent workflow."""

from __future__ import annotations

from .contracts import RunPhase, RunState, TERMINAL_PHASES


ALLOWED_TRANSITIONS: dict[RunPhase, set[RunPhase]] = {
    RunPhase.CREATED: {RunPhase.CONTRACT_REGISTERED, RunPhase.BLOCKED, RunPhase.FAILED},
    RunPhase.CONTRACT_REGISTERED: {RunPhase.INPUTS_AUDITED, RunPhase.BLOCKED, RunPhase.FAILED},
    RunPhase.INPUTS_AUDITED: {RunPhase.FEATURES_PROPOSED, RunPhase.BLOCKED, RunPhase.FAILED},
    RunPhase.FEATURES_PROPOSED: {RunPhase.EVIDENCE_EVALUATED, RunPhase.BLOCKED, RunPhase.FAILED},
    RunPhase.EVIDENCE_EVALUATED: {RunPhase.CLAIMS_REVIEWED, RunPhase.BLOCKED, RunPhase.FAILED},
    RunPhase.CLAIMS_REVIEWED: {
        RunPhase.WAITING_FOR_EVIDENCE,
        RunPhase.WAITING_FOR_APPROVAL,
        RunPhase.BLOCKED,
        RunPhase.FAILED,
    },
    RunPhase.WAITING_FOR_APPROVAL: {RunPhase.RELEASED, RunPhase.BLOCKED, RunPhase.FAILED},
    RunPhase.RELEASED: {RunPhase.ROLLED_BACK},
    RunPhase.WAITING_FOR_EVIDENCE: set(),
    RunPhase.ROLLED_BACK: set(),
    RunPhase.BLOCKED: set(),
    RunPhase.FAILED: set(),
}


def transition(state: RunState, target: RunPhase, *, reason: str | None = None) -> None:
    if state.phase in TERMINAL_PHASES and target not in ALLOWED_TRANSITIONS.get(state.phase, set()):
        raise ValueError(f"terminal phase {state.phase.value} cannot transition to {target.value}")
    allowed = ALLOWED_TRANSITIONS.get(state.phase, set())
    if target not in allowed:
        raise ValueError(f"illegal transition: {state.phase.value} -> {target.value}")
    state.phase = target
    state.terminal_reason = reason if target in TERMINAL_PHASES else None
