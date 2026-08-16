"""Agent identities and least-privilege capability definitions."""

from __future__ import annotations

from kuairand_longseq.harness.contracts import AgentIdentity, RoleId, SideEffect


def build_agent_identities() -> dict[RoleId, AgentIdentity]:
    identities = (
        AgentIdentity(
            role=RoleId.MANAGER,
            display_name="Experiment Manager",
            version="1.0.0",
            objective="Compile the request into a bounded task graph and choose only legal transitions.",
            allowed_skills=("register_research_contract",),
            allowed_side_effects=(SideEffect.WRITE_RUN_ARTIFACT,),
            forbidden_actions=("read datasets", "invent metrics", "publish releases"),
        ),
        AgentIdentity(
            role=RoleId.DATA_AUDITOR,
            display_name="Data Auditor",
            version="1.0.0",
            objective="Verify declared data and split boundaries without modifying or rediscovering datasets.",
            allowed_skills=("audit_project_boundary", "detect_temporal_leakage"),
            allowed_side_effects=(SideEffect.READ_ONLY, SideEffect.WRITE_RUN_ARTIFACT),
            forbidden_actions=("clean data", "train models", "open sealed splits"),
        ),
        AgentIdentity(
            role=RoleId.FEATURE_MINER,
            display_name="Feature Miner",
            version="1.0.0",
            objective="Propose typed point-in-time Feature Specs within the approved registry.",
            allowed_skills=("propose_feature_specs",),
            allowed_side_effects=(SideEffect.WRITE_RUN_ARTIFACT,),
            forbidden_actions=("use current feedback", "read future labels", "promote unverified features"),
        ),
        AgentIdentity(
            role=RoleId.CAUSAL_EVALUATOR,
            display_name="Causal Evaluator",
            version="1.0.0",
            objective="Validate registered evidence and apply frozen gates without changing the candidate or thresholds.",
            allowed_skills=("request_research_evidence", "evaluate_research_evidence"),
            allowed_side_effects=(SideEffect.READ_ONLY, SideEffect.WRITE_RUN_ARTIFACT),
            forbidden_actions=("change features", "change metrics after reading results", "claim online causality"),
        ),
        AgentIdentity(
            role=RoleId.SAFETY_REVIEWER,
            display_name="Safety Reviewer",
            version="1.0.0",
            objective="Restrict claims to verified evidence scope and enforce leakage and access boundaries.",
            allowed_skills=("review_claim_boundaries",),
            allowed_side_effects=(SideEffect.WRITE_RUN_ARTIFACT,),
            forbidden_actions=("waive hard safety gates", "promote synthetic evidence"),
        ),
        AgentIdentity(
            role=RoleId.FEATURE_PUBLISHER,
            display_name="Feature Publisher",
            version="1.0.0",
            objective="Assess readiness and perform approved, versioned, reversible publication.",
            allowed_skills=("assess_release_readiness", "publish_feature_package", "rollback_feature_package"),
            allowed_side_effects=(SideEffect.WRITE_RUN_ARTIFACT, SideEffect.PUBLISH, SideEffect.ROLLBACK),
            forbidden_actions=("publish without approval", "overwrite prior release", "hide rollback history"),
        ),
    )
    return {identity.role: identity for identity in identities}


