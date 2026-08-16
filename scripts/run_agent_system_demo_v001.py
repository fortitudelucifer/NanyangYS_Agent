"""Run the offline CausalFeatureOps Agent Infra demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kuairand_longseq.agents.workflow import CausalFeatureOpsWorkflow, verify_run_directory
from kuairand_longseq.evidence import ManifestEvidenceProvider, NullEvidenceProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "agent_system_v001.yaml")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts" / "agent_runs")
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def exit_code_for(terminal_state: str, verified: bool | None) -> int:
    if verified is False:
        return 2
    if terminal_state in {"blocked", "failed"}:
        return 3
    return 0


def main() -> int:
    args = parse_args()
    provider = ManifestEvidenceProvider(args.evidence_manifest) if args.evidence_manifest else NullEvidenceProvider()
    workflow = CausalFeatureOpsWorkflow(
        config_path=args.config, output_root=args.output_root,
        evidence_provider=provider, run_id=args.run_id,
    )
    run_dir = workflow.run()
    verification = verify_run_directory(run_dir) if args.verify else {"verified": None}
    print(json.dumps({"run_dir": str(run_dir), "terminal_state": workflow.state.phase.value,
                      "verification": verification}, ensure_ascii=False, indent=2))
    return exit_code_for(workflow.state.phase.value, verification.get("verified"))


if __name__ == "__main__":
    raise SystemExit(main())

