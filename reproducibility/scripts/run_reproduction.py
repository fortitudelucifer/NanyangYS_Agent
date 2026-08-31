from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "reference" / "kuairand-longseq-agent"
SCRIPT_ROOT = REPO_ROOT / "reproducibility" / "scripts"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run explicit KuaiRand reproduction stages.")
    parser.add_argument("--build-silver", action="store_true")
    parser.add_argument("--capture-environment", action="store_true")
    args = parser.parse_args()
    if not args.build_silver and not args.capture_environment:
        raise SystemExit("select at least one explicit stage")

    if args.capture_environment:
        run([sys.executable, str(SCRIPT_ROOT / "capture_environment.py")])

    if args.build_silver:
        run([sys.executable, str(SCRIPT_ROOT / "verify_reproduction.py"), "--raw"])
        run(
            [
                sys.executable,
                str(AGENT_ROOT / "scripts" / "build_silver.py"),
                "--project-root",
                str(AGENT_ROOT),
            ],
            cwd=AGENT_ROOT,
        )
        run([sys.executable, str(SCRIPT_ROOT / "verify_reproduction.py"), "--silver"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
