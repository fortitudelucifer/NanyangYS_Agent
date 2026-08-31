#!/usr/bin/env python3
"""Read-only maintenance audit for the KuaiRand governed project."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
COLLISION_PATTERN = re.compile(r"\(\d+\)$")
REPORT_NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_[^\\/]+_v(?P<version>\d{3})\.md$", re.IGNORECASE)
SCRIPT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*_v(?P<version>\d{3})\.py$", re.IGNORECASE)
VERSIONED_STREAMS = {
    "pre-v010": lambda version: version < 10,
    "v011-target-domain-calibration": lambda version: version == 11,
    "v012-target-domain-retraining": lambda version: version == 12,
    "v013-neural-sequence": lambda version: version == 13,
}


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def files_matching(predicate):
    return [path for path in PROJECT_ROOT.rglob("*") if path.is_file() and predicate(path)]


def versioned_path_issues(base: Path, suffix: str, pattern: re.Pattern[str]) -> tuple[list[str], list[str]]:
    """Return malformed names and correctly named files in the wrong stream."""
    malformed: list[str] = []
    misplaced: list[str] = []
    if not base.is_dir():
        return malformed, misplaced

    for path in base.rglob(f"*{suffix}"):
        if path.name.lower() == "readme.md":
            continue
        match = pattern.fullmatch(path.name)
        if not match:
            malformed.append(relative(path))
            continue
        parts = path.relative_to(base).parts
        stream = parts[0] if parts else ""
        version = int(match.group("version"))
        allowed = VERSIONED_STREAMS.get(stream)
        if allowed is None or not allowed(version):
            misplaced.append(relative(path))
    return malformed, misplaced


def main() -> int:
    parser = argparse.ArgumentParser(description="Report workspace-layout maintenance issues without modifying files.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of a readable report.")
    args = parser.parse_args()

    required = [
        WORKSPACE_ROOT / "README.md",
        WORKSPACE_ROOT / "WORKSPACE_CONVENTIONS.md",
        PROJECT_ROOT / "PROJECT_INDEX.md",
        PROJECT_ROOT / "artifacts" / "README.md",
        PROJECT_ROOT / "experiments" / "README.md",
        PROJECT_ROOT / "reports" / "generated" / "README.md",
        PROJECT_ROOT / "reports" / "experiments" / "README.md",
        PROJECT_ROOT / "scripts" / "experiments" / "README.md",
    ]
    versioned_directories = [
        base / stream
        for base in (PROJECT_ROOT / "reports" / "experiments", PROJECT_ROOT / "scripts" / "experiments")
        for stream in VERSIONED_STREAMS
    ]
    collisions = files_matching(lambda path: bool(COLLISION_PATTERN.search(path.stem)))
    bytecode = files_matching(lambda path: path.suffix == ".pyc")
    quick_dirs = sorted(path for path in (PROJECT_ROOT / "reports" / "generated").glob("*_quick") if path.is_dir())
    temp_dirs = [
        path
        for path in [PROJECT_ROOT / ".pytest_cache", PROJECT_ROOT / "artifacts" / ".silver_build_tmp"]
        if path.exists()
    ]
    malformed_reports, misplaced_reports = versioned_path_issues(
        PROJECT_ROOT / "reports" / "experiments", ".md", REPORT_NAME_PATTERN
    )
    malformed_scripts, misplaced_scripts = versioned_path_issues(
        PROJECT_ROOT / "scripts" / "experiments", ".py", SCRIPT_NAME_PATTERN
    )
    report = {
        "workspace_root": str(WORKSPACE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "missing_indexes": [str(path.relative_to(WORKSPACE_ROOT)) for path in required if not path.is_file()],
        "collision_files": [relative(path) for path in collisions],
        "python_bytecode_files": [relative(path) for path in bytecode],
        "quick_output_directories": [relative(path) for path in quick_dirs],
        "temporary_directories": [relative(path) for path in temp_dirs],
        "missing_versioned_directories": [relative(path) for path in versioned_directories if not path.is_dir()],
        "misnamed_future_reports": malformed_reports,
        "misplaced_future_reports": misplaced_reports,
        "misnamed_future_scripts": malformed_scripts,
        "misplaced_future_scripts": misplaced_scripts,
    }
    report["ok"] = not any(report[key] for key in report if key not in {"workspace_root", "project_root", "ok"})

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("WORKSPACE_LAYOUT_AUDIT")
        print(f"- workspace: {report['workspace_root']}")
        print(f"- project: {report['project_root']}")
        for key in (
            "missing_indexes",
            "collision_files",
            "python_bytecode_files",
            "quick_output_directories",
            "temporary_directories",
            "missing_versioned_directories",
            "misnamed_future_reports",
            "misplaced_future_reports",
            "misnamed_future_scripts",
            "misplaced_future_scripts",
        ):
            values = report[key]
            print(f"- {key}: {len(values)}")
            for value in values:
                print(f"  - {value}")
        print(f"- status: {'OK' if report['ok'] else 'ACTION_REQUIRED'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
