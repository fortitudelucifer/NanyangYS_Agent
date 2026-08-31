from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = REPO_ROOT / "reproducibility" / "manifests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root: Path, manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for entry in payload["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        size = path.stat().st_size
        if size != int(entry["size_bytes"]):
            failures.append(
                f"size mismatch: {path} expected={entry['size_bytes']} actual={size}"
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            failures.append(
                f"sha256 mismatch: {path} expected={entry['sha256']} actual={actual_hash}"
            )
            continue
        print(f"OK {entry['path']} {size} {actual_hash}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify downloaded KuaiRand Raw files or rebuilt Silver outputs."
    )
    parser.add_argument("--raw", action="store_true", help="verify official extracted CSV files")
    parser.add_argument("--silver", action="store_true", help="verify rebuilt governed outputs")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPO_ROOT / "reference" / "KuaiRand-1K" / "data",
    )
    parser.add_argument(
        "--agent-root",
        type=Path,
        default=REPO_ROOT / "reference" / "kuairand-longseq-agent",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.raw and not args.silver:
        raise SystemExit("select --raw, --silver, or both")
    failures: list[str] = []
    if args.raw:
        failures.extend(verify_files(args.raw_root, MANIFEST_ROOT / "raw-files.json"))
    if args.silver:
        failures.extend(verify_files(args.agent_root, MANIFEST_ROOT / "silver-expected.json"))
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"verification failed: {len(failures)} problem(s)")
        return 1
    print("verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
