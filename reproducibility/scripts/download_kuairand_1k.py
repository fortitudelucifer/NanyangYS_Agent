from __future__ import annotations

import argparse
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_URL = "https://chongming.myds.me:61364/data/KuaiRand-1K.tar.gz"


def download(url: str, destination: Path, allow_insecure_tls: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "NanyangYS-repro/1"})
    context = ssl._create_unverified_context() if allow_insecure_tls else None
    try:
        with urllib.request.urlopen(request, context=context) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise RuntimeError(f"archive links are not accepted: {member.name}")
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination_resolved):
                raise RuntimeError(f"unsafe archive path: {member.name}")
        bundle.extractall(destination, members=members)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify official KuaiRand-1K data.")
    parser.add_argument("--url", default=OFFICIAL_URL)
    parser.add_argument("--archive", type=Path, help="use a previously downloaded archive")
    parser.add_argument(
        "--destination",
        type=Path,
        default=REPO_ROOT / "reference",
        help="parent directory that will contain KuaiRand-1K",
    )
    parser.add_argument(
        "--allow-insecure-tls",
        action="store_true",
        help="explicitly disable TLS certificate verification for the historical official host",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.destination / "KuaiRand-1K" / "data"
    verifier = REPO_ROOT / "reproducibility" / "scripts" / "verify_reproduction.py"
    if data_root.is_dir():
        print(f"existing data directory found; verifying without overwriting: {data_root}")
        return subprocess.call([sys.executable, str(verifier), "--raw", "--raw-root", str(data_root)])

    archive = args.archive
    if archive is None:
        archive = REPO_ROOT / ".cache" / "kuairand" / "KuaiRand-1K.tar.gz"
        if not archive.is_file():
            print(f"downloading {args.url} -> {archive}")
            download(args.url, archive, args.allow_insecure_tls)
    archive = archive.resolve()
    if not archive.is_file():
        raise SystemExit(f"archive not found: {archive}")

    print(f"extracting {archive} -> {args.destination}")
    safe_extract(archive, args.destination)
    return subprocess.call([sys.executable, str(verifier), "--raw", "--raw-root", str(data_root)])


if __name__ == "__main__":
    raise SystemExit(main())
