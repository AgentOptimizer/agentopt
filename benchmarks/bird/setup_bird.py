#!/usr/bin/env python3
"""Set up the BIRD dev benchmark data for AgentOpt examples.

The official dev archive is large. This script keeps it outside git under
``benchmarks/bird/data`` by default.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
DEFAULT_TARGET = Path("benchmarks/bird/data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-cortex",
        action="store_true",
        help=(
            "Copy from cortex/lg2sql/lg/data/bird/data when it exists locally. "
            "This does not modify cortex."
        ),
    )
    source.add_argument(
        "--source-dir",
        type=Path,
        help="Copy from an existing BIRD dev data directory.",
    )
    source.add_argument(
        "--zip",
        type=Path,
        help="Set up from a local BIRD dev.zip archive instead of downloading.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_BIRD_DEV_URL,
        help=f"Official BIRD dev archive URL. Default: {DEFAULT_BIRD_DEV_URL}",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Target data directory. Default: {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing target directory.",
    )
    return parser.parse_args()


def validate_bird_dir(path: Path) -> Path:
    """Return the normalized directory that contains dev.json."""

    candidates = [
        path,
        path / "data",
        path / "dev_20240627",
        path / "lg" / "data" / "bird" / "data",
    ]
    for candidate in candidates:
        if (
            (candidate / "dev.json").exists()
            and (candidate / "dev_tables.json").exists()
            and (candidate / "dev_databases").exists()
        ):
            return candidate
    raise FileNotFoundError(
        f"Could not find BIRD dev files under {path}. Expected dev.json, "
        "dev_tables.json, and dev_databases/."
    )


def ensure_target_ready(target: Path, force: bool) -> None:
    if target.exists() and any(target.iterdir()):
        if not force:
            raise FileExistsError(
                f"Target already has files: {target}. Pass --force to replace it."
            )
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def copy_bird_dir(source: Path, target: Path, force: bool) -> None:
    source = validate_bird_dir(source)
    ensure_target_ready(target, force=force)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".DS_Store", "__MACOSX", "*.zip"),
    )


def safe_extract(zip_file: zipfile.ZipFile, destination: Path) -> None:
    """Extract a zip file while rejecting path traversal entries."""

    destination = destination.resolve()
    for member in zip_file.infolist():
        member_path = (destination / member.filename).resolve()
        if destination not in member_path.parents and member_path != destination:
            raise ValueError(f"Refusing unsafe zip entry: {member.filename}")
    zip_file.extractall(destination)


def unpack_zip(zip_path: Path, target: Path, force: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="agentopt-bird-") as tmp_name:
        tmp_dir = Path(tmp_name)
        with zipfile.ZipFile(zip_path) as archive:
            safe_extract(archive, tmp_dir)

        extracted = validate_extracted_dir(tmp_dir)
        nested_db_zip = extracted / "dev_databases.zip"
        if nested_db_zip.exists() and not (extracted / "dev_databases").exists():
            with zipfile.ZipFile(nested_db_zip) as archive:
                safe_extract(archive, extracted)

        copy_bird_dir(extracted, target, force=force)


def validate_extracted_dir(path: Path) -> Path:
    candidates = [path, path / "dev_20240627"]
    for child in path.iterdir():
        if child.is_dir():
            candidates.append(child)
    for candidate in candidates:
        if (candidate / "dev.json").exists() and (candidate / "dev_tables.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find dev.json and dev_tables.json in extracted archive: {path}"
    )


def download_archive(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def setup_from_download(url: str, target: Path, force: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="agentopt-bird-download-") as tmp_name:
        zip_path = Path(tmp_name) / "dev.zip"
        download_archive(url, zip_path)
        unpack_zip(zip_path, target, force=force)


def main() -> int:
    args = parse_args()
    target = args.target.expanduser()

    try:
        if args.from_cortex:
            copy_bird_dir(Path("cortex/lg2sql/lg/data/bird/data"), target, args.force)
        elif args.source_dir:
            copy_bird_dir(args.source_dir.expanduser(), target, args.force)
        elif args.zip:
            unpack_zip(args.zip.expanduser(), target, args.force)
        else:
            setup_from_download(args.url, target, args.force)
    except Exception as exc:
        print(f"Failed to set up BIRD data: {exc}", file=sys.stderr)
        return 1

    normalized = validate_bird_dir(target)
    print(f"BIRD dev data is ready at: {normalized}")
    print(f"Questions: {normalized / 'dev.json'}")
    print(f"Tables:    {normalized / 'dev_tables.json'}")
    print(f"DBs:       {normalized / 'dev_databases'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

