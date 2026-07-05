import os
import sys
import json
import shutil
import zipfile
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

import gdown

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import download_file, ensure_parent, sha256_file, write_json
from data_preprocess.config import (
    RAW_ROOT,
    PROJECT_ROOT,
    ARK_RAW_ROOT,
    QRECC_REVISION,
    MUSIQUE_RAW_ROOT,
    MUSIQUE_REVISION,
    QRECC_RAW_ROOT,
    QRECC_ARCHIVE_NAME,
    MUSIQUE_DRIVE_ID,
    MUSIQUE_ARCHIVE_NAME,
    CONDITIONALQA_FILES,
    CONDITIONALQA_RAW_ROOT,
    CONDITIONALQA_REVISION,
    ARK_SCRIPT_REVISION,
    QRECC_ARCHIVE_SHA256
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Download pinned project datasets")
    parser.add_argument("--force", action = "store_true", help = "Replace existing downloads")
    return parser.parse_args()


def downloaded_at(path: Path) -> str:
    """Return the source file modification time in UTC.

    Args:
        path: Downloaded source file.

    Returns:
        ISO-8601 UTC timestamp.
    """
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def extract_archive(archive_path: Path, destination: Path, force: bool) -> None:
    """Extract a ZIP archive into a stable directory.

    Args:
        archive_path: Source ZIP archive.
        destination: Extraction directory.
        force: Whether to replace an existing extraction.
    """
    marker = destination / ".complete"
    if marker.exists() and not force:
        logger.info("Skipping extracted archive: %s", destination)
        return
    if destination.exists() and force:
        shutil.rmtree(destination)
    destination.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)
    marker.touch()


def download_conditionalqa(force: bool) -> list[dict[str, object]]:
    """Download the pinned ConditionalQA files.

    Args:
        force: Whether to replace existing files.

    Returns:
        Download manifest entries.
    """
    entries = []
    for file_name in CONDITIONALQA_FILES:
        url = (
            "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/"
            f"{CONDITIONALQA_REVISION}/v1_0/{file_name}"
        )
        path = CONDITIONALQA_RAW_ROOT / file_name
        download_file(url, path, force = force)
        entries.append(
            {
                "dataset": "conditionalqa",
                "file": str(path.relative_to(PROJECT_ROOT)),
                "url": url,
                "revision": CONDITIONALQA_REVISION,
                "license": "CC BY-SA 4.0",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "downloaded_at": downloaded_at(path)
            }
        )
    return entries


def download_qrecc(force: bool) -> list[dict[str, object]]:
    """Download and extract the pinned QReCC rewrite archive.

    Args:
        force: Whether to replace existing files.

    Returns:
        Download manifest entries.
    """
    url = (
        "https://raw.githubusercontent.com/apple/ml-qrecc/"
        f"{QRECC_REVISION}/dataset/{QRECC_ARCHIVE_NAME}"
    )
    archive_path = QRECC_RAW_ROOT / "archives" / QRECC_ARCHIVE_NAME
    download_file(url, archive_path, force = force)
    checksum = sha256_file(archive_path)
    if checksum != QRECC_ARCHIVE_SHA256:
        raise ValueError(f"QReCC archive checksum mismatch: {checksum}")
    extract_archive(archive_path, QRECC_RAW_ROOT / "original", force = force)
    return [
        {
            "dataset": "qrecc",
            "file": str(archive_path.relative_to(PROJECT_ROOT)),
            "url": url,
            "revision": QRECC_REVISION,
            "license": "CC BY-SA 3.0",
            "size_bytes": archive_path.stat().st_size,
            "sha256": checksum,
            "downloaded_at": downloaded_at(archive_path)
        }
    ]


def download_musique(force: bool) -> list[dict[str, object]]:
    """Download and extract the official MuSiQue archive.

    Args:
        force: Whether to replace existing files.

    Returns:
        Download manifest entries.
    """
    archive_path = MUSIQUE_RAW_ROOT / "archives" / MUSIQUE_ARCHIVE_NAME
    if force and archive_path.exists():
        archive_path.unlink()
    if not archive_path.exists():
        ensure_parent(archive_path)
        result = gdown.download(id = MUSIQUE_DRIVE_ID, output = str(archive_path), quiet = False)
        if result is None or not archive_path.exists():
            raise RuntimeError("MuSiQue download failed")
    extract_archive(archive_path, MUSIQUE_RAW_ROOT / "original", force = force)
    return [
        {
            "dataset": "musique",
            "file": str(archive_path.relative_to(PROJECT_ROOT)),
            "url": f"https://drive.google.com/file/d/{MUSIQUE_DRIVE_ID}/view",
            "revision": MUSIQUE_REVISION,
            "license": "CC BY 4.0",
            "size_bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "downloaded_at": downloaded_at(archive_path)
        }
    ]


def download_ark_helper(force: bool) -> list[dict[str, object]]:
    """Download the pinned upstream Ark helper for provenance.

    Args:
        force: Whether to replace the existing file.

    Returns:
        Download manifest entries.
    """
    url = (
        "https://raw.githubusercontent.com/BrenchCC/OpenAI_API_utils/"
        f"{ARK_SCRIPT_REVISION}/ark_llm_call.py"
    )
    path = ARK_RAW_ROOT / "ark_llm_call.py"
    download_file(url, path, force = force)
    return [
        {
            "dataset": "ark_helper",
            "file": str(path.relative_to(PROJECT_ROOT)),
            "url": url,
            "revision": ARK_SCRIPT_REVISION,
            "license": "Not declared upstream",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "downloaded_at": downloaded_at(path)
        }
    ]


def main() -> None:
    """Download all source datasets and write a reproducibility manifest."""
    args = parse_args()
    logger.info("=" * 80)
    logger.info("Downloading pinned source datasets")
    logger.info("=" * 80)
    entries = []
    entries.extend(download_conditionalqa(args.force))
    entries.extend(download_qrecc(args.force))
    entries.extend(download_musique(args.force))
    entries.extend(download_ark_helper(args.force))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "excluded_sources": ["QReCC 27.5 GB passage archive"],
        "files": entries
    }
    write_json(RAW_ROOT / "download_manifest.json", manifest)
    logger.info("Downloaded %d source artifacts", len(entries))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
