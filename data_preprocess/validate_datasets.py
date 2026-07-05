import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Any, Callable

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import read_json, read_jsonl, sha256_file, normalized_key
from data_preprocess.config import (
    RAW_ROOT,
    INTERIM_ROOT,
    PROCESSED_ROOT,
    REQUEST_ROOT,
    TARGET_COUNTS,
    EXPECTED_QRECC_COUNTS,
    QRECC_ARCHIVE_SHA256,
    EXPECTED_CONDITIONALQA_COUNTS,
    EVIDENCE_MAPPING_THRESHOLD
)
from data_preprocess.schemas import (
    validate_sft_record,
    validate_dpo_record,
    validate_grpo_record,
    validate_knowledge_record
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Validate all project data artifacts")
    parser.add_argument(
        "--stage",
        choices = ["raw", "base", "train", "api", "all"],
        default = "all"
    )
    return parser.parse_args()


def assert_unique_ids(records: list[dict[str, Any]], label: str) -> None:
    """Assert that record IDs are unique.

    Args:
        records: Records containing an id field.
        label: Human-readable dataset label.

    Raises:
        ValueError: If any duplicate ID exists.
    """
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs detected in {label}")


def validate_raw() -> None:
    """Validate raw downloads, checksums, and source counts."""
    manifest = read_json(RAW_ROOT / "download_manifest.json")
    for entry in manifest["files"]:
        path = Path(__file__).resolve().parents[1] / entry["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
    qrecc_archive = RAW_ROOT / "qrecc" / "archives" / "qrecc_data.zip"
    if sha256_file(qrecc_archive) != QRECC_ARCHIVE_SHA256:
        raise ValueError("QReCC pinned checksum mismatch")

    documents = read_json(RAW_ROOT / "conditionalqa" / "documents.json")
    if len(documents) != EXPECTED_CONDITIONALQA_COUNTS["documents"]:
        raise ValueError("Unexpected ConditionalQA document count")
    for split in ["train", "dev", "test_no_answer"]:
        records = read_json(RAW_ROOT / "conditionalqa" / f"{split}.json")
        if len(records) != EXPECTED_CONDITIONALQA_COUNTS[split]:
            raise ValueError(f"Unexpected ConditionalQA {split} count")
    for split in ["train", "test"]:
        records = read_json(RAW_ROOT / "qrecc" / "original" / f"qrecc_{split}.json")
        if len(records) != EXPECTED_QRECC_COUNTS[split]:
            raise ValueError(f"Unexpected QReCC {split} count")
    logger.info("Raw source validation passed")


def validate_base() -> None:
    """Validate knowledge bases, cleaning coverage, and official benchmarks."""
    for file_name in ["policy.jsonl", "musique_aux.jsonl"]:
        records = read_jsonl(PROCESSED_ROOT / "knowledge_base" / file_name)
        assert_unique_ids(records, file_name)
        for record in records:
            validate_knowledge_record(record)
    summary = read_json(INTERIM_ROOT / "cleaning_summary.json")
    mapping_rate = summary["conditionalqa"]["evidence_mapping_rate"]
    if mapping_rate < EVIDENCE_MAPPING_THRESHOLD:
        raise ValueError(f"Evidence mapping rate below threshold: {mapping_rate:.4f}")

    expected_eval_counts = {
        "conditionalqa_dev.jsonl": 285,
        "conditionalqa_test_blind.jsonl": 804,
        "qrecc_test.jsonl": 16451,
        "musique_dev.jsonl": 2417
    }
    for file_name, expected_count in expected_eval_counts.items():
        records = read_jsonl(PROCESSED_ROOT / "eval" / file_name)
        if len(records) != expected_count:
            raise ValueError(f"Unexpected benchmark count for {file_name}")
        assert_unique_ids(records, file_name)
    logger.info("Knowledge-base and benchmark validation passed")


def validate_records(
    path: Path,
    expected_count: int,
    validator: Callable[[dict[str, Any]], None]
) -> list[dict[str, Any]]:
    """Validate count, IDs, and schema for one JSONL dataset.

    Args:
        path: Dataset JSONL path.
        expected_count: Exact expected record count.
        validator: Record-level validation function.

    Returns:
        Validated records.
    """
    records = read_jsonl(path)
    if len(records) != expected_count:
        raise ValueError(f"Unexpected count for {path}: {len(records)}")
    assert_unique_ids(records, path.name)
    for record in records:
        validator(record)
        if "reward" in record or "reward_weights" in record:
            raise ValueError(f"Reward design leaked into data: {record['id']}")
    return records


def validate_train() -> None:
    """Validate train schemas, exact counts, distributions, and split leakage."""
    sft_records = validate_records(
        PROCESSED_ROOT / "train" / "sft_train.jsonl",
        TARGET_COUNTS["sft"],
        validate_sft_record
    )
    dpo_records = validate_records(
        PROCESSED_ROOT / "train" / "dpo_train.jsonl",
        TARGET_COUNTS["dpo"],
        validate_dpo_record
    )
    grpo_records = validate_records(
        PROCESSED_ROOT / "train" / "grpo_train.jsonl",
        TARGET_COUNTS["grpo"],
        validate_grpo_record
    )

    held_out_ids = set()
    for file_name in [
        "conditionalqa_dev.jsonl",
        "conditionalqa_test_blind.jsonl",
        "qrecc_test.jsonl",
        "musique_dev.jsonl"
    ]:
        held_out_ids.update(
            record["id"]
            for record in read_jsonl(PROCESSED_ROOT / "eval" / file_name)
        )
    for record in sft_records + dpo_records + grpo_records:
        if record["source_id"] in held_out_ids:
            raise ValueError(f"Benchmark leakage detected: {record['id']}")

    qrecc_train = {
        record["id"]: record
        for record in read_jsonl(INTERIM_ROOT / "qrecc_train.jsonl")
    }
    qrecc_test_keys = {
        (
            normalized_key(record["question"]),
            normalized_key(record["rewrite"]),
            normalized_key(record["answer"])
        )
        for record in read_jsonl(INTERIM_ROOT / "qrecc_test.jsonl")
    }
    for record in sft_records + dpo_records:
        if record["source_dataset"] != "qrecc":
            continue
        source = qrecc_train[record["source_id"]]
        content_key = (
            normalized_key(source["question"]),
            normalized_key(source["rewrite"]),
            normalized_key(source["answer"])
        )
        if content_key in qrecc_test_keys:
            raise ValueError(f"QReCC content leakage detected: {record['id']}")

    error_counts = {}
    for record in dpo_records:
        error_counts[record["error_type"]] = error_counts.get(record["error_type"], 0) + 1
    if set(error_counts.values()) != {1000}:
        raise ValueError(f"DPO error categories are not balanced: {error_counts}")
    if not (PROCESSED_ROOT / "dataset_info.json").exists():
        raise FileNotFoundError("Missing LLaMA-Factory dataset_info.json")
    logger.info("Training data validation passed")


def validate_api() -> None:
    """Validate optional generation request queues."""
    expected_counts = {"sft": 2338, "dpo": 1000, "grpo": 1811}
    for stage, expected_count in expected_counts.items():
        records = read_jsonl(REQUEST_ROOT / f"{stage}_requests.jsonl")
        if len(records) != expected_count:
            raise ValueError(f"Unexpected {stage} request count: {len(records)}")
        assert_unique_ids(records, f"{stage} requests")
        for record in records:
            if not record["prompt"].strip() or not record["system"].strip():
                raise ValueError(f"Empty API prompt: {record['id']}")
    logger.info("Optional API request validation passed")


def main() -> None:
    """Run selected validation stages."""
    args = parse_args()
    functions = {
        "raw": validate_raw,
        "base": validate_base,
        "train": validate_train,
        "api": validate_api
    }
    stages = list(functions) if args.stage == "all" else [args.stage]
    for stage in stages:
        logger.info("-" * 60)
        logger.info("Validating stage: %s", stage)
        logger.info("-" * 60)
        functions[stage]()
    logger.info("All selected validations passed")


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
