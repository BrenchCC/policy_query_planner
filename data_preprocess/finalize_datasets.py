import os
import sys
import json
import logging
import argparse
from collections import Counter
from typing import Any

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import read_jsonl, write_json, write_jsonl
from data_preprocess.config import INTERIM_ROOT, PROCESSED_ROOT, RESPONSE_ROOT, PLANNER_SYSTEM_PROMPT
from data_preprocess.schemas import validate_sft_record, validate_dpo_record, validate_grpo_record

logger = logging.getLogger(__name__)

STAGES = ["sft", "dpo", "grpo"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Merge validated API outputs into datasets")
    parser.add_argument("--stage", choices = STAGES + ["all"], required = True)
    return parser.parse_args()


def successful_responses(stage: str) -> list[dict[str, Any]]:
    """Load successful responses for one generation stage.

    Args:
        stage: Generation stage name.

    Returns:
        Successful response records, or an empty list when absent.
    """
    path = RESPONSE_ROOT / f"{stage}_responses.jsonl"
    if not path.exists():
        return []
    return [record for record in read_jsonl(path) if record["status"] == "success"]


def finalize_sft() -> dict[str, int]:
    """Replace local policy SFT plans with validated API plans.

    Returns:
        Merge statistics.
    """
    responses = successful_responses("sft")
    replacements = {record["source_id"]: record["parsed_output"] for record in responses}
    records = read_jsonl(PROCESSED_ROOT / "train" / "sft_train.jsonl")
    replaced = 0
    for record in records:
        if record["source_dataset"] == "conditionalqa" and record["source_id"] in replacements:
            record["output"] = replacements[record["source_id"]]
            validate_sft_record(record)
            replaced += 1
    write_jsonl(PROCESSED_ROOT / "train" / "sft_train_api_augmented.jsonl", records)
    return {"available": len(responses), "replaced": replaced, "total": len(records)}


def finalize_dpo() -> dict[str, int]:
    """Replace local policy DPO negatives with validated API negatives.

    Returns:
        Merge statistics.
    """
    responses = successful_responses("dpo")
    replacements = {
        record["target_record_id"]: record["parsed_output"]
        for record in responses
        if "target_record_id" in record
    }
    records = read_jsonl(PROCESSED_ROOT / "train" / "dpo_train.jsonl")
    replaced = 0
    for record in records:
        if record["id"] in replacements:
            record["rejected"] = replacements[record["id"]]
            validate_dpo_record(record)
            replaced += 1
    write_jsonl(PROCESSED_ROOT / "train" / "dpo_train_api_augmented.jsonl", records)
    return {"available": len(responses), "replaced": replaced, "total": len(records)}


def flatten_answers(answers: list[Any]) -> str:
    """Flatten ConditionalQA answers for GRPO metadata.

    Args:
        answers: Raw answer structures.

    Returns:
        Semicolon-separated reference answer.
    """
    values = []
    for answer in answers:
        if isinstance(answer, list) and answer and answer[0] not in values:
            values.append(str(answer[0]))
    return "; ".join(values) or "not answerable"


def finalize_grpo() -> dict[str, int]:
    """Create the 4K MuSiQue plus 1K policy GRPO mixture.

    Returns:
        Merge statistics.

    Raises:
        ValueError: If fewer than 1,000 valid policy plans are available.
    """
    responses = sorted(successful_responses("grpo"), key = lambda record: record["source_id"])
    if len(responses) < 1000:
        raise ValueError(f"GRPO finalization requires 1000 valid responses; found {len(responses)}")
    selected_responses = responses[:1000]
    conditional_records = {
        record["id"]: record
        for record in read_jsonl(INTERIM_ROOT / "conditionalqa_train.jsonl")
    }
    policy_records = []
    for response in selected_responses:
        source = conditional_records[response["source_id"]]
        plan = json.loads(response["parsed_output"])
        record = {
            "id": "grpo_policy_" + source["id"],
            "instruction": (
                "Decompose the multi-hop policy question into ordered retrieval queries. "
                "Return a JSON query plan only and do not answer the question."
            ),
            "input": f"Scenario:\n{source['scenario']}\n\nQuestion:\n{source['question']}",
            "output": response["parsed_output"],
            "system": PLANNER_SYSTEM_PROMPT,
            "source_dataset": "conditionalqa",
            "source_id": source["id"],
            "namespace": "policy",
            "hop_count": len(plan["queries"]),
            "reference_answer": flatten_answers(source["answers"]),
            "answer_aliases": [],
            "hop_answers": [],
            "gold_doc_ids": source["gold_doc_ids"]
        }
        validate_grpo_record(record)
        policy_records.append(record)

    baseline = read_jsonl(PROCESSED_ROOT / "train" / "grpo_train.jsonl")
    mixed_records = baseline[:4000] + policy_records
    write_jsonl(
        PROCESSED_ROOT / "train" / "grpo_train_domain_augmented.jsonl",
        mixed_records
    )
    return {
        "available": len(responses),
        "policy_records": len(policy_records),
        "total": len(mixed_records)
    }


def main() -> None:
    """Finalize selected API-enhanced datasets."""
    args = parse_args()
    stages = STAGES if args.stage == "all" else [args.stage]
    functions = {"sft": finalize_sft, "dpo": finalize_dpo, "grpo": finalize_grpo}
    summary = {}
    for stage in stages:
        try:
            summary[stage] = functions[stage]()
        except ValueError as error:
            summary[stage] = {"status": "not_ready", "error": str(error)}
            logger.warning("Skipping %s finalization: %s", stage, error)
    summary["output_counts"] = {
        key: value
        for key, value in Counter(
            path.name
            for path in (PROCESSED_ROOT / "train").glob("*_api_augmented.jsonl")
        ).items()
    }
    write_json(INTERIM_ROOT / "api_finalization_summary.json", summary)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()

