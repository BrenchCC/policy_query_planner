import re
import json
import logging
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)

PLANNER_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["queries"],
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 4,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "query", "depends_on"],
                "properties": {
                    "id": {"type": "string", "pattern": "^q[1-4]$"},
                    "query": {"type": "string", "minLength": 3},
                    "depends_on": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^q[1-4]$"}
                    }
                }
            }
        }
    }
}


def serialize_plan(queries: list[dict[str, Any]]) -> str:
    """Serialize a planner query list as compact JSON.

    Args:
        queries: Ordered query objects.

    Returns:
        Compact JSON planner response.
    """
    plan = {"queries": queries}
    validate_planner_plan(plan)
    return json.dumps(plan, ensure_ascii = False, separators = (",", ":"))


def validate_planner_plan(value: str | dict[str, Any]) -> dict[str, Any]:
    """Parse and validate a planner output and dependency graph.

    Args:
        value: JSON string or parsed planner object.

    Returns:
        Validated planner object.

    Raises:
        ValueError: If JSON, fields, ordering, or placeholders are invalid.
    """
    if isinstance(value, str):
        try:
            plan = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid planner JSON: {error}") from error
    else:
        plan = value
    try:
        jsonschema.validate(instance = plan, schema = PLANNER_PLAN_SCHEMA)
    except jsonschema.ValidationError as error:
        raise ValueError(f"Planner schema validation failed: {error.message}") from error

    seen_ids = set()
    for query_item in plan["queries"]:
        query_id = query_item["id"]
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query ID: {query_id}")
        dependencies = query_item["depends_on"]
        for dependency in dependencies:
            if dependency not in seen_ids:
                raise ValueError(f"Dependency {dependency} must refer to an earlier query")
            placeholder = "{{" + dependency + ".answer}}"
            if placeholder not in query_item["query"]:
                raise ValueError(f"Dependent query {query_id} must contain {placeholder}")
        placeholders = re.findall(r"\{\{(q[1-4])\.answer\}\}", query_item["query"])
        if set(placeholders) != set(dependencies):
            raise ValueError(f"Query {query_id} placeholders must match depends_on")
        seen_ids.add(query_id)
    return plan


def validate_sft_record(record: dict[str, Any]) -> None:
    """Validate one Alpaca SFT record.

    Args:
        record: SFT record to validate.

    Raises:
        ValueError: If required fields or planner output are invalid.
    """
    required_fields = ["id", "instruction", "input", "output", "system"]
    for field in required_fields:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"SFT field {field} must be a non-empty string")
    validate_planner_plan(record["output"])


def validate_dpo_record(record: dict[str, Any]) -> None:
    """Validate one LLaMA-Factory preference record.

    Args:
        record: DPO preference record to validate.

    Raises:
        ValueError: If fields or preference outputs are invalid.
    """
    required_fields = [
        "id",
        "instruction",
        "input",
        "chosen",
        "rejected",
        "system"
    ]
    for field in required_fields:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"DPO field {field} must be a non-empty string")
    if record["chosen"] == record["rejected"]:
        raise ValueError("DPO chosen and rejected outputs must differ")
    validate_planner_plan(record["chosen"])
    validate_planner_plan(record["rejected"])


def validate_grpo_record(record: dict[str, Any]) -> None:
    """Validate one GRPO-compatible record.

    Args:
        record: GRPO record to validate.

    Raises:
        ValueError: If prompt, plan, or reference metadata are invalid.
    """
    validate_sft_record(record)
    hop_count = record.get("hop_count")
    if hop_count not in {2, 3, 4}:
        raise ValueError("GRPO hop_count must be 2, 3, or 4")
    if not isinstance(record.get("reference_answer"), str) or not record["reference_answer"].strip():
        raise ValueError("GRPO reference_answer must be non-empty")
    if record.get("namespace") not in {"musique_aux", "policy"}:
        raise ValueError("GRPO namespace is invalid")
    hop_answers = record.get("hop_answers")
    gold_doc_ids = record.get("gold_doc_ids")
    if not isinstance(hop_answers, list):
        raise ValueError("GRPO hop_answers must be a list")
    if not isinstance(gold_doc_ids, list) or not gold_doc_ids:
        raise ValueError("GRPO gold_doc_ids must be a non-empty list")
    if record["namespace"] == "musique_aux":
        if len(hop_answers) != hop_count:
            raise ValueError("MuSiQue GRPO hop_answers must match hop_count")
        if len(gold_doc_ids) != hop_count:
            raise ValueError("MuSiQue GRPO gold_doc_ids must match hop_count")


def validate_knowledge_record(record: dict[str, Any]) -> None:
    """Validate one knowledge-base JSONL record.

    Args:
        record: Knowledge-base record to validate.

    Raises:
        ValueError: If required fields are missing or empty.
    """
    required_fields = [
        "id",
        "text",
        "title",
        "source",
        "source_dataset",
        "namespace",
        "content_hash"
    ]
    for field in required_fields:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Knowledge-base field {field} must be a non-empty string")
