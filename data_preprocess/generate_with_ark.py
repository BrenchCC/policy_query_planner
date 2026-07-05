import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.ark_llm_call import call_llm_on_volcengine
from data_preprocess.common import extract_json_object, normalized_key, read_jsonl
from data_preprocess.config import LLM_ENDPOINT, REQUEST_ROOT, RESPONSE_ROOT
from data_preprocess.schemas import validate_planner_plan

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Run optional Ark data generation")
    parser.add_argument("--stage", choices = ["sft", "dpo", "grpo"], required = True)
    parser.add_argument("--dry-run", action = "store_true", help = "Inspect requests without API calls")
    parser.add_argument("--resume", action = "store_true", help = "Skip completed request IDs")
    parser.add_argument("--limit", type = int, default = None, help = "Maximum requests to process")
    parser.add_argument(
        "--reasoning-option",
        default = None,
        help = "Optional Ark thinking mode"
    )
    return parser.parse_args()


def load_completed_ids(path: Path) -> set[str]:
    """Load request IDs already present in a response file.

    Args:
        path: Response JSONL path.

    Returns:
        Completed request IDs.
    """
    if not path.exists():
        return set()
    return {record["id"] for record in read_jsonl(path)}


def validate_generated_plan(
    request: dict[str, Any],
    response_text: str
) -> str:
    """Parse and validate one generated planner response.

    Args:
        request: Source generation request.
        response_text: Raw model response text.

    Returns:
        Compact serialized planner response.

    Raises:
        ValueError: If JSON, schema, preference, or leakage checks fail.
    """
    if response_text == "dummy_result":
        raise ValueError("Ark helper returned dummy_result")
    plan = extract_json_object(response_text)
    validate_planner_plan(plan)
    serialized = json.dumps(plan, ensure_ascii = False, separators = (",", ":"))
    if request["stage"] == "dpo" and serialized == request["payload"]["chosen"]:
        raise ValueError("Generated rejected plan equals the chosen plan")
    if request["stage"] == "grpo":
        answer = request.get("reference_answer", "")
        if len(normalized_key(answer)) > 4 and normalized_key(answer) in normalized_key(serialized):
            raise ValueError("Generated plan leaks the reference answer")
    return serialized


def build_response_record(
    request: dict[str, Any],
    reasoning: str | None,
    response_text: str,
    prompt_tokens: int | str,
    completion_tokens: int | str
) -> dict[str, Any]:
    """Build one validated or rejected API response record.

    Args:
        request: Source generation request.
        reasoning: Optional model reasoning content.
        response_text: Raw model response.
        prompt_tokens: Prompt token usage.
        completion_tokens: Completion token usage.

    Returns:
        Structured response record with validation status.
    """
    record = {
        "id": request["id"],
        "stage": request["stage"],
        "source_id": request["source_id"],
        "status": "success",
        "parsed_output": "",
        "raw_response": response_text,
        "reasoning_content": reasoning or "",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": ""
    }
    if "target_record_id" in request:
        record["target_record_id"] = request["target_record_id"]
    try:
        record["parsed_output"] = validate_generated_plan(request, response_text)
    except ValueError as error:
        record["status"] = "invalid"
        record["error"] = str(error)
    return record


def run_dry_run(stage: str, requests: list[dict[str, Any]], limit: int | None) -> None:
    """Log queue details without making an API call.

    Args:
        stage: Generation stage.
        requests: Prepared generation requests.
        limit: Optional request limit.
    """
    selected = requests[:limit] if limit is not None else requests
    logger.info("Dry run stage=%s requests=%d", stage, len(selected))
    if selected:
        logger.info("First request ID: %s", selected[0]["id"])
        logger.info("First prompt preview:\n%s", selected[0]["prompt"][:1200])


def main() -> None:
    """Run or inspect one optional Ark generation stage."""
    args = parse_args()
    request_path = REQUEST_ROOT / f"{args.stage}_requests.jsonl"
    if not request_path.exists():
        raise FileNotFoundError(f"Prepare requests first: {request_path}")
    requests = read_jsonl(request_path)
    if args.dry_run:
        run_dry_run(args.stage, requests, args.limit)
        return
    if not LLM_ENDPOINT:
        raise RuntimeError("Missing LLM_ENDPOINT in environment")

    response_path = RESPONSE_ROOT / f"{args.stage}_responses.jsonl"
    if response_path.exists() and not args.resume:
        raise FileExistsError("Response file exists; use --resume to continue safely")
    completed_ids = load_completed_ids(response_path) if args.resume else set()
    pending = [request for request in requests if request["id"] not in completed_ids]
    if args.limit is not None:
        pending = pending[:args.limit]
    response_path.parent.mkdir(parents = True, exist_ok = True)
    with response_path.open("a", encoding = "utf-8") as file:
        for request in tqdm(pending, desc = f"Generating {args.stage}"):
            reasoning, result, prompt_tokens, completion_tokens = call_llm_on_volcengine(
                request["prompt"],
                LLM_ENDPOINT,
                system_prompt = request["system"],
                stream = False,
                reasoning_option = args.reasoning_option
            )
            response_record = build_response_record(
                request,
                reasoning,
                result,
                prompt_tokens,
                completion_tokens
            )
            file.write(json.dumps(response_record, ensure_ascii = False) + "\n")
            file.flush()


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()

