import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable

from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.getcwd())

from data_preprocess.common import (
    read_json,
    write_json,
    write_jsonl,
    stable_hash,
    strip_html,
    normalize_text,
    normalized_key
)
from data_preprocess.config import (
    INTERIM_ROOT,
    PROCESSED_ROOT,
    QRECC_RAW_ROOT,
    MUSIQUE_RAW_ROOT,
    MAX_POLICY_CHUNK_CHARS,
    CONDITIONALQA_RAW_ROOT,
    MIN_POLICY_CHUNK_CHARS
)

logger = logging.getLogger(__name__)

HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description = "Clean all source datasets")
    parser.add_argument("--force", action = "store_true", help = "Replace processed outputs")
    return parser.parse_args()


def update_heading_path(path: list[str], tag: str, text: str) -> list[str]:
    """Update a hierarchical heading path.

    Args:
        path: Existing heading hierarchy.
        tag: Current HTML heading tag.
        text: Visible heading text.

    Returns:
        Updated heading hierarchy.
    """
    level = HEADING_LEVELS[tag]
    updated = path[:level - 1]
    updated.append(text)
    return updated


def flush_policy_chunk(
    chunks: list[dict[str, Any]],
    document: dict[str, Any],
    section_path: list[str],
    blocks: list[dict[str, Any]],
    split_membership: list[str]
) -> None:
    """Create one policy chunk from accumulated content blocks.

    Args:
        chunks: Destination chunk list.
        document: Source ConditionalQA document.
        section_path: Current heading hierarchy.
        blocks: Content blocks included in the chunk.
        split_membership: Official splits referencing the document.
    """
    if not blocks:
        return
    body = "\n".join(block["text"] for block in blocks)
    heading = " > ".join(section_path)
    text = f"{heading}\n{body}" if heading else body
    text = normalize_text(text)
    if not text:
        return
    content_hash = stable_hash(text, length = 64)
    chunk_id = "policy_" + stable_hash(document["url"], heading, text)
    chunks.append(
        {
            "id": chunk_id,
            "text": text,
            "title": normalize_text(document["title"]),
            "source": document["url"],
            "source_dataset": "conditionalqa",
            "namespace": "policy",
            "url": document["url"],
            "section_path": section_path,
            "source_indices": [block["index"] for block in blocks],
            "split_membership": split_membership,
            "content_hash": content_hash
        }
    )


def chunk_policy_document(
    document: dict[str, Any],
    split_membership: list[str]
) -> list[dict[str, Any]]:
    """Convert one ConditionalQA document into heading-aware chunks.

    Args:
        document: Raw ConditionalQA document.
        split_membership: Official splits referencing the document.

    Returns:
        Clean knowledge-base chunks.
    """
    chunks = []
    section_path: list[str] = []
    pending_blocks: list[dict[str, Any]] = []
    seen_blocks = set()
    pending_length = 0

    for index, fragment in enumerate(document.get("contents", [])):
        tag, text = strip_html(fragment)
        if not text:
            continue
        if tag in HEADING_LEVELS:
            flush_policy_chunk(chunks, document, section_path, pending_blocks, split_membership)
            pending_blocks = []
            pending_length = 0
            section_path = update_heading_path(section_path, tag, text)
            continue
        block_key = normalized_key(text)
        if not block_key or block_key in seen_blocks:
            continue
        seen_blocks.add(block_key)
        projected_length = pending_length + len(text) + 1
        if pending_blocks and projected_length > MAX_POLICY_CHUNK_CHARS:
            flush_policy_chunk(chunks, document, section_path, pending_blocks, split_membership)
            pending_blocks = []
            pending_length = 0
        pending_blocks.append({"index": index, "tag": tag, "text": text})
        pending_length += len(text) + 1

    flush_policy_chunk(chunks, document, section_path, pending_blocks, split_membership)
    if len(chunks) > 1 and len(chunks[-1]["text"]) < MIN_POLICY_CHUNK_CHARS:
        last = chunks.pop()
        previous = chunks[-1]
        previous["text"] = normalize_text(previous["text"] + "\n" + last["text"])
        previous["source_indices"].extend(last["source_indices"])
        previous["content_hash"] = stable_hash(previous["text"], length = 64)
    return chunks


def best_evidence_chunk(
    evidence: str,
    candidates: list[dict[str, Any]]
) -> tuple[str | None, float]:
    """Find the best chunk for one evidence string.

    Args:
        evidence: Gold evidence text.
        candidates: Chunks from the gold source document.

    Returns:
        Best chunk ID and similarity score.
    """
    _, evidence_text = strip_html(evidence)
    evidence_key = normalized_key(evidence_text)
    if not evidence_key:
        return None, 0.0
    best_id = None
    best_score = 0.0
    for candidate in candidates:
        candidate_key = normalized_key(candidate["text"])
        if evidence_key in candidate_key or candidate_key in evidence_key:
            containment = min(len(evidence_key), len(candidate_key)) / max(len(evidence_key), 1)
            score = 1.0 + containment
        else:
            score = SequenceMatcher(None, evidence_key, candidate_key).ratio()
        if score > best_score:
            best_id = candidate["id"]
            best_score = score
    if best_score < 0.45:
        return None, best_score
    return best_id, best_score


def clean_conditionalqa() -> dict[str, Any]:
    """Clean ConditionalQA documents and examples.

    Returns:
        Cleaning statistics and unresolved evidence details.
    """
    documents = read_json(CONDITIONALQA_RAW_ROOT / "documents.json")
    split_records = {
        "train": read_json(CONDITIONALQA_RAW_ROOT / "train.json"),
        "dev": read_json(CONDITIONALQA_RAW_ROOT / "dev.json"),
        "test_no_answer": read_json(CONDITIONALQA_RAW_ROOT / "test_no_answer.json")
    }
    url_memberships: dict[str, set[str]] = defaultdict(set)
    for split, records in split_records.items():
        for record in records:
            url_memberships[record["url"]].add(split)

    chunks = []
    for document in tqdm(documents, desc = "Cleaning ConditionalQA documents"):
        memberships = sorted(url_memberships.get(document["url"], set()))
        chunks.extend(chunk_policy_document(document, memberships))
    write_jsonl(PROCESSED_ROOT / "knowledge_base" / "policy.jsonl", chunks)

    chunks_by_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_url[chunk["url"]].append(chunk)
    titles_by_url = {document["url"]: normalize_text(document["title"]) for document in documents}

    unresolved = []
    evidence_total = 0
    evidence_mapped = 0
    for split, records in split_records.items():
        cleaned_records = []
        for raw_record in tqdm(records, desc = f"Mapping ConditionalQA {split}"):
            gold_doc_ids = []
            unresolved_evidence = []
            for evidence in raw_record.get("evidences", []):
                if not normalize_text(evidence):
                    continue
                evidence_total += 1
                chunk_id, score = best_evidence_chunk(evidence, chunks_by_url[raw_record["url"]])
                if chunk_id:
                    evidence_mapped += 1
                    if chunk_id not in gold_doc_ids:
                        gold_doc_ids.append(chunk_id)
                else:
                    issue = {
                        "id": raw_record["id"],
                        "split": split,
                        "url": raw_record["url"],
                        "evidence": normalize_text(evidence),
                        "best_score": score
                    }
                    unresolved.append(issue)
                    unresolved_evidence.append(issue)
            cleaned_records.append(
                {
                    "id": raw_record["id"],
                    "split": split,
                    "url": raw_record["url"],
                    "title": titles_by_url.get(raw_record["url"], "Unknown policy"),
                    "scenario": normalize_text(raw_record.get("scenario", "")),
                    "question": normalize_text(raw_record.get("question", "")),
                    "not_answerable": bool(raw_record.get("not_answerable", False)),
                    "answers": raw_record.get("answers", []),
                    "evidences": [
                        strip_html(value)[1]
                        for value in raw_record.get("evidences", [])
                        if strip_html(value)[1]
                    ],
                    "gold_doc_ids": gold_doc_ids,
                    "unresolved_evidence": unresolved_evidence
                }
            )
        write_jsonl(INTERIM_ROOT / f"conditionalqa_{split}.jsonl", cleaned_records)

    referenced_urls = set(url_memberships)
    all_urls = {document["url"] for document in documents}
    return {
        "raw_document_count": len(documents),
        "policy_chunk_count": len(chunks),
        "referenced_document_count": len(referenced_urls),
        "unreferenced_document_count": len(all_urls - referenced_urls),
        "unreferenced_urls": sorted(all_urls - referenced_urls),
        "evidence_total": evidence_total,
        "evidence_mapped": evidence_mapped,
        "evidence_mapping_rate": evidence_mapped / evidence_total if evidence_total else 1.0,
        "unresolved_evidence": unresolved
    }


def format_qrecc_history(context: list[str]) -> str:
    """Format an alternating QReCC context as dialogue text.

    Args:
        context: Alternating question and answer strings.

    Returns:
        Human-readable conversation history.
    """
    lines = []
    for index, text in enumerate(context):
        role = "User" if index % 2 == 0 else "Assistant"
        lines.append(f"{role}: {normalize_text(text)}")
    return "\n".join(lines)


def clean_qrecc() -> dict[str, Any]:
    """Clean QReCC train and official test records.

    Returns:
        QReCC cleaning statistics.
    """
    summary: dict[str, Any] = {}
    for split in ["train", "test"]:
        raw_path = QRECC_RAW_ROOT / "original" / f"qrecc_{split}.json"
        records = read_json(raw_path)
        cleaned_records = []
        seen_ids = set()
        duplicate_count = 0
        nontrivial_count = 0
        for raw_record in tqdm(records, desc = f"Cleaning QReCC {split}"):
            record_id = (
                f"qrecc_{split}_{raw_record['Conversation_no']}_{raw_record['Turn_no']}"
            )
            if record_id in seen_ids:
                duplicate_count += 1
                continue
            seen_ids.add(record_id)
            question = normalize_text(raw_record["Question"])
            rewrite = normalize_text(raw_record["Rewrite"])
            nontrivial = normalized_key(question) != normalized_key(rewrite)
            nontrivial_count += int(nontrivial)
            context = [normalize_text(value) for value in raw_record.get("Context", [])]
            cleaned_records.append(
                {
                    "id": record_id,
                    "split": split,
                    "conversation_no": int(raw_record["Conversation_no"]),
                    "turn_no": int(raw_record["Turn_no"]),
                    "conversation_source": raw_record["Conversation_source"],
                    "context": context,
                    "history_text": format_qrecc_history(context),
                    "context_turns": len(context) // 2,
                    "question": question,
                    "rewrite": rewrite,
                    "answer": normalize_text(raw_record.get("Answer", "")),
                    "answer_url": raw_record.get("Answer_URL", ""),
                    "nontrivial_rewrite": nontrivial
                }
            )
        write_jsonl(INTERIM_ROOT / f"qrecc_{split}.jsonl", cleaned_records)
        summary[split] = {
            "raw_count": len(records),
            "clean_count": len(cleaned_records),
            "duplicate_count": duplicate_count,
            "nontrivial_rewrite_count": nontrivial_count,
            "source_counts": dict(Counter(record["conversation_source"] for record in cleaned_records))
        }
    return summary


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Stream JSON objects from a JSONL file.

    Args:
        path: Source JSONL file.

    Yields:
        Parsed JSON object from each non-empty line.
    """
    with path.open("r", encoding = "utf-8") as file:
        for line_number, line in enumerate(file, start = 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error


def normalize_decomposition(
    decomposition: list[dict[str, Any]],
    idx_to_doc_id: dict[int, str]
) -> list[dict[str, Any]]:
    """Normalize a MuSiQue question decomposition.

    Args:
        decomposition: Raw ordered decomposition steps.
        idx_to_doc_id: Paragraph index to knowledge record mapping.

    Returns:
        Normalized decomposition steps.
    """
    steps = []
    for position, step in enumerate(decomposition, start = 1):
        support_index = int(step["paragraph_support_idx"])
        steps.append(
            {
                "step": position,
                "source_id": step.get("id"),
                "question": normalize_text(step["question"]),
                "answer": normalize_text(step["answer"]),
                "paragraph_support_idx": support_index,
                "gold_doc_id": idx_to_doc_id[support_index]
            }
        )
    return steps


def clean_musique() -> dict[str, Any]:
    """Clean MuSiQue answerable splits and build the auxiliary corpus.

    Returns:
        MuSiQue cleaning statistics.
    """
    data_root = MUSIQUE_RAW_ROOT / "original" / "data"
    knowledge_records: dict[str, dict[str, Any]] = {}
    split_summary: dict[str, Any] = {}

    for split in ["train", "dev", "test"]:
        raw_path = data_root / f"musique_ans_v1.0_{split}.jsonl"
        cleaned_records = []
        hop_counts: Counter[int] = Counter()
        for raw_record in tqdm(iter_jsonl(raw_path), desc = f"Cleaning MuSiQue {split}"):
            idx_to_doc_id = {}
            paragraph_doc_ids = []
            for paragraph in raw_record["paragraphs"]:
                title = normalize_text(paragraph["title"])
                text = normalize_text(paragraph["paragraph_text"])
                doc_id = "musique_" + stable_hash(title, text)
                idx_to_doc_id[int(paragraph["idx"])] = doc_id
                paragraph_doc_ids.append(doc_id)
                if doc_id not in knowledge_records:
                    knowledge_records[doc_id] = {
                        "id": doc_id,
                        "text": text,
                        "title": title,
                        "source": f"musique:{title}",
                        "source_dataset": "musique",
                        "namespace": "musique_aux",
                        "url": "",
                        "section_path": [],
                        "source_indices": [],
                        "split_membership": [split],
                        "content_hash": stable_hash(text, length = 64)
                    }
                elif split not in knowledge_records[doc_id]["split_membership"]:
                    knowledge_records[doc_id]["split_membership"].append(split)

            decomposition = normalize_decomposition(
                raw_record.get("question_decomposition", []),
                idx_to_doc_id
            )
            hop_count = len(decomposition)
            hop_counts[hop_count] += 1
            cleaned_records.append(
                {
                    "id": raw_record["id"],
                    "split": split,
                    "question": normalize_text(raw_record["question"]),
                    "answer": normalize_text(raw_record.get("answer", "")),
                    "answer_aliases": [
                        normalize_text(value)
                        for value in raw_record.get("answer_aliases", [])
                    ],
                    "answerable": bool(raw_record.get("answerable", True)),
                    "hop_count": hop_count,
                    "question_decomposition": decomposition,
                    "paragraph_doc_ids": paragraph_doc_ids,
                    "gold_doc_ids": [step["gold_doc_id"] for step in decomposition]
                }
            )
        write_jsonl(INTERIM_ROOT / f"musique_{split}.jsonl", cleaned_records)
        split_summary[split] = {
            "count": len(cleaned_records),
            "hop_counts": {str(key): value for key, value in sorted(hop_counts.items())}
        }

    knowledge_list = sorted(knowledge_records.values(), key = lambda record: record["id"])
    for record in knowledge_list:
        record["split_membership"].sort()
    write_jsonl(PROCESSED_ROOT / "knowledge_base" / "musique_aux.jsonl", knowledge_list)
    split_summary["knowledge_record_count"] = len(knowledge_list)
    return split_summary


def main() -> None:
    """Clean all raw datasets and write canonical intermediate files."""
    args = parse_args()
    summary_path = INTERIM_ROOT / "cleaning_summary.json"
    if summary_path.exists() and not args.force:
        logger.info("Cleaning outputs already exist; use --force to rebuild")
        return
    logger.info("=" * 80)
    logger.info("Cleaning source datasets")
    logger.info("=" * 80)
    conditionalqa_summary = clean_conditionalqa()
    qrecc_summary = clean_qrecc()
    musique_summary = clean_musique()
    summary = {
        "conditionalqa": {
            key: value
            for key, value in conditionalqa_summary.items()
            if key != "unresolved_evidence"
        },
        "qrecc": qrecc_summary,
        "musique": musique_summary
    }
    write_json(summary_path, summary)
    write_json(
        INTERIM_ROOT / "gold_coverage_issues.json",
        {
            "count": len(conditionalqa_summary["unresolved_evidence"]),
            "issues": conditionalqa_summary["unresolved_evidence"]
        }
    )
    logger.info("ConditionalQA evidence mapping rate: %.4f", summary["conditionalqa"]["evidence_mapping_rate"])
    logger.info("MuSiQue auxiliary knowledge records: %d", musique_summary["knowledge_record_count"])


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    main()
