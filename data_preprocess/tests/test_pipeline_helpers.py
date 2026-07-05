import json

from data_preprocess.prompts import build_prompt
from data_preprocess.clean_data import best_evidence_chunk, chunk_policy_document
from data_preprocess.build_datasets import convert_musique_plan, make_rejected_query


def test_policy_chunking_and_evidence_mapping() -> None:
    """Build a policy chunk and map source HTML evidence to it."""
    document = {
        "title": "Paternity Leave",
        "url": "https://example.test/paternity",
        "contents": [
            "<h1>Eligibility</h1>",
            "<p>You must work for your employer for at least 26 weeks.</p>"
        ]
    }
    chunks = chunk_policy_document(document, ["train"])
    chunk_id, score = best_evidence_chunk(
        "<p>You must work for your employer for at least 26 weeks.</p>",
        chunks
    )
    assert chunk_id == chunks[0]["id"]
    assert score > 1.0


def test_musique_plan_conversion() -> None:
    """Convert MuSiQue references into explicit planner dependencies."""
    record = {
        "question_decomposition": [
            {"step": 1, "question": "The Collegian >> owned by"},
            {"step": 2, "question": "When was #1 founded?"}
        ]
    }
    plan = json.loads(convert_musique_plan(record))
    assert plan["queries"][1]["depends_on"] == ["q1"]
    assert "{{q1.answer}}" in plan["queries"][1]["query"]


def test_rejected_query_differs() -> None:
    """Create a plausible negative that differs from the chosen query."""
    record = {"question": "Did he qualify?", "title": "Paternity Leave"}
    chosen = "Paternity Leave eligibility after 26 weeks employment"
    rejected = make_rejected_query(record, chosen, "constraint_omission", "wrong topic")
    assert rejected != chosen


def test_generation_prompt_contains_schema() -> None:
    """Render a complete SFT generation prompt."""
    prompt = build_prompt(
        "sft",
        {
            "title": "Paternity Leave",
            "scenario": "I have worked for two months.",
            "question": "Do I qualify?"
        }
    )
    assert "Return exactly one JSON object" in prompt
    assert '"queries"' in prompt

