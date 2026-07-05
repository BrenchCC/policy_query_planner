import pytest

from data_preprocess.schemas import serialize_plan, validate_planner_plan


def test_single_query_plan() -> None:
    """Accept a valid one-query planner response."""
    value = serialize_plan(
        [
            {
                "id": "q1",
                "query": "paternity leave eligibility",
                "depends_on": []
            }
        ]
    )
    assert validate_planner_plan(value)["queries"][0]["id"] == "q1"


def test_dependency_placeholder() -> None:
    """Accept matching ordered dependencies and placeholders."""
    plan = {
        "queries": [
            {"id": "q1", "query": "policy owner", "depends_on": []},
            {
                "id": "q2",
                "query": "{{q1.answer}} eligibility rules",
                "depends_on": ["q1"]
            }
        ]
    }
    assert len(validate_planner_plan(plan)["queries"]) == 2


def test_reject_forward_dependency() -> None:
    """Reject references to a future query."""
    plan = {
        "queries": [
            {
                "id": "q1",
                "query": "{{q2.answer}} eligibility rules",
                "depends_on": ["q2"]
            },
            {"id": "q2", "query": "policy owner", "depends_on": []}
        ]
    }
    with pytest.raises(ValueError, match = "earlier query"):
        validate_planner_plan(plan)

