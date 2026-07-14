"""Contract tests for the ActivitiesAgent JSON facade (no LLM required)."""

import json

import pytest

from travel_agent.agent import AWAITING, ActivitiesAgent
from travel_agent.graph import _payload_from_text, should_continue_chat


@pytest.fixture()
def agent():
    return ActivitiesAgent()


def test_invalid_json_string(agent):
    response = agent.handle("{not json")
    assert response["status"] == "error"
    assert "Invalid JSON" in response["error"]


def test_unknown_action(agent):
    response = agent.handle({"action": "book_hotel"})
    assert response["status"] == "error"
    assert "Unknown action" in response["error"]


def test_start_requires_trip_or_request(agent):
    response = agent.handle({"action": "start"})
    assert response["status"] == "error"


def test_start_rejects_invalid_trip(agent):
    # missing required fields (dates, budget) → validation error, not a crash
    response = agent.handle({"action": "start", "trip": {"destination": "Kyoto"}})
    assert response["status"] == "error"
    assert "Invalid trip" in response["error"]


def test_feedback_requires_session_and_feedback(agent):
    response = agent.handle({"action": "feedback", "session_id": "x"})
    assert response["status"] == "error"


def test_feedback_unknown_session(agent):
    response = agent.handle({"action": "feedback", "session_id": "nope", "feedback": "approve"})
    assert response["status"] == "error"
    assert "Unknown session" in response["error"]


def test_status_unknown_session(agent):
    response = agent.handle({"action": "status", "session_id": "nope"})
    assert response["status"] == "error"


def test_feedback_must_be_structured_object(agent):
    agent._sessions["s1"] = AWAITING
    response = agent.handle({"action": "feedback", "session_id": "s1", "feedback": "approve"})
    assert response["status"] == "error"
    assert "JSON object" in response["error"]


def test_feedback_rejects_invalid_shape(agent):
    agent._sessions["s2"] = AWAITING
    response = agent.handle(
        {"action": "feedback", "session_id": "s2", "feedback": {"selected": "not-a-list"}}
    )
    assert response["status"] == "error"
    assert "Invalid feedback" in response["error"]


def test_routing_from_structured_feedback():
    assert should_continue_chat({"last_feedback": {"finish": True}}) == "synthesize_itinerary"
    assert should_continue_chat({"last_feedback": {"quit": True}}) == "__end__"
    assert (
        should_continue_chat(
            {"last_feedback": {"approve": True}, "category_refinement_count": 1}
        )
        == "advance_category"
    )
    assert (
        should_continue_chat(
            {"last_feedback": {"selected": ["Temple walk"]}, "category_refinement_count": 2}
        )
        == "generate_recommendations"
    )
    # round limit forces the advance even without an approval
    assert (
        should_continue_chat({"last_feedback": {}, "category_refinement_count": 6})
        == "advance_category"
    )


def test_payload_from_text_legacy_cli_keywords():
    assert _payload_from_text("create itinerary").finish
    assert _payload_from_text("exit").quit
    assert _payload_from_text("approve").approve
    payload = _payload_from_text("I love the temple walk")
    assert payload.selected == ["I love the temple walk"]
    assert not (payload.approve or payload.finish or payload.quit)


def test_handle_json_returns_string(agent):
    raw = agent.handle_json(json.dumps({"action": "status", "session_id": "nope"}))
    assert isinstance(raw, str)
    assert json.loads(raw)["status"] == "error"
