"""Contract tests for the ActivitiesAgent JSON facade (no LLM required)."""

import json

import pytest

from travel_agent.agent import ActivitiesAgent


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


def test_handle_json_returns_string(agent):
    raw = agent.handle_json(json.dumps({"action": "status", "session_id": "nope"}))
    assert isinstance(raw, str)
    assert json.loads(raw)["status"] == "error"
