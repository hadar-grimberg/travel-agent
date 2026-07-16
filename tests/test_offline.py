"""Keyless / offline contract tests — run with zero secrets configured.

These exercise the mock provider end to end: import safety, the definition-of-done
one-liner, and a full start -> feedback -> completed conversation with no LLM or
network. Live tests belong elsewhere and are marked ``e2e``.
"""

import importlib

from travel_agent import ActivitiesAgent, Settings

# A structured trip spanning several category groups
SAMPLE_TRIP = {
    "destination": "Kyoto",
    "start_date": "2026-04-03",
    "end_date": "2026-04-06",
    "budget_usd": 1800,
    "travelers": 2,
    "interests": ["buddhist_temples", "view_points", "restaurants"],
}


def _mock_agent() -> ActivitiesAgent:
    """Agent forced into keyless mock mode regardless of ambient env."""
    return ActivitiesAgent(settings=Settings(llm_provider="mock"))


def test_import_is_side_effect_free():
    # Re-importing must not require keys, hit the network, or start a server.
    importlib.import_module("travel_agent")
    importlib.import_module("travel_agent.agent")
    importlib.import_module("travel_agent.graph")


def test_settings_defaults_to_mock_without_keys():
    assert Settings.from_env(env={}).use_mock is True
    assert Settings.from_env(env={"NEBIUS_API_KEY": "k", "NEBIUS_BASE_URL": "u"}).use_mock is False


def test_definition_of_done_one_liner():
    # from travel_agent import ActivitiesAgent; ActivitiesAgent().handle({...})
    response = _mock_agent().handle({"action": "start", "trip": SAMPLE_TRIP})
    assert response["status"] == "awaiting_feedback"
    # buddhist_temples belongs to the culture group, so the first round is culture
    assert response["recommendations"]["category"] == "culture"
    assert response["recommendations"]["recommendations"], "expected mock recommendations"


def test_full_keyless_conversation_reaches_itinerary():
    agent = _mock_agent()
    response = agent.handle({"action": "start", "trip": SAMPLE_TRIP})
    session_id = response["session_id"]

    # Approve every category until the itinerary is built (guard against loops)
    for _ in range(20):
        if response["status"] != "awaiting_feedback":
            break
        recs = response["recommendations"]["recommendations"]
        selected = [recs[0]["name"]] if recs else []
        response = agent.handle({
            "action": "feedback",
            "session_id": session_id,
            "feedback": {"selected": selected, "approve": True},
        })

    assert response["status"] == "completed"
    itinerary = response["itinerary"]
    assert itinerary["destination"] == "Kyoto"
    assert itinerary["trip_days"] == 4
    assert len(itinerary["days"]) == 4
    # selections were threaded into the plan
    assert response["user_preferences"]


def test_keyless_free_text_start_does_not_require_keys():
    response = _mock_agent().handle({"action": "start", "request": "Lisbon for a few days"})
    assert response["status"] == "awaiting_feedback"


def test_quit_ends_without_itinerary():
    agent = _mock_agent()
    response = agent.handle({"action": "start", "trip": SAMPLE_TRIP})
    response = agent.handle({
        "action": "feedback",
        "session_id": response["session_id"],
        "feedback": {"quit": True},
    })
    assert response["status"] == "ended"
