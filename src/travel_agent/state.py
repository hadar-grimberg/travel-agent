"""LangGraph state definition."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from travel_agent.models import Itinerary, ResearchBundle, TripRequest


class TravelAgentState(TypedDict):
    """Shared state across graph nodes."""

    messages: Annotated[list, add_messages]
    trip_request: TripRequest | None
    active_skill_names: list[str]
    research: ResearchBundle | None
    itinerary: Itinerary | None
    phase: str
    current_recommendations: str | None  # latest AI-generated recommendations (JSON string)
    user_preferences: list[str]          # user's expressed interests/picks accumulated over the chat
    last_user_input: str | None          # most recent user message (used for routing)
    categories: list[str]                # ordered activity categories to walk through
    current_category_index: int          # position in categories
    category_refinement_count: int       # recommendation rounds done for the current category
