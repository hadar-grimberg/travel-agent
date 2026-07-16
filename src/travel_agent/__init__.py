"""Activities domain agent — category-by-category activity planning.

Stable entry point for hosts (e.g. Tripper): construct ``ActivitiesAgent`` and
call ``.handle(request) -> response`` with plain dicts. Importing this package
has no side effects — no keys, no network, no server. With no LLM credentials
configured the agent runs in a keyless mock mode.
"""

from __future__ import annotations

from travel_agent.agent import ActivitiesAgent
from travel_agent.config import Settings
from travel_agent.models import (
    FeedbackPayload,
    Itinerary,
    RecommendationResponse,
    TripRequest,
)

__all__ = [
    "ActivitiesAgent",
    "Settings",
    "TripRequest",
    "FeedbackPayload",
    "RecommendationResponse",
    "Itinerary",
]

__version__ = "0.1.0"
