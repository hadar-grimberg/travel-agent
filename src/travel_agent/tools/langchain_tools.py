"""LangChain tool wrappers exposed to the ReAct research agent."""

from __future__ import annotations

import json
from datetime import date
from typing import Annotated

from langchain_core.tools import tool

from travel_agent.models import Interest, TravelStyle, TripRequest
from travel_agent.tools.activities import discover_activities
from travel_agent.tools.budget import estimate_budget_breakdown
from travel_agent.tools.geo import geocode_destination, max_drive_radius_miles
from travel_agent.tools.reservations import (
    build_reservation_plan,
    reservation_item_for_activity,
    search_booking_info_with_tavily,
)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


@tool
def geocode_destination_tool(destination: Annotated[str, "City, region, or address"]) -> str:
    """Geocode a destination to latitude/longitude for drive-radius searches."""
    loc = geocode_destination(destination)
    return loc.model_dump_json()


@tool
def discover_nearby_activities_tool(
    destination: Annotated[str, "Base destination for the trip"],
    start_date: Annotated[str, "ISO date YYYY-MM-DD"],
    end_date: Annotated[str, "ISO date YYYY-MM-DD"],
    budget_usd: Annotated[float, "Total trip budget in USD"],
    interests: Annotated[
        str,
        "Comma-separated interests: nature, food, culture, adventure, nightlife, family, shopping, beach",
    ] = "culture,food",
    travel_style: Annotated[str, "relaxed, balanced, or packed"] = "balanced",
    travelers: Annotated[int, "Number of travelers"] = 2,
) -> str:
    """Find activities and day-trip ideas within a 4-hour drive of the destination."""
    parsed_interests = []
    for part in interests.split(","):
        part = part.strip().lower()
        try:
            parsed_interests.append(Interest(part))
        except ValueError:
            continue
    if not parsed_interests:
        parsed_interests = [Interest.CULTURE, Interest.FOOD]

    trip = TripRequest(
        destination=destination,
        start_date=_parse_date(start_date),
        end_date=_parse_date(end_date),
        budget_usd=budget_usd,
        travelers=travelers,
        interests=parsed_interests,
        travel_style=TravelStyle(travel_style),
    )
    origin = geocode_destination(destination)
    activities = discover_activities(origin, trip)
    payload = {
        "max_drive_radius_miles": max_drive_radius_miles(),
        "origin": origin.model_dump(),
        "trip_days": trip.trip_days,
        "daily_budget_usd": trip.daily_budget_usd,
        "activities": [a.model_dump() for a in activities],
    }
    return json.dumps(payload, default=str)


@tool
def estimate_trip_budget_tool(
    destination: Annotated[str, "Trip destination"],
    start_date: Annotated[str, "ISO date YYYY-MM-DD"],
    end_date: Annotated[str, "ISO date YYYY-MM-DD"],
    budget_usd: Annotated[float, "Total budget USD"],
    travel_style: Annotated[str, "relaxed, balanced, or packed"] = "balanced",
    travelers: Annotated[int, "Number of travelers"] = 2,
) -> str:
    """Estimate budget allocation across food, activities, transport, and buffer (lodging excluded)."""
    trip = TripRequest(
        destination=destination,
        start_date=_parse_date(start_date),
        end_date=_parse_date(end_date),
        budget_usd=budget_usd,
        travelers=travelers,
        travel_style=TravelStyle(travel_style),
    )
    origin = geocode_destination(destination)
    activities = discover_activities(origin, trip, max_results=10)
    breakdown = estimate_budget_breakdown(trip, activities)
    return breakdown.model_dump_json()


@tool
def check_reservation_requirements_tool(
    venue_name: Annotated[str, "Restaurant, landmark, or excursion name"],
    destination: Annotated[str, "Trip destination city/region"],
    category: Annotated[str, "restaurant, landmark, excursion, or other"] = "other",
) -> str:
    """Check whether a venue likely requires advance booking and return booking links."""
    item = reservation_item_for_activity(venue_name, destination, category)
    live = search_booking_info_with_tavily(venue_name, destination)
    return json.dumps(
        {"reservation": item.model_dump(), "live_research": live},
        default=str,
    )


@tool
def build_reservation_checklist_tool(
    destination: Annotated[str, "Trip destination"],
    venue_names: Annotated[str, "Comma-separated list of venues to check for reservations"],
) -> str:
    """Build a prioritized reservation checklist with booking URLs for multiple venues."""
    names = [v.strip() for v in venue_names.split(",") if v.strip()]
    items = build_reservation_plan([], destination, extra_venues=names)
    return json.dumps([i.model_dump() for i in items], default=str)


@tool
def web_search_travel_tool(
    query: Annotated[str, "Search query for travel research, restaurants, or excursions"],
) -> str:
    """Search the web for up-to-date travel information (requires TAVILY_API_KEY)."""
    return search_booking_info_with_tavily(query, "")


RESEARCH_TOOLS = [
    geocode_destination_tool,
    discover_nearby_activities_tool,
    estimate_trip_budget_tool,
    check_reservation_requirements_tool,
    build_reservation_checklist_tool,
    web_search_travel_tool,
]
