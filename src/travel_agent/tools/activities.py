"""Activity discovery within drive radius."""

from __future__ import annotations

from travel_agent.models import Activity, GeoLocation, Interest, TripRequest
from travel_agent.tools.geo import (
    drive_minutes,
    fetch_opentripmap_pois,
    within_drive_radius,
)

# Interest → OpenTripMap kind mapping
_INTEREST_KINDS: dict[Interest, str] = {
    Interest.NATURE: "natural",
    Interest.CULTURE: "cultural,historic,museums",
    Interest.FOOD: "foods",
    Interest.ADVENTURE: "sport",
    Interest.FAMILY: "amusements,zoos,aquariums",
    Interest.BEACH: "beaches",
    Interest.SHOPPING: "shops",
    Interest.NIGHTLIFE: "foods",
}

# Fallback curated suggestions when no API key (agent enriches via web search)
_FALLBACK_BY_REGION: dict[str, list[dict]] = {
    "default": [
        {
            "name": "Historic downtown walking tour",
            "category": "culture",
            "description": "Self-guided or guided walk through the historic core.",
            "duration_hours": 2.5,
            "estimated_cost_usd": 25,
            "reservation_required": False,
        },
        {
            "name": "Local farmers market",
            "category": "food",
            "description": "Regional produce, street food, and artisan goods.",
            "duration_hours": 1.5,
            "estimated_cost_usd": 30,
            "reservation_required": False,
        },
        {
            "name": "Scenic overlook day trip",
            "category": "nature",
            "description": "Half-day drive to a well-known viewpoint or park.",
            "duration_hours": 4,
            "estimated_cost_usd": 15,
            "reservation_required": False,
        },
    ],
}


def _kinds_for_interests(interests: list[Interest]) -> str:
    kinds = set()
    for interest in interests:
        for part in _INTEREST_KINDS.get(interest, "interesting_places").split(","):
            kinds.add(part.strip())
    return ",".join(sorted(kinds)) or "interesting_places"


def _poi_to_activity(poi: dict, origin: GeoLocation) -> Activity | None:
    point = poi.get("point") or {}
    lat, lon = point.get("lat"), point.get("lon")
    if lat is None or lon is None:
        return None
    if not within_drive_radius(origin, lat, lon):
        return None

    props = poi.get("properties") or poi
    name = props.get("name") or props.get("name_en") or "Unnamed place"
    kinds = props.get("kinds", "place")
    return Activity(
        name=name,
        category=kinds.split(",")[0],
        description=props.get("wikidata") or f"Popular {kinds.replace(',', ' / ')} near {origin.name}.",
        latitude=lat,
        longitude=lon,
        drive_minutes_from_base=drive_minutes(origin, lat, lon),
        estimated_cost_usd=_estimate_cost_from_kinds(kinds),
        duration_hours=_duration_from_kinds(kinds),
        reservation_required=_reservation_likely(kinds, name),
        source="opentripmap",
    )


def _estimate_cost_from_kinds(kinds: str) -> float:
    if "museums" in kinds or "theatres" in kinds:
        return 35
    if "foods" in kinds:
        return 60
    if "natural" in kinds or "beaches" in kinds:
        return 10
    return 25


def _duration_from_kinds(kinds: str) -> float:
    if "museums" in kinds:
        return 3.0
    if "natural" in kinds or "historic" in kinds:
        return 2.5
    if "foods" in kinds:
        return 1.5
    return 2.0


def _reservation_likely(kinds: str, name: str) -> bool:
    keywords = ("museum", "tour", "restaurant", "theatre", "observatory", "winery")
    lower = f"{kinds} {name}".lower()
    return any(k in lower for k in keywords)


def discover_activities(
    origin: GeoLocation,
    trip: TripRequest,
    max_results: int = 20,
) -> list[Activity]:
    """Discover activities within a 4-hour drive of the destination."""
    kinds = _kinds_for_interests(trip.interests)
    pois = fetch_opentripmap_pois(origin, kinds=kinds, limit=max_results * 2)

    activities: list[Activity] = []
    for poi in pois:
        activity = _poi_to_activity(poi, origin)
        if activity:
            activities.append(activity)

    if not activities:
        for template in _FALLBACK_BY_REGION.get("default", []):
            activities.append(
                Activity(
                    name=template["name"],
                    category=template["category"],
                    description=template["description"],
                    duration_hours=template["duration_hours"],
                    estimated_cost_usd=template["estimated_cost_usd"],
                    reservation_required=template["reservation_required"],
                    source="fallback",
                )
            )

    # Sort: closer first, then lower cost for budget trips
    activities.sort(
        key=lambda a: (
            a.drive_minutes_from_base if a.drive_minutes_from_base is not None else 999,
            a.estimated_cost_usd or 0,
        )
    )
    return activities[:max_results]
