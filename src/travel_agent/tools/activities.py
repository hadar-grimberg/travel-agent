"""Activity discovery within drive radius."""

from __future__ import annotations

from travel_agent.models import (
    NON_POI_OPTIONS,
    Activity,
    GeoLocation,
    Interest,
    InterestOption,
    TripRequest,
)
from travel_agent.tools.geo import (
    drive_minutes,
    fetch_opentripmap_pois,
    within_drive_radius,
)

# Interest → OpenTripMap kind mapping (query side).
# historic_architecture lives under the "architecture" branch, not "historic",
# so culture must request it explicitly to get palaces/amphitheatres.
# Nightlife queries specific adult sub-kinds (casino, nightclubs, alcohol)
# rather than the whole "adult" branch, which would pull in seedier venues.
_INTEREST_KINDS: dict[Interest, str] = {
    # gardens_and_parks sits under cultural/urban_environment and view_points
    # under "other" — neither is inside the natural branch. nature_reserves is
    # a child of natural (already fetched), listed explicitly for clarity.
    Interest.NATURE: "natural,nature_reserves,gardens_and_parks,view_points",
    # religion (temples, monasteries, cathedrals) is its own branch — without
    # it a "Kyoto temples" trip fetches no temples
    Interest.CULTURE: "cultural,historic,museums,historic_architecture,religion",
    # restaurants/cafes are children of foods (already fetched), explicit for clarity
    Interest.FOOD: "foods,restaurants,cafes",
    # specific sub-kinds instead of the whole "sport" branch, which also
    # contains non-adventure venues (stadiums, pools)
    Interest.ADVENTURE: "climbing,diving,surfing,kitesurfing,winter_sports",
    # children_theatres/circuses sit under cultural/theatres_and_entertainments,
    # planetariums under cultural/museums; amusements covers water parks, rides
    Interest.FAMILY: "amusements,zoos,aquariums,children_theatres,circuses,planetariums",
    Interest.BEACH: "beaches",
    Interest.SHOPPING: "shops",
    # restaurants (not all of foods) — evening dining, not bakeries/picnic sites
    Interest.NIGHTLIFE: "restaurants,nightclubs,casino,alcohol",
}

# OpenTripMap kind → Interest category (bucketing side). POIs carry many
# hierarchical kind slugs (e.g. "historic_architecture,interesting_places");
# matching is by substring per slug, first hit wins, so specific tokens
# (casino, beach) come before broad ones (natural, cultural).
_KIND_CATEGORY: dict[str, str] = {
    "casino": "nightlife",
    "nightclub": "nightlife",
    "alcohol": "nightlife",
    "hookah": "nightlife",
    # pubs/bars/biergartens are foods-branch kinds but belong in the
    # nightlife round; must precede the "foods" token
    "pub": "nightlife",
    "bar": "nightlife",
    "biergarten": "nightlife",
    "beach": "beach",
    # water_park must precede the nature tokens ("water") below
    "water_park": "family",
    "amusement": "family",
    "roller_coaster": "family",
    "ferris": "family",
    "zoo": "family",
    "aquarium": "family",
    "children_theatre": "family",
    "circus": "family",
    "planetarium": "family",
    # "natur" covers both "natural" and "nature_reserves"
    "natur": "nature",
    "garden": "nature",
    "view_point": "nature",
    "water": "nature",
    "foods": "food",
    "restaurant": "food",
    "cafe": "food",
    # shop before sport: the shops branch contains "sport_shops"
    "shop": "shopping",
    "mall": "shopping",
    "market": "shopping",
    "sport": "adventure",
    "climbing": "adventure",
    "diving": "adventure",
    "surf": "adventure",
}


def _category_from_kinds(kinds: str) -> str:
    """Normalize OpenTripMap kinds onto the closed 8-group vocabulary.

    Token-priority matching: tokens are tried in _KIND_CATEGORY order against
    ALL of the POI's kind slugs, so specific tokens (casino, bar, water_park)
    win over broad branch names (foods, natural) regardless of the order
    OpenTripMap lists the slugs in. Unmatched kinds (cultural, historic,
    religion, ...) default to "culture", the generic sightseeing bucket.
    """
    kind_list = [kind.strip().lower() for kind in kinds.split(",")]
    for token, category in _KIND_CATEGORY.items():
        if any(token in kind for kind in kind_list):
            return category
    return "culture"

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


def _kinds_for_interests(interests: list[InterestOption]) -> str:
    """Build the OpenTripMap kinds query from checked checklist items.

    Whole-group selections (e.g. "culture") expand to the group's curated
    kinds; granular items pass through as-is (their value IS the API slug);
    web-search-backed items (free_tours, ...) have no POI kind
    and are skipped — the research agent covers them via web search.
    """
    kinds: set[str] = set()
    for option in interests:
        if option in NON_POI_OPTIONS:
            continue
        if option.value in Interest._value2member_map_:
            group_kinds = _INTEREST_KINDS[Interest(option.value)]
            kinds.update(part.strip() for part in group_kinds.split(","))
        else:
            kinds.add(option.value)
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
        category=_category_from_kinds(kinds),
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
