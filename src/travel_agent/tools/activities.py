"""Activity discovery within drive radius (Geoapify Places)."""

from __future__ import annotations

from travel_agent.models import (
    GROUP_ITEMS,
    NON_POI_OPTIONS,
    Activity,
    GeoLocation,
    Interest,
    InterestOption,
    TripRequest,
)
from travel_agent.tools.geo import (
    drive_minutes,
    fetch_geoapify_pois,
    within_drive_radius,
)

# Checklist item → Geoapify Places category ids (query side).
# Ids verified against https://apidocs.geoapify.com/docs/places/ — unknown ids
# make the API return 400, so only add categories confirmed in the docs.
# Items without an exact Geoapify match map to the closest verified category.
_ITEM_CATEGORIES: dict[InterestOption, str] = {
    # culture & history
    InterestOption.MUSEUMS: "entertainment.museum",
    InterestOption.THEATRES_AND_ENTERTAINMENTS: "entertainment.culture.theatre",
    InterestOption.URBAN_ENVIRONMENT: "tourism.attraction",
    InterestOption.HISTORICAL_PLACES: "tourism.sights",
    InterestOption.FORTIFICATIONS: "tourism.sights.castle",
    InterestOption.MONUMENTS_AND_MEMORIALS: "tourism.sights.memorial",
    InterestOption.ARCHAEOLOGY: "tourism.sights.archaeological_site",
    InterestOption.BURIAL_PLACES: "tourism.sights",
    InterestOption.HISTORIC_ARCHITECTURE: "tourism.sights.castle,tourism.sights.building",
    InterestOption.SKYSCRAPERS: "man_made.tower",
    InterestOption.BRIDGES: "man_made.bridge",
    InterestOption.TOWERS: "man_made.tower",
    InterestOption.LIGHTHOUSES: "man_made.lighthouse",
    InterestOption.RELIGION: "tourism.sights.place_of_worship",
    InterestOption.CHURCHES: "religion.place_of_worship.christianity",
    InterestOption.CATHEDRALS: "religion.place_of_worship.christianity",
    InterestOption.MOSQUES: "religion.place_of_worship.islam",
    InterestOption.SYNAGOGUES: "religion.place_of_worship.judaism",
    InterestOption.BUDDHIST_TEMPLES: "religion.place_of_worship.buddhism",
    InterestOption.HINDU_TEMPLES: "religion.place_of_worship.hinduism",
    InterestOption.EGYPTIAN_TEMPLES: "tourism.sights.archaeological_site",
    InterestOption.OTHER_TEMPLES: "tourism.sights.place_of_worship",
    InterestOption.MONASTERIES: "tourism.sights.monastery",
    # nature
    InterestOption.NATURE_RESERVES: "natural",
    InterestOption.GARDENS_AND_PARKS: "leisure.park,leisure.park.garden",
    InterestOption.VIEW_POINTS: "tourism.attraction.viewpoint",
    InterestOption.ISLANDS: "natural",
    InterestOption.NATURAL_SPRINGS: "natural.water.spring",
    InterestOption.GEOLOGICAL_FORMATIONS: "natural.mountain.peak",
    InterestOption.WATER: "natural.water",
    InterestOption.GLACIERS: "natural.mountain.peak",
    # beach
    InterestOption.BEACHES: "beach",
    # food & drink
    InterestOption.RESTAURANTS: "catering.restaurant",
    InterestOption.CAFES: "catering.cafe",
    InterestOption.BAKERIES: "commercial.food_and_drink.bakery",
    InterestOption.FAST_FOOD: "catering.fast_food",
    InterestOption.FOOD_COURTS: "catering.food_court",
    InterestOption.PICNIC_SITES: "leisure.picnic.picnic_site",
    # nightlife
    InterestOption.PUBS: "catering.pub",
    InterestOption.BARS: "catering.bar",
    InterestOption.BIERGARTENS: "catering.biergarten",
    InterestOption.NIGHTCLUBS: "adult.nightclub",
    InterestOption.CASINO: "adult.casino",
    InterestOption.ALCOHOL: "catering.bar,catering.pub",
    InterestOption.HOOKAH: "catering.bar",
    # adventure & sport
    InterestOption.CLIMBING: "entertainment.activity_park.climbing",
    InterestOption.DIVING: "sport.dive_centre",
    InterestOption.SURFING: "sport",
    InterestOption.KITESURFING: "sport",
    InterestOption.WINTER_SPORTS: "sport.ice_rink",
    InterestOption.STADIUMS: "sport.stadium",
    InterestOption.POOLS: "sport.swimming_pool",
    # family
    InterestOption.AMUSEMENT_PARKS: "entertainment.theme_park",
    InterestOption.WATER_PARKS: "entertainment.water_park",
    InterestOption.ZOOS: "entertainment.zoo",
    InterestOption.AQUARIUMS: "entertainment.aquarium",
    InterestOption.CHILDREN_THEATRES: "entertainment.culture.theatre",
    InterestOption.CIRCUSES: "entertainment.culture.theatre",
    InterestOption.PLANETARIUMS: "entertainment.planetarium",
    InterestOption.MINIATURE_PARKS: "entertainment.theme_park",
    InterestOption.ROLLER_COASTERS: "entertainment.theme_park",
    InterestOption.FERRIS_WHEELS: "entertainment.theme_park",
    InterestOption.BATHS_AND_SAUNAS: "leisure.spa.public_bath,leisure.spa.sauna",
    # shopping
    InterestOption.MALLS: "commercial.shopping_mall",
    InterestOption.MARKETPLACES: "commercial.marketplace",
    InterestOption.OUTDOOR_SHOPS: "commercial",
    InterestOption.SUPERMARKETS: "commercial.supermarket",
    InterestOption.CONVENIENCES: "commercial.supermarket",
    InterestOption.FISH_STORES: "commercial.food_and_drink",
}

# Geoapify category token → category group (bucketing side). Token-priority
# matching: tokens are tried in order against all of the POI's category ids,
# so specific tokens (casino, catering.bar) win over broad branch names
# (catering, natural). "catering.pub"/"catering.bar" are deliberately dotted:
# a bare "pub" would collide with leisure.spa.public_bath.
_KIND_CATEGORY: dict[str, str] = {
    "casino": "nightlife",
    "nightclub": "nightlife",
    "catering.pub": "nightlife",
    "catering.bar": "nightlife",
    "biergarten": "nightlife",
    "beach": "beach",
    "water_park": "family",
    "theme_park": "family",
    "amusement": "family",
    "zoo": "family",
    "aquarium": "family",
    "planetarium": "family",
    "spa": "family",
    "natural": "nature",
    "leisure.park": "nature",
    "garden": "nature",
    "viewpoint": "nature",
    "picnic": "food",
    "bakery": "food",
    "catering": "food",
    "commercial": "shopping",
    "sport": "adventure",
    "climbing": "adventure",
    "dive": "adventure",
}


def _category_from_kinds(kinds: str) -> str:
    """Normalize Geoapify category ids onto the closed 8-group vocabulary.

    Unmatched ids (entertainment.museum, tourism.sights.*, religion.*,
    man_made.*, heritage.*) default to "culture", the generic sightseeing
    bucket.
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


def _categories_for_interests(interests: list[InterestOption]) -> str:
    """Build the Geoapify categories query from checked checklist items.

    Whole-group selections (e.g. "culture") expand to the union of the group's
    item categories; granular items map through _ITEM_CATEGORIES; web-search
    backed items (free_tours, guided_tours) have no POI category and are
    skipped — the research agent covers them via web search.
    """
    categories: set[str] = set()
    for option in interests:
        if option in NON_POI_OPTIONS:
            continue
        if option.value in Interest._value2member_map_:
            for item in GROUP_ITEMS[Interest(option.value)]:
                if item in NON_POI_OPTIONS or item.value in Interest._value2member_map_:
                    continue
                categories.update(_ITEM_CATEGORIES[item].split(","))
        else:
            categories.update(_ITEM_CATEGORIES[option].split(","))
    return ",".join(sorted(categories))


def _poi_to_activity(feature: dict, origin: GeoLocation) -> Activity | None:
    """Convert a Geoapify GeoJSON feature into an Activity."""
    props = feature.get("properties") or {}
    lat, lon = props.get("lat"), props.get("lon")
    if lat is None or lon is None:
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coords) == 2:
            lon, lat = coords
    if lat is None or lon is None:
        return None
    if not within_drive_radius(origin, lat, lon):
        return None

    name = props.get("name") or props.get("address_line1") or "Unnamed place"
    categories = ",".join(props.get("categories") or [])
    primary = (props.get("categories") or ["place"])[0].split(".")[-1].replace("_", " ")
    description = props.get("address_line2") or f"Popular {primary} near {origin.name}."
    return Activity(
        name=name,
        category=_category_from_kinds(categories),
        description=description,
        latitude=lat,
        longitude=lon,
        drive_minutes_from_base=drive_minutes(origin, lat, lon),
        estimated_cost_usd=_estimate_cost_from_kinds(categories),
        duration_hours=_duration_from_kinds(categories),
        reservation_required=_reservation_likely(categories, name),
        source="geoapify",
    )


def _estimate_cost_from_kinds(kinds: str) -> float:
    if "museum" in kinds or "theatre" in kinds:
        return 35
    if "catering" in kinds:
        return 60
    if "natural" in kinds or "beach" in kinds:
        return 10
    return 25


def _duration_from_kinds(kinds: str) -> float:
    if "museum" in kinds:
        return 3.0
    if "natural" in kinds or "sights" in kinds:
        return 2.5
    if "catering" in kinds:
        return 1.5
    return 2.0


def _reservation_likely(kinds: str, name: str) -> bool:
    keywords = ("museum", "tour", "restaurant", "theatre", "observatory", "winery", "casino")
    lower = f"{kinds} {name}".lower()
    return any(k in lower for k in keywords)


def discover_activities(
    origin: GeoLocation,
    trip: TripRequest,
    max_results: int = 20,
) -> list[Activity]:
    """Discover activities within a 4-hour drive of the destination."""
    categories = _categories_for_interests(trip.interests)
    pois = fetch_geoapify_pois(origin, categories, limit=max_results * 2)

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
