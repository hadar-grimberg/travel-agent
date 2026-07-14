"""Unit tests for deterministic tool logic (no LLM required)."""

from datetime import date

import pytest
from pydantic import ValidationError

from travel_agent.models import (
    INTEREST_GROUP,
    NON_POI_OPTIONS,
    GeoLocation,
    Interest,
    InterestOption,
    TravelStyle,
    TripRequest,
)
from travel_agent.tools.activities import (
    _ITEM_CATEGORIES,
    _categories_for_interests,
    _category_from_kinds,
)
from travel_agent.tools.budget import estimate_budget_breakdown
from travel_agent.tools.geo import max_drive_radius_miles, within_drive_radius
from travel_agent.tools.reservations import reservation_item_for_activity


def test_max_drive_radius():
    assert max_drive_radius_miles(4) == 220


def test_within_drive_radius():
    origin = GeoLocation(name="Test", latitude=40.0, longitude=-74.0)
    # ~50 miles away
    assert within_drive_radius(origin, 40.4, -74.0, max_hours=4)
    # ~500 miles away
    assert not within_drive_radius(origin, 44.0, -74.0, max_hours=4)


def test_budget_breakdown_within_budget():
    trip = TripRequest(
        destination="Lisbon",
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 5),
        budget_usd=2000,
        travel_style=TravelStyle.BALANCED,
    )
    breakdown = estimate_budget_breakdown(trip, [])
    assert breakdown.within_budget
    assert breakdown.total_usd == 2000


def test_category_from_kinds_maps_to_interest_vocabulary():
    assert _category_from_kinds("tourism.sights.castle,tourism.sights") == "culture"
    assert _category_from_kinds("religion.place_of_worship.buddhism") == "culture"
    assert _category_from_kinds("adult.casino") == "nightlife"
    assert _category_from_kinds("beach,natural") == "beach"
    assert _category_from_kinds("catering.restaurant,catering") == "food"
    assert _category_from_kinds("entertainment.zoo") == "family"
    assert _category_from_kinds("entertainment.water_park") == "family"
    assert _category_from_kinds("entertainment.planetarium") == "family"
    assert _category_from_kinds("leisure.park.garden,leisure.park") == "nature"
    assert _category_from_kinds("natural.water.spring") == "nature"
    assert _category_from_kinds("tourism.attraction.viewpoint") == "nature"
    assert _category_from_kinds("sport.dive_centre") == "adventure"
    assert _category_from_kinds("commercial.shopping_mall") == "shopping"
    # unknown ids land in the generic sightseeing bucket
    assert _category_from_kinds("some.new.geoapify.category") == "culture"


def test_category_from_kinds_always_in_closed_list():
    valid = {i.value for i in Interest}
    for kinds in (
        "entertainment.museum",
        "man_made.tower",
        "sport,climbing",
        "commercial.marketplace",
        "",
    ):
        assert _category_from_kinds(kinds) in valid


def test_every_checklist_item_has_a_group():
    assert set(INTEREST_GROUP) == set(InterestOption)


def test_every_poi_item_has_geoapify_categories():
    group_names = {i.value for i in Interest}
    for option in InterestOption:
        if option in NON_POI_OPTIONS or option.value in group_names:
            continue
        assert option in _ITEM_CATEGORIES, f"missing Geoapify mapping for {option}"


def test_categories_granular_items_map_to_geoapify():
    categories = _categories_for_interests(
        [InterestOption.BUDDHIST_TEMPLES, InterestOption.VIEW_POINTS]
    )
    assert set(categories.split(",")) == {
        "religion.place_of_worship.buddhism",
        "tourism.attraction.viewpoint",
    }


def test_categories_group_selection_expands():
    categories = _categories_for_interests([InterestOption.CULTURE])
    assert "entertainment.museum" in categories
    assert "tourism.sights.place_of_worship" in categories


def test_categories_non_poi_items_are_skipped():
    assert _categories_for_interests(list(NON_POI_OPTIONS)) == ""
    categories = _categories_for_interests([InterestOption.FREE_TOURS, InterestOption.CAFES])
    assert categories == "catering.cafe"


def test_trip_request_accepts_granular_interests():
    trip = TripRequest(
        destination="Kyoto",
        start_date=date(2026, 4, 3),
        end_date=date(2026, 4, 9),
        budget_usd=2800,
        interests=["buddhist_temples", "free_tours"],
    )
    groups = {INTEREST_GROUP[opt] for opt in trip.interests}
    assert groups == {Interest.CULTURE}


def test_trip_request_rejects_unknown_interest():
    with pytest.raises(ValidationError):
        TripRequest(
            destination="Kyoto",
            start_date=date(2026, 4, 3),
            end_date=date(2026, 4, 9),
            budget_usd=2800,
            interests=["ziplines"],
        )


def test_bars_and_malls_bucketing():
    assert _category_from_kinds("catering.bar,catering") == "nightlife"
    assert _category_from_kinds("catering.pub") == "nightlife"
    assert _category_from_kinds("commercial.shopping_mall,commercial") == "shopping"
    # public_bath must NOT hit a bare "pub" token
    assert _category_from_kinds("leisure.spa.public_bath") == "family"


def test_reservation_item_restaurant():
    item = reservation_item_for_activity("The French Laundry", "Napa", "restaurant")
    assert item.reservation_required
    assert item.booking_url is not None
    assert "opentable" in item.booking_url.lower()
