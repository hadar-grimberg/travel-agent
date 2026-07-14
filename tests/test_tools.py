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
from travel_agent.tools.activities import _category_from_kinds, _kinds_for_interests
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
    assert _category_from_kinds("historic_architecture,interesting_places") == "culture"
    assert _category_from_kinds("casino,adult") == "nightlife"
    assert _category_from_kinds("beaches,natural") == "beach"
    assert _category_from_kinds("foods,restaurants") == "food"
    assert _category_from_kinds("zoos,amusements") == "family"
    assert _category_from_kinds("water_parks,amusements") == "family"
    assert _category_from_kinds("circuses,theatres_and_entertainments") == "family"
    assert _category_from_kinds("planetariums,museums") == "family"
    assert _category_from_kinds("gardens_and_parks,urban_environment") == "nature"
    assert _category_from_kinds("nature_reserves") == "nature"
    assert _category_from_kinds("view_points,other") == "nature"
    assert _category_from_kinds("restaurants") == "food"
    assert _category_from_kinds("cafes") == "food"
    assert _category_from_kinds("religion,temples") == "culture"
    assert _category_from_kinds("kitesurfing,sport") == "adventure"
    assert _category_from_kinds("sport_shops,shops") == "shopping"
    # unknown kinds land in the generic sightseeing bucket
    assert _category_from_kinds("some_new_opentripmap_kind") == "culture"


def test_category_from_kinds_always_in_closed_list():
    valid = {i.value for i in Interest}
    for kinds in ("museums", "skyscrapers,architecture", "sport,climbing", "shops,marketplaces", ""):
        assert _category_from_kinds(kinds) in valid


def test_every_checklist_item_has_a_group():
    assert set(INTEREST_GROUP) == set(InterestOption)


def test_kinds_granular_items_pass_through():
    kinds = _kinds_for_interests([InterestOption.BUDDHIST_TEMPLES, InterestOption.VIEW_POINTS])
    assert set(kinds.split(",")) == {"buddhist_temples", "view_points"}


def test_kinds_group_selection_expands():
    kinds = _kinds_for_interests([InterestOption.CULTURE])
    assert "museums" in kinds and "religion" in kinds


def test_kinds_non_poi_items_are_skipped():
    assert _kinds_for_interests(list(NON_POI_OPTIONS)) == "interesting_places"
    kinds = _kinds_for_interests([InterestOption.FREE_TOURS, InterestOption.CAFES])
    assert kinds == "cafes"


def test_trip_request_accepts_granular_interests():
    trip = TripRequest(
        destination="Kyoto",
        start_date=date(2026, 4, 3),
        end_date=date(2026, 4, 9),
        budget_usd=2800,
        interests=["buddhist_temples", "free_tours"],
    )
    groups = {INTEREST_GROUP[opt] for opt in trip.interests}
    assert groups == {Interest.CULTURE, Interest.FOOD}


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
    assert _category_from_kinds("foods,bars") == "nightlife"
    assert _category_from_kinds("pubs,foods") == "nightlife"
    assert _category_from_kinds("malls,shops") == "shopping"


def test_reservation_item_restaurant():
    item = reservation_item_for_activity("The French Laundry", "Napa", "restaurant")
    assert item.reservation_required
    assert item.booking_url is not None
    assert "opentable" in item.booking_url.lower()
