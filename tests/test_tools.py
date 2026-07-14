"""Unit tests for deterministic tool logic (no LLM required)."""

from datetime import date

from travel_agent.models import GeoLocation, TravelStyle, TripRequest
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


def test_reservation_item_restaurant():
    item = reservation_item_for_activity("The French Laundry", "Napa", "restaurant")
    assert item.reservation_required
    assert item.booking_url is not None
    assert "opentable" in item.booking_url.lower()
