"""Budget estimation helpers."""

from __future__ import annotations

from travel_agent.models import Activity, BudgetBreakdown, TripRequest


def estimate_budget_breakdown(
    trip: TripRequest,
    activities: list[Activity],
) -> BudgetBreakdown:
    """Allocate budget across food, activities, transport, and buffer.

    Accommodation is out of scope — the budget covers everything except lodging.
    """
    total = trip.budget_usd
    days = trip.trip_days

    # Style-based allocation ratios
    style = trip.travel_style.value
    if style == "relaxed":
        food_pct, activity_pct, transport_pct, buffer_pct = 0.50, 0.25, 0.13, 0.12
    elif style == "packed":
        food_pct, activity_pct, transport_pct, buffer_pct = 0.32, 0.45, 0.14, 0.09
    else:
        food_pct, activity_pct, transport_pct, buffer_pct = 0.38, 0.39, 0.14, 0.09

    food = total * food_pct
    activities_budget = total * activity_pct
    transport = total * transport_pct
    buffer = total * buffer_pct

    # Sanity check against selected activities
    activity_costs = sum(a.estimated_cost_usd or 0 for a in activities[: days * 2])
    per_person_food_daily = food / days / trip.travelers

    notes_parts = []
    if per_person_food_daily < 25:
        notes_parts.append("Food budget is tight; prioritize markets and casual dining.")
    if activity_costs > activities_budget * 1.2:
        notes_parts.append(
            "Selected activities may exceed the activity slice; consider free/low-cost options on some days."
        )

    computed_total = food + activities_budget + transport + buffer
    return BudgetBreakdown(
        food_estimate_usd=round(food, 2),
        activities_estimate_usd=round(activities_budget, 2),
        transport_estimate_usd=round(transport, 2),
        buffer_usd=round(buffer, 2),
        total_usd=round(computed_total, 2),
        within_budget=computed_total <= total * 1.01,
        notes=" ".join(notes_parts) if notes_parts else None,
    )
