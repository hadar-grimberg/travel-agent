"""Reservation guidance and booking link discovery."""

from __future__ import annotations

import os
from urllib.parse import quote_plus

import httpx

from travel_agent.models import Activity, ReservationItem

# Platforms commonly used for advance booking
_BOOKING_HINTS: dict[str, tuple[str, int]] = {
    "restaurant": ("OpenTable or Resy", 7),
    "museum": ("Official venue site or Tiqets", 14),
    "tour": ("Viator or GetYourGuide", 7),
    "winery": ("Winery direct booking", 14),
    "theatre": ("Official box office", 21),
    "observatory": ("Timed entry on official site", 30),
    "national park": ("Recreation.gov or park site", 60),
}


def _category_from_text(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ("restaurant", "bistro", "dining", "café", "cafe")):
        return "restaurant"
    if any(k in lower for k in ("tour", "excursion", "cruise", "safari")):
        return "excursion"
    if any(k in lower for k in ("museum", "landmark", "monument", "castle", "cathedral")):
        return "landmark"
    return "other"


def _lead_time_for(category: str, name: str) -> tuple[bool, int | None, str | None]:
    lower = f"{category} {name}".lower()
    for keyword, (platform, days) in _BOOKING_HINTS.items():
        if keyword in lower:
            return True, days, platform
    if category in ("restaurant", "landmark", "excursion"):
        return True, 7, "Official website"
    return False, None, None


def build_booking_search_url(name: str, destination: str) -> str:
    """Google search URL focused on official booking pages."""
    query = quote_plus(f"{name} {destination} official reservations book tickets")
    return f"https://www.google.com/search?q={query}"


def build_opentable_search_url(name: str, destination: str) -> str:
    query = quote_plus(f"{name} {destination}")
    return f"https://www.opentable.com/s?term={query}"


def build_viator_search_url(name: str, destination: str) -> str:
    query = quote_plus(f"{name} {destination}")
    return f"https://www.viator.com/searchResults/all?text={query}"


def reservation_item_for_activity(name: str, destination: str, category: str | None = None) -> ReservationItem:
    cat = category or _category_from_text(name)
    required, lead_days, platform = _lead_time_for(cat, name)

    if cat == "restaurant":
        url = build_opentable_search_url(name, destination)
    elif cat == "excursion":
        url = build_viator_search_url(name, destination)
    else:
        url = build_booking_search_url(name, destination)

    return ReservationItem(
        name=name,
        category=cat if cat in ("restaurant", "landmark", "excursion", "tour", "other") else "other",
        reservation_required=required,
        lead_time_days=lead_days,
        booking_url=url,
        booking_platform=platform,
        notes="Book on the official site when possible; third-party aggregators as backup.",
    )


def build_reservation_plan(
    activities: list[Activity],
    destination: str,
    extra_venues: list[str] | None = None,
) -> list[ReservationItem]:
    """Build reservation checklist from activities and named venues."""
    items: list[ReservationItem] = []
    seen: set[str] = set()

    for activity in activities:
        if not activity.reservation_required and activity.category not in ("foods", "restaurant"):
            continue
        key = activity.name.lower()
        if key in seen:
            continue
        seen.add(key)
        item = reservation_item_for_activity(activity.name, destination, activity.category)
        if activity.booking_url:
            item = item.model_copy(update={"booking_url": activity.booking_url})
        items.append(item)

    for venue in extra_venues or []:
        key = venue.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(reservation_item_for_activity(venue, destination))

    items.sort(key=lambda i: (-int(i.reservation_required), -(i.lead_time_days or 0)))
    return items


def search_booking_info_with_tavily(name: str, destination: str) -> str:
    """Optional live web lookup for booking requirements via Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            f"No TAVILY_API_KEY set. Manual search: {build_booking_search_url(name, destination)}"
        )

    query = f"{name} {destination} reservations booking required advance tickets"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 3,
    }
    with httpx.Client(timeout=20) as client:
        response = client.post("https://api.tavily.com/search", json=payload)
        response.raise_for_status()
        data = response.json()

    snippets = []
    for result in data.get("results", [])[:3]:
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")
        snippets.append(f"- {title}: {content[:200]}... ({url})")

    return "\n".join(snippets) if snippets else "No live results; use official venue search."
