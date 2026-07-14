"""Geocoding and drive-time utilities."""

from __future__ import annotations

import math
import os

import httpx
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

from travel_agent.models import GeoLocation

# Average highway speed for drive-time estimates (mph)
_AVERAGE_SPEED_MPH = 55
_MAX_DRIVE_HOURS = 4


def geocode_destination(destination: str) -> GeoLocation:
    """Resolve a destination string to coordinates using Nominatim (OpenStreetMap)."""
    geolocator = Nominatim(user_agent="travel-agent-langgraph/0.1")
    location = geolocator.geocode(destination, addressdetails=True, timeout=15)
    if location is None:
        raise ValueError(f"Could not geocode destination: {destination}")

    address = location.raw.get("display_name") if hasattr(location, "raw") else None
    return GeoLocation(
        name=destination,
        latitude=location.latitude,
        longitude=location.longitude,
        formatted_address=address,
    )


def drive_minutes(origin: GeoLocation, lat: float, lon: float) -> int:
    """Estimate drive time in minutes from origin to a point."""
    miles = geodesic((origin.latitude, origin.longitude), (lat, lon)).miles
    hours = miles / _AVERAGE_SPEED_MPH
    return max(1, int(hours * 60))


def within_drive_radius(
    origin: GeoLocation,
    lat: float,
    lon: float,
    max_hours: float = _MAX_DRIVE_HOURS,
) -> bool:
    """Return True if point is within max drive hours of origin."""
    minutes = drive_minutes(origin, lat, lon)
    return minutes <= max_hours * 60


def max_drive_radius_miles(max_hours: float = _MAX_DRIVE_HOURS) -> float:
    return max_hours * _AVERAGE_SPEED_MPH


def bounding_box(
    origin: GeoLocation,
    max_hours: float = _MAX_DRIVE_HOURS,
) -> tuple[float, float, float, float]:
    """Approximate lat/lon bounding box for search APIs."""
    radius_m = max_drive_radius_miles(max_hours) * 1609.34
    lat_delta = radius_m / 111_320
    lon_delta = radius_m / (111_320 * math.cos(math.radians(origin.latitude)))
    return (
        origin.longitude - lon_delta,
        origin.latitude - lat_delta,
        origin.longitude + lon_delta,
        origin.latitude + lat_delta,
    )


def fetch_opentripmap_pois(
    origin: GeoLocation,
    kinds: str = "interesting_places,cultural,historic,natural,foods",
    limit: int = 25,
) -> list[dict]:
    """Fetch POIs from OpenTripMap within drive radius (requires API key)."""
    api_key = os.getenv("OPENTRIPMAP_API_KEY")
    if not api_key:
        return []

    lon_min, lat_min, lon_max, lat_max = bounding_box(origin)
    url = "https://api.opentripmap.com/0.1/en/places/bbox"
    params = {
        "lon_min": lon_min,
        "lat_min": lat_min,
        "lon_max": lon_max,
        "lat_max": lat_max,
        "kinds": kinds,
        "format": "json",
        "limit": limit,
        "apikey": api_key,
    }
    with httpx.Client(timeout=20) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()
