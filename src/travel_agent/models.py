"""Pydantic models for structured trip planning."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TravelStyle(str, Enum):
    RELAXED = "relaxed"
    BALANCED = "balanced"
    PACKED = "packed"


class Interest(str, Enum):
    NATURE = "nature"
    FOOD = "food"
    CULTURE = "culture"
    ADVENTURE = "adventure"
    NIGHTLIFE = "nightlife"
    FAMILY = "family"
    SHOPPING = "shopping"
    BEACH = "beach"


class TripRequest(BaseModel):
    destination: str = Field(description="City, region, or landmark base for the trip")
    start_date: date
    end_date: date
    budget_usd: float = Field(
        gt=0,
        description="Trip budget in USD for food, activities, and transport (accommodation excluded)",
    )
    travelers: int = Field(default=2, ge=1, le=20)
    interests: list[Interest] = Field(default_factory=lambda: [Interest.CULTURE, Interest.FOOD])
    travel_style: TravelStyle = TravelStyle.BALANCED
    dietary_restrictions: list[str] = Field(default_factory=list)
    mobility_notes: str | None = Field(
        default=None,
        description="Accessibility or mobility constraints",
    )
    notes: str | None = Field(default=None, description="Free-form preferences")

    @property
    def trip_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @property
    def daily_budget_usd(self) -> float:
        return self.budget_usd / self.trip_days


class GeoLocation(BaseModel):
    name: str
    latitude: float
    longitude: float
    formatted_address: str | None = None


class Activity(BaseModel):
    name: str
    category: str
    description: str
    latitude: float | None = None
    longitude: float | None = None
    drive_minutes_from_base: int | None = None
    estimated_cost_usd: float | None = None
    duration_hours: float = Field(default=2.0, ge=0.5, le=12)
    rating_hint: str | None = None
    reservation_required: bool = False
    booking_url: str | None = None
    source: str = "research"


class DayPlan(BaseModel):
    day_number: int
    date: date
    theme: str
    morning: list[str] = Field(default_factory=list)
    afternoon: list[str] = Field(default_factory=list)
    evening: list[str] = Field(default_factory=list)
    meals: list[str] = Field(default_factory=list)
    estimated_daily_cost_usd: float | None = None
    driving_notes: str | None = None


class ReservationItem(BaseModel):
    name: str
    category: Literal["restaurant", "landmark", "excursion", "tour", "other"]
    reservation_required: bool
    lead_time_days: int | None = Field(
        default=None,
        description="How far in advance to book",
    )
    booking_url: str | None = None
    booking_platform: str | None = None
    notes: str | None = None
    scheduled_day: int | None = None


class BudgetBreakdown(BaseModel):
    """Trip budget split. Accommodation is out of scope — the client books lodging separately."""

    food_estimate_usd: float
    activities_estimate_usd: float
    transport_estimate_usd: float
    buffer_usd: float
    total_usd: float
    within_budget: bool
    notes: str | None = None


class Itinerary(BaseModel):
    destination: str
    trip_days: int
    summary: str
    days: list[DayPlan]
    day_trips: list[str] = Field(
        default_factory=list,
        description="Excursions within 4-hour drive",
    )
    budget: BudgetBreakdown | None = None
    tips: list[str] = Field(default_factory=list)


class RecommendedActivity(BaseModel):
    name: str
    description: str
    estimated_cost_usd: float | None = None
    duration_hours: float | None = None
    drive_minutes_from_base: int | None = None
    reservation_required: bool = False


class RecommendationResponse(BaseModel):
    """Structured agent reply for one round of the category recommendation loop."""

    category: str = Field(description="Category these recommendations belong to")
    round: int = Field(description="Recommendation round within this category, starting at 1")
    message: str = Field(description="Short conversational intro for the user")
    recommendations: list[RecommendedActivity]
    question: str = Field(description="Follow-up question asking the user to pick or approve")
    categories_remaining: list[str] = Field(
        default_factory=list,
        description="Categories not yet covered after this one",
    )


class ResearchBundle(BaseModel):
    geolocation: GeoLocation
    activities: list[Activity] = Field(default_factory=list)
    reservation_candidates: list[ReservationItem] = Field(default_factory=list)
    raw_notes: str = ""
