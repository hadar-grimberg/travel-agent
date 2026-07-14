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
    """The 8 category groups — one recommendation round per group with checked items."""

    NATURE = "nature"
    FOOD = "food"
    CULTURE = "culture"
    ADVENTURE = "adventure"
    NIGHTLIFE = "nightlife"
    FAMILY = "family"
    SHOPPING = "shopping"
    BEACH = "beach"


class InterestOption(str, Enum):
    """Closed checklist the orchestrator presents to the user.

    Contains the 8 group names (check one to get its whole group), granular
    activity items (mapped internally to POI-provider categories), and a few
    web-search-backed items that have no POI category (free_tours, ...).
    """

    # whole-group selections
    NATURE = "nature"
    FOOD = "food"
    CULTURE = "culture"
    ADVENTURE = "adventure"
    NIGHTLIFE = "nightlife"
    FAMILY = "family"
    SHOPPING = "shopping"
    BEACH = "beach"
    # culture & history
    MUSEUMS = "museums"
    THEATRES_AND_ENTERTAINMENTS = "theatres_and_entertainments"
    URBAN_ENVIRONMENT = "urban_environment"
    HISTORICAL_PLACES = "historical_places"
    FORTIFICATIONS = "fortifications"
    MONUMENTS_AND_MEMORIALS = "monuments_and_memorials"
    ARCHAEOLOGY = "archaeology"
    BURIAL_PLACES = "burial_places"
    HISTORIC_ARCHITECTURE = "historic_architecture"
    SKYSCRAPERS = "skyscrapers"
    BRIDGES = "bridges"
    TOWERS = "towers"
    LIGHTHOUSES = "lighthouses"
    RELIGION = "religion"
    CHURCHES = "churches"
    CATHEDRALS = "cathedrals"
    MOSQUES = "mosques"
    SYNAGOGUES = "synagogues"
    BUDDHIST_TEMPLES = "buddhist_temples"
    HINDU_TEMPLES = "hindu_temples"
    EGYPTIAN_TEMPLES = "egyptian_temples"
    OTHER_TEMPLES = "other_temples"
    MONASTERIES = "monasteries"
    # nature
    NATURE_RESERVES = "nature_reserves"
    GARDENS_AND_PARKS = "gardens_and_parks"
    VIEW_POINTS = "view_points"
    ISLANDS = "islands"
    NATURAL_SPRINGS = "natural_springs"
    GEOLOGICAL_FORMATIONS = "geological_formations"
    WATER = "water"
    GLACIERS = "glaciers"
    # beach
    BEACHES = "beaches"
    # food & drink
    RESTAURANTS = "restaurants"
    CAFES = "cafes"
    BAKERIES = "bakeries"
    FAST_FOOD = "fast_food"
    FOOD_COURTS = "food_courts"
    PICNIC_SITES = "picnic_site"
    # nightlife
    PUBS = "pubs"
    BARS = "bars"
    BIERGARTENS = "biergartens"
    NIGHTCLUBS = "nightclubs"
    CASINO = "casino"
    ALCOHOL = "alcohol"
    HOOKAH = "hookah"
    # adventure & sport
    CLIMBING = "climbing"
    DIVING = "diving"
    SURFING = "surfing"
    KITESURFING = "kitesurfing"
    WINTER_SPORTS = "winter_sports"
    STADIUMS = "stadiums"
    POOLS = "pools"
    # family
    AMUSEMENT_PARKS = "amusement_parks"
    WATER_PARKS = "water_parks"
    ZOOS = "zoos"
    AQUARIUMS = "aquariums"
    CHILDREN_THEATRES = "children_theatres"
    CIRCUSES = "circuses"
    PLANETARIUMS = "planetariums"
    MINIATURE_PARKS = "miniature_parks"
    ROLLER_COASTERS = "roller_coasters"
    FERRIS_WHEELS = "ferris_wheels"
    BATHS_AND_SAUNAS = "baths_and_saunas"
    # shopping
    MALLS = "malls"
    MARKETPLACES = "marketplaces"
    OUTDOOR_SHOPS = "outdoor"
    SUPERMARKETS = "supermarkets"
    CONVENIENCES = "conveniences"
    FISH_STORES = "fish_stores"
    # web-search backed (no POI kind exists for these)
    FREE_TOURS = "free_tours"
    GUIDED_TOURS = "guided_tours"


GROUP_ITEMS: dict[Interest, list[InterestOption]] = {
    # tours are grouped under culture & history
    Interest.CULTURE: [
        InterestOption.CULTURE, InterestOption.MUSEUMS,
        InterestOption.THEATRES_AND_ENTERTAINMENTS, InterestOption.URBAN_ENVIRONMENT,
        InterestOption.HISTORICAL_PLACES, InterestOption.FORTIFICATIONS,
        InterestOption.MONUMENTS_AND_MEMORIALS, InterestOption.ARCHAEOLOGY,
        InterestOption.BURIAL_PLACES, InterestOption.HISTORIC_ARCHITECTURE,
        InterestOption.SKYSCRAPERS, InterestOption.BRIDGES, InterestOption.TOWERS,
        InterestOption.LIGHTHOUSES, InterestOption.RELIGION, InterestOption.CHURCHES,
        InterestOption.CATHEDRALS, InterestOption.MOSQUES, InterestOption.SYNAGOGUES,
        InterestOption.BUDDHIST_TEMPLES, InterestOption.HINDU_TEMPLES,
        InterestOption.EGYPTIAN_TEMPLES, InterestOption.OTHER_TEMPLES,
        InterestOption.MONASTERIES, InterestOption.FREE_TOURS, InterestOption.GUIDED_TOURS,
    ],
    Interest.NATURE: [
        InterestOption.NATURE, InterestOption.NATURE_RESERVES,
        InterestOption.GARDENS_AND_PARKS, InterestOption.VIEW_POINTS,
        InterestOption.ISLANDS, InterestOption.NATURAL_SPRINGS,
        InterestOption.GEOLOGICAL_FORMATIONS, InterestOption.WATER,
        InterestOption.GLACIERS,
    ],
    Interest.BEACH: [InterestOption.BEACH, InterestOption.BEACHES],
    # dietary dining is grouped under food
    Interest.FOOD: [
        InterestOption.FOOD, InterestOption.RESTAURANTS, InterestOption.CAFES,
        InterestOption.BAKERIES, InterestOption.FAST_FOOD, InterestOption.FOOD_COURTS,
        InterestOption.PICNIC_SITES,
    ],
    Interest.NIGHTLIFE: [
        InterestOption.NIGHTLIFE, InterestOption.PUBS, InterestOption.BARS,
        InterestOption.BIERGARTENS, InterestOption.NIGHTCLUBS, InterestOption.CASINO,
        InterestOption.ALCOHOL, InterestOption.HOOKAH,
    ],
    Interest.ADVENTURE: [
        InterestOption.ADVENTURE, InterestOption.CLIMBING, InterestOption.DIVING,
        InterestOption.SURFING, InterestOption.KITESURFING, InterestOption.WINTER_SPORTS,
        InterestOption.STADIUMS, InterestOption.POOLS,
    ],
    Interest.FAMILY: [
        InterestOption.FAMILY, InterestOption.AMUSEMENT_PARKS, InterestOption.WATER_PARKS,
        InterestOption.ZOOS, InterestOption.AQUARIUMS, InterestOption.CHILDREN_THEATRES,
        InterestOption.CIRCUSES, InterestOption.PLANETARIUMS, InterestOption.MINIATURE_PARKS,
        InterestOption.ROLLER_COASTERS, InterestOption.FERRIS_WHEELS,
        InterestOption.BATHS_AND_SAUNAS,
    ],
    Interest.SHOPPING: [
        InterestOption.SHOPPING, InterestOption.MALLS, InterestOption.MARKETPLACES,
        InterestOption.OUTDOOR_SHOPS, InterestOption.SUPERMARKETS,
        InterestOption.CONVENIENCES, InterestOption.FISH_STORES,
    ],
}

# checklist item -> its category group (round)
INTEREST_GROUP: dict[InterestOption, Interest] = {
    item: group for group, items in GROUP_ITEMS.items() for item in items
}

# items with no POI category — researched via web search / LLM knowledge
NON_POI_OPTIONS: frozenset[InterestOption] = frozenset({
    InterestOption.FREE_TOURS,
    InterestOption.GUIDED_TOURS,
})


class TripRequest(BaseModel):
    destination: str = Field(description="City, region, or landmark base for the trip")
    start_date: date
    end_date: date
    budget_usd: float = Field(
        gt=0,
        description="Trip budget in USD for food, activities, and transport (accommodation excluded)",
    )
    travelers: int = Field(default=2, ge=1, le=20)
    interests: list[InterestOption] = Field(
        default_factory=lambda: [InterestOption.CULTURE, InterestOption.FOOD],
        description="Checked items from the closed interest checklist",
    )
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


class FeedbackPayload(BaseModel):
    """Structured user feedback for one recommendation round, relayed by the orchestrator."""

    selected: list[str] = Field(
        default_factory=list,
        description="Names of recommended activities the user wants to keep",
    )
    approve: bool = Field(
        default=False,
        description="Current category is done — move to the next one",
    )
    finish: bool = Field(
        default=False,
        description="Skip remaining categories and build the itinerary now",
    )
    quit: bool = Field(
        default=False,
        description="End the session without an itinerary",
    )


class ResearchBundle(BaseModel):
    geolocation: GeoLocation
    activities: list[Activity] = Field(default_factory=list)
    reservation_candidates: list[ReservationItem] = Field(default_factory=list)
    raw_notes: str = ""
