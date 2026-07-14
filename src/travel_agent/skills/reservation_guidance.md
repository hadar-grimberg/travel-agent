# Reservation Guidance Skill

Popular landmarks, restaurants, and excursions **require advance booking**. Your reputation depends on flagging this early.

## When reservation is required (default assume YES for)
- Fine dining and trending restaurants
- Major museums with timed entry
- Popular tours, wine tastings, boat trips
- National parks / observatories with quota systems
- Weekend brunch hotspots in busy cities

## Workflow
1. For every named restaurant, landmark, or tour in the itinerary, call `check_reservation_requirements_tool`.
2. Batch remaining venues with `build_reservation_checklist_tool`.
3. Prefer **official venue websites** over aggregators when linking.
4. State recommended **lead time** (e.g., "book 2–3 weeks ahead for Saturday dinner").

## Platforms (fallback search links)
- Restaurants: OpenTable, Resy, Tock, or venue site
- Tours/excursions: official operator, Viator, GetYourGuide
- Museums/landmarks: official timed-entry page, Tiqets

## Client deliverable
Provide a **Reservation Action List** sorted by urgency (longest lead time first) with:
- Venue name, date/day, booking URL, platform, and backup option.
