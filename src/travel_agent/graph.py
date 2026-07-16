"""LangGraph workflow for the travel agent."""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from travel_agent.config import PROVIDER_SERVERLESS, Settings
from travel_agent.models import (
    INTEREST_GROUP,
    Activity,
    BudgetBreakdown,
    DayPlan,
    FeedbackPayload,
    GeoLocation,
    Itinerary,
    RecommendationResponse,
    RecommendedActivity,
    ResearchBundle,
    TripRequest,
)
from travel_agent.prompts import BASE_PERSONA, phase_prompt
from travel_agent.skills.loader import load_skills_for_phase
from travel_agent.state import TravelAgentState
from travel_agent.tools.activities import discover_activities
from travel_agent.tools.budget import estimate_budget_breakdown
from travel_agent.tools.geo import geocode_destination
from travel_agent.tools.langchain_tools import RESEARCH_TOOLS
from travel_agent.tools.reservations import build_reservation_plan


def _token_model(settings: Settings, max_tokens=4096, reasoning_effort="low") -> ChatOpenAI:
    """Nebius per-token API."""
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.nebius_base_url,
        api_key=settings.nebius_api_key,
        temperature=0.3,
        max_tokens=max_tokens,
        # gpt-oss reasoning models: pass provider-specific params via extra_body
        # so they land in the raw request body regardless of ChatOpenAI's schema.
        extra_body={"reasoning_effort": reasoning_effort},
    )


def _serverless_model(settings: Settings, max_tokens=4096) -> ChatOpenAI:
    """Nebius serverless dedicated endpoint (OpenAI-compatible).

    The endpoint ID is passed as the model name; reasoning_effort is not sent —
    dedicated endpoints serve a fixed model that may not accept it.
    """
    return ChatOpenAI(
        model=settings.nebius_endpoint_id,
        base_url=settings.nebius_endpoint_url,
        api_key=settings.nebius_api_key or "not-needed",
        temperature=0.3,
        max_tokens=max_tokens,
    )


def _model(settings: Settings, max_tokens=4096, reasoning_effort="low") -> ChatOpenAI:
    """Construct the configured LLM backend (never called in mock mode)."""
    if settings.resolved_provider == PROVIDER_SERVERLESS:
        return _serverless_model(settings, max_tokens=max_tokens)
    return _token_model(settings, max_tokens=max_tokens, reasoning_effort=reasoning_effort)


def _parse_trip_request(text: str, settings: Settings) -> TripRequest:
    """Extract structured trip parameters from natural language via LLM.

    Uses a plain prompt + JSON parser instead of with_structured_output().
    Forced tool-calling combined with this model's reasoning-token behavior
    causes it to never emit a stop token, burning the full max_tokens budget
    and raising LengthFinishReasonError regardless of how high max_tokens is set.
    """
    parser = JsonOutputParser(pydantic_object=TripRequest)

    system = (
        "Extract trip planning parameters from the user message. "
        "If dates are relative (e.g. 'next month'), infer reasonable ISO dates. "
        "Default interests to culture and food if unspecified.\n\n"
        f"{parser.get_format_instructions()}"
    )

    llm = _model(settings, max_tokens=1500, reasoning_effort="low")
    raw = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=text),
        ]
    )
    parsed = parser.parse(raw.content)
    return TripRequest.model_validate(parsed)


def _heuristic_trip_request(text: str) -> TripRequest:
    """Keyless fallback trip parse — no LLM, sensible defaults.

    Used only in mock mode when free text (not a structured trip) is supplied;
    the orchestrator normally passes a structured trip so this never runs.
    """
    destination = text.strip().splitlines()[0].split(",")[0].strip() or "Unknown"
    start = date.today() + timedelta(days=30)
    return TripRequest(
        destination=destination,
        start_date=start,
        end_date=start + timedelta(days=3),
        budget_usd=1500.0,
    )


def _categories_for_trip(trip: TripRequest) -> list[str]:
    """One round per category GROUP of the checked items, in checklist order —
    granular picks (buddhist_temples, view_points, ...) share their group's round."""
    return list(dict.fromkeys(INTEREST_GROUP[opt].value for opt in trip.interests)) or ["culture"]


def _mock_activities(trip: TripRequest) -> list[Activity]:
    """Deterministic placeholder activities, three per chosen category group."""
    activities: list[Activity] = []
    for category in _categories_for_trip(trip):
        for i in range(1, 4):
            activities.append(
                Activity(
                    name=f"{trip.destination} {category} highlight {i}",
                    category=category,
                    description=f"A well-loved {category} activity in {trip.destination}. "
                    "(Offline mock data — configure an LLM provider for real recommendations.)",
                    estimated_cost_usd=float(15 * i),
                    duration_hours=2.0,
                    reservation_required=(i == 1),
                    source="mock",
                )
            )
    return activities


def make_intake_node(settings: Settings):
    def intake_node(state: TravelAgentState) -> dict:
        """Parse the latest user message into a TripRequest.

        A structured trip_request (the orchestrator's normal path) is used as-is.
        Free text is parsed by the LLM, or heuristically in keyless mock mode.
        """
        trip = state.get("trip_request")
        if trip is None:
            user_text = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break
            trip = (
                _heuristic_trip_request(user_text)
                if settings.use_mock
                else _parse_trip_request(user_text, settings)
            )

        skill_names, _ = load_skills_for_phase("research")
        return {
            "trip_request": trip,
            "phase": "research",
            "active_skill_names": skill_names,
        }

    return intake_node


def make_bootstrap_research_node(settings: Settings):
    def bootstrap_research_node(state: TravelAgentState) -> dict:
        """Deterministic research pass: geocode, activities, budget — feeds the LLM.

        In mock mode this stays fully offline: a stub geolocation and placeholder
        activities, no Nominatim/Geoapify network calls.
        """
        trip = state["trip_request"]
        assert trip is not None

        if settings.use_mock:
            origin = GeoLocation(name=trip.destination, latitude=0.0, longitude=0.0)
            activities = _mock_activities(trip)
        else:
            origin = geocode_destination(trip.destination)
            activities = discover_activities(origin, trip)

        budget = estimate_budget_breakdown(trip, activities)
        reservations = build_reservation_plan(activities, trip.destination)

        bundle = ResearchBundle(
            geolocation=origin,
            activities=activities,
            reservation_candidates=reservations,
            raw_notes=(
                f"Found {len(activities)} activities within 4h drive of {trip.destination}. "
                f"Trip: {trip.trip_days} days, ${trip.budget_usd:.0f} budget, "
                f"style={trip.travel_style.value}, interests={[i.value for i in trip.interests]}."
            ),
        )
        return {
            "research": bundle,
            "categories": _categories_for_trip(trip),
            "current_category_index": 0,
            "category_refinement_count": 0,
        }

    return bootstrap_research_node


def _agent_node_factory(phase: str, settings: Settings, tools: list | None = None):
    """Create a phase-specific agent node (ReAct when tools provided).

    In mock mode the node is a no-op that emits a plain summary message (no tool
    calls), so tools_condition routes straight past the ReAct loop.
    """

    def node(state: TravelAgentState) -> dict:
        skill_names, skills_text = load_skills_for_phase(phase)

        if settings.use_mock:
            return {
                "messages": [AIMessage(content=f"(mock) {phase} phase complete.")],
                "phase": phase,
                "active_skill_names": skill_names,
            }

        trip = state["trip_request"]
        research = state.get("research")
        system = phase_prompt(phase, skills_text)

        context_parts = []
        if trip:
            context_parts.append(f"Trip request:\n{trip.model_dump_json()}")
        if research:
            context_parts.append(
                f"Research bundle:\n{research.model_dump_json()[:12000]}"
            )
        if state.get("itinerary"):
            context_parts.append(
                f"Draft itinerary:\n{state['itinerary'].model_dump_json()[:8000]}"
            )

        llm = _model(settings, max_tokens=4096, reasoning_effort="medium")
        if tools:
            llm = llm.bind_tools(tools)

        response = llm.invoke(
            [
                SystemMessage(content=system),
                SystemMessage(content="\n\n".join(context_parts)),
                *state["messages"],
            ]
        )
        return {
            "messages": [response],
            "phase": phase,
            "active_skill_names": skill_names,
        }

    return node


def merge_research_notes_node(state: TravelAgentState) -> dict:
    """Fold the research agent's final summary into the ResearchBundle.

    Web-search findings (free tours, dietary dining, booking info) only exist
    in the research agent's closing message; recommendation rounds read the
    bundle, so persist the summary there instead of relying on it surviving
    the rolling chat window.
    """
    research = state.get("research")
    summary = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if content.strip():
                summary = content
                break
    if research is None or not summary:
        return {}
    merged = research.model_copy(
        update={"raw_notes": f"{research.raw_notes}\n\nResearch agent findings:\n{summary[:4000]}"}
    )
    return {"research": merged}


# Recommendation rounds allowed per category: 1 initial + up to 5 refinements
MAX_ROUNDS_PER_CATEGORY = 6

# Legacy free-text keywords — only used to map CLI debug input onto the
# structured FeedbackPayload; the orchestrator sends the payload directly.
APPROVE_PHRASES = [
    "approve", "approved", "next category", "next", "move on",
    "looks good", "that's enough", "done with this",
]
FINISH_PHRASES = [
    "create itinerary", "build itinerary", "make itinerary",
    "finalize", "book it", "ready to book",
    "create the plan", "build the plan", "make the plan",
]
QUIT_WORDS = {"exit", "quit", "bye", "goodbye", "stop"}


def _payload_from_text(text: str) -> FeedbackPayload:
    """Map a legacy free-text reply (standalone CLI debugging) onto FeedbackPayload."""
    lowered = text.lower().strip()
    if any(phrase in lowered for phrase in FINISH_PHRASES):
        return FeedbackPayload(finish=True)
    if lowered in QUIT_WORDS:
        return FeedbackPayload(quit=True)
    if any(phrase in lowered for phrase in APPROVE_PHRASES):
        return FeedbackPayload(approve=True)
    return FeedbackPayload(selected=[text] if text.strip() else [])


def _feedback_to_text(payload: FeedbackPayload, category: str) -> str:
    """Render the structured feedback as a readable chat message for the LLM."""
    parts = []
    if payload.selected:
        parts.append("I want to keep these activities: " + ", ".join(payload.selected) + ".")
    if payload.finish:
        parts.append("Please build the itinerary now.")
    elif payload.quit:
        parts.append("I'm done — no itinerary needed.")
    elif payload.approve:
        parts.append(f"I'm done with the {category} category, move on.")
    elif payload.selected:
        parts.append("Show me more like these.")
    else:
        parts.append("None of these appeal to me — show me different options.")
    return " ".join(parts)


def _current_category(state: TravelAgentState) -> str:
    categories = state.get("categories") or ["general"]
    index = min(state.get("current_category_index", 0), len(categories) - 1)
    return categories[index]


def _mock_recommendation(
    state: TravelAgentState, category: str, round_number: int, remaining: list[str]
) -> RecommendationResponse:
    """Build a RecommendationResponse from research activities, no LLM."""
    trip = state["trip_request"]
    research = state.get("research")
    all_activities = research.activities if research else []
    picked = {
        p.split("] ", 1)[-1]
        for p in (state.get("user_preferences") or [])
        if p.startswith(f"[{category}]")
    }
    pool = [a for a in all_activities if a.category == category and a.name not in picked]
    if not pool:  # category not represented in research — offer anything left
        pool = [a for a in all_activities if a.name not in picked] or all_activities

    recommendations = [
        RecommendedActivity(
            name=a.name,
            description=a.description,
            estimated_cost_usd=a.estimated_cost_usd,
            duration_hours=a.duration_hours,
            drive_minutes_from_base=a.drive_minutes_from_base,
            reservation_required=a.reservation_required,
        )
        for a in pool[:5]
    ]
    return RecommendationResponse(
        category=category,
        round=round_number,
        message=f"Here are some {category} options in {trip.destination}.",
        recommendations=recommendations,
        question=(
            "Select the ones you'd like to keep, ask for more like these, "
            "or approve to move on."
        ),
        categories_remaining=remaining,
    )


def make_generate_recommendations_node(settings: Settings):
    def generate_recommendations_node(state: TravelAgentState) -> dict:
        """Recommend activities for the current category, as structured JSON.

        First round presents the category's options from research; later rounds
        recommend more activities similar to what the user just picked. In mock
        mode the response is built deterministically from research (no LLM).
        """
        trip = state["trip_request"]
        research = state.get("research")
        messages = state.get("messages", [])
        assert trip is not None

        categories = state.get("categories") or ["general"]
        index = state.get("current_category_index", 0)
        category = _current_category(state)
        remaining = categories[index + 1:]
        round_number = state.get("category_refinement_count", 0) + 1

        if settings.use_mock:
            recommendation = _mock_recommendation(state, category, round_number, remaining)
            text = recommendation.model_dump_json(indent=2)
            return {
                "current_recommendations": text,
                "messages": [AIMessage(content=text)],
                "phase": "recommendation",
                "category_refinement_count": round_number,
            }

        parser = JsonOutputParser(pydantic_object=RecommendationResponse)
        llm = _model(settings, max_tokens=2000, reasoning_effort="low")
        research_json = research.model_dump_json()[:6000] if research else "{}"

        # Granular checklist items the user checked within this category group
        checked_items = [
            opt.value for opt in trip.interests if INTEREST_GROUP[opt].value == category
        ]

        system = (
            BASE_PERSONA + "\n\n"
            f"You are walking the user through activity options one category at a time. "
            f"Current category: '{category}' (round {round_number} of {MAX_ROUNDS_PER_CATEGORY}). "
            f"The user specifically checked these interests in this category: {checked_items}. "
            f"Remaining categories after this one: {remaining or 'none'}.\n"
            "Recommend only activities from the current category, prioritizing the checked "
            "interests. Checked items like free_tours and guided_tours have no POI data — "
            "cover them from the research notes and your knowledge of the destination. "
            "When recommending food or dining, strictly honor the trip's dietary_restrictions "
            "(e.g. vegan, vegetarian, kosher). "
            "In the question field, ask the user to select the activities they want to keep, "
            "request more like these, or approve the category to move on.\n\n"
            + parser.get_format_instructions()
        )

        if round_number == 1:
            human = (
                f"Trip: {trip.model_dump_json()}\n\n"
                f"Research findings:\n{research_json}\n\n"
                f"Present the '{category}' activities from the research as inviting options."
            )
        else:
            picks = [
                p for p in (state.get("user_preferences") or [])
                if p.startswith(f"[{category}]")
            ]
            human = (
                f"Trip: {trip.model_dump_json()}\n\n"
                f"Research findings:\n{research_json}\n\n"
                f"What the user picked in '{category}' so far:\n"
                + "\n".join(f"- {p}" for p in picks)
                + f"\n\nRecommend more '{category}' activities similar to their picks, "
                "drawing on the research and your knowledge of the destination. "
                "Do not repeat activities already discussed."
            )

        # Include recent conversation for continuity (last 8 turns)
        recent = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))][-8:]

        raw = llm.invoke([SystemMessage(content=system), *recent, HumanMessage(content=human)])
        parsed = parser.parse(raw.content)
        recommendation = RecommendationResponse.model_validate(
            {**parsed, "category": category, "round": round_number,
             "categories_remaining": remaining}
        )
        text = recommendation.model_dump_json(indent=2)

        return {
            "current_recommendations": text,
            "messages": [AIMessage(content=text)],
            "phase": "recommendation",
            "category_refinement_count": round_number,
        }

    return generate_recommendations_node


def wait_for_user_node(state: TravelAgentState) -> dict:
    """Display the current recommendations and pause for structured feedback.

    The orchestrator resumes with a FeedbackPayload dict ({selected, approve,
    finish, quit}); plain strings (standalone CLI debugging) are mapped onto
    the same payload via keyword matching. Selections are always recorded as
    preferences — including when sent together with approve.
    """
    recommendations = state.get("current_recommendations", "")
    category = _current_category(state)

    user_input = interrupt({
        "display": recommendations,
        "prompt": (
            "Reply with a JSON feedback object: 'selected' = names of the "
            f"'{category}' activities to keep, 'approve' = category done, "
            "'finish' = build the itinerary now, 'quit' = end without one."
        ),
    })
    if isinstance(user_input, dict):
        payload = FeedbackPayload.model_validate(user_input)
    else:
        payload = _payload_from_text(str(user_input))

    text = _feedback_to_text(payload, category)
    updates: dict = {
        "last_user_input": text,
        "last_feedback": payload.model_dump(),
        "messages": [HumanMessage(content=text)],
    }
    if payload.selected:
        updates["user_preferences"] = (state.get("user_preferences") or []) + [
            f"[{category}] {name}" for name in payload.selected
        ]
    return updates


def should_continue_chat(
    state: TravelAgentState,
) -> Literal["generate_recommendations", "advance_category", "synthesize_itinerary", "__end__"]:
    """Route on the structured feedback flags:
    - finish → build the full itinerary immediately
    - quit   → end without an itinerary
    - approve, or refinement limit reached → advance to the next category
    - else   → refine recommendations within the current category
    """
    feedback = state.get("last_feedback") or {}

    if feedback.get("finish"):
        return "synthesize_itinerary"
    if feedback.get("quit"):
        return "__end__"
    if (
        feedback.get("approve")
        or state.get("category_refinement_count", 0) >= MAX_ROUNDS_PER_CATEGORY
    ):
        return "advance_category"
    return "generate_recommendations"


def advance_category_node(state: TravelAgentState) -> dict:
    """Move to the next category and reset the per-category round counter."""
    return {
        "current_category_index": state.get("current_category_index", 0) + 1,
        "category_refinement_count": 0,
    }


def has_more_categories(
    state: TravelAgentState,
) -> Literal["generate_recommendations", "synthesize_itinerary"]:
    """After advancing: keep recommending if categories remain, else build the itinerary."""
    if state.get("current_category_index", 0) < len(state.get("categories") or []):
        return "generate_recommendations"
    return "synthesize_itinerary"


def _mock_itinerary(state: TravelAgentState) -> Itinerary:
    """Build a simple day-by-day itinerary from the confirmed picks, no LLM."""
    trip = state["trip_request"]
    research = state.get("research")
    picks = [p.split("] ", 1)[-1] for p in (state.get("user_preferences") or [])]

    days: list[DayPlan] = []
    for i in range(trip.trip_days):
        slice_ = picks[i::trip.trip_days] if picks else []
        days.append(
            DayPlan(
                day_number=i + 1,
                date=trip.start_date + timedelta(days=i),
                theme=f"Day {i + 1} in {trip.destination}",
                morning=slice_[:1],
                afternoon=slice_[1:2],
                evening=slice_[2:3],
                meals=[],
            )
        )

    budget: BudgetBreakdown | None = (
        estimate_budget_breakdown(trip, research.activities) if research else None
    )
    return Itinerary(
        destination=trip.destination,
        trip_days=trip.trip_days,
        summary=(
            f"A {trip.trip_days}-day {trip.destination} plan built from your selected "
            "activities. (Offline mock itinerary — configure an LLM provider for a "
            "richly detailed plan.)"
        ),
        days=days,
        day_trips=[],
        budget=budget,
        tips=["This is offline mock output; set an LLM provider for full planning."],
    )


def make_synthesize_itinerary_node(settings: Settings):
    def synthesize_itinerary_node(state: TravelAgentState) -> dict:
        """Build the full structured itinerary from the user's confirmed preferences."""
        trip = state["trip_request"]
        research = state.get("research")
        user_preferences = state.get("user_preferences") or []
        assert trip is not None

        if settings.use_mock:
            return {"itinerary": _mock_itinerary(state), "phase": "planning"}

        _, skills_text = load_skills_for_phase("planning")
        parser = JsonOutputParser(pydantic_object=Itinerary)
        llm = _model(settings, max_tokens=8000, reasoning_effort="medium")

        research_json = research.model_dump_json() if research else "{}"
        preferences_text = "\n".join(f"- {p}" for p in user_preferences)

        system = (
            phase_prompt("planning", skills_text)
            + "\n\nOutput a complete Itinerary object matching the schema below.\n\n"
            + parser.get_format_instructions()
        )
        human = (
            f"Trip:\n{trip.model_dump_json()}\n\n"
            f"Research:\n{research_json}\n\n"
            f"Activities and preferences the user confirmed during the chat:\n{preferences_text}\n\n"
            "Build a complete day-by-day itinerary that reflects exactly what the user chose. "
            "Add timings, specific restaurant names, practical tips, driving notes, and a budget breakdown."
        )

        raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        parsed = parser.parse(raw.content)
        itinerary = Itinerary.model_validate(parsed)

        if research and not itinerary.budget:
            itinerary = itinerary.model_copy(
                update={"budget": estimate_budget_breakdown(trip, research.activities)}
            )

        return {"itinerary": itinerary, "phase": "planning"}

    return synthesize_itinerary_node


def enrich_reservations_node(state: TravelAgentState) -> dict:
    """Merge tool-based reservation plan into itinerary context."""
    trip = state["trip_request"]
    itinerary = state.get("itinerary")
    assert trip is not None and itinerary is not None

    venue_names: list[str] = []
    for day in itinerary.days:
        venue_names.extend(day.meals)
        venue_names.extend(day.morning + day.afternoon + day.evening)
    venue_names.extend(itinerary.day_trips)

    reservations = build_reservation_plan([], trip.destination, extra_venues=venue_names[:30])
    research = state.get("research")
    if research:
        merged = {r.name.lower(): r for r in research.reservation_candidates}
        for item in reservations:
            merged[item.name.lower()] = item
        research = research.model_copy(
            update={"reservation_candidates": list(merged.values())}
        )
    return {"research": research, "phase": "reservations"}


def should_continue_research(state: TravelAgentState) -> Literal["tools", "plan"]:
    """Route after research agent: tools loop or proceed to recommendations."""
    route = tools_condition(state)
    if route == "tools":
        return "tools"
    return "plan"


def build_graph(checkpointer=None, settings: Settings | None = None):
    """Compile the travel agent graph.

    After research the graph walks the user through activities one category at
    a time: it presents options, refines with similar suggestions after each
    pick (up to 5 refinements per category), and advances when the user
    approves. Agent replies are structured JSON (RecommendationResponse). Once
    all categories are covered — or the user asks to finish — it builds a full
    itinerary with reservations. A checkpointer is required.

    ``settings`` selects the LLM backend; when it resolves to the keyless mock
    provider the graph runs fully offline with deterministic output.
    """
    settings = settings or Settings.from_env()
    graph = StateGraph(TravelAgentState)

    graph.add_node("intake", make_intake_node(settings))
    graph.add_node("bootstrap_research", make_bootstrap_research_node(settings))
    graph.add_node("research_agent", _agent_node_factory("research", settings, RESEARCH_TOOLS))
    graph.add_node("research_tools", ToolNode(RESEARCH_TOOLS))
    graph.add_node("merge_research_notes", merge_research_notes_node)
    graph.add_node("generate_recommendations", make_generate_recommendations_node(settings))
    graph.add_node("wait_for_user", wait_for_user_node)
    graph.add_node("advance_category", advance_category_node)
    graph.add_node("synthesize_itinerary", make_synthesize_itinerary_node(settings))
    graph.add_node("enrich_reservations", enrich_reservations_node)
    graph.add_node(
        "reservations_agent", _agent_node_factory("reservations", settings, RESEARCH_TOOLS)
    )
    graph.add_node("reservation_tools", ToolNode(RESEARCH_TOOLS))
    graph.add_node("finalize_agent", _agent_node_factory("finalize", settings))

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "bootstrap_research")
    graph.add_edge("bootstrap_research", "research_agent")

    graph.add_conditional_edges(
        "research_agent",
        should_continue_research,
        {"tools": "research_tools", "plan": "merge_research_notes"},
    )
    graph.add_edge("research_tools", "research_agent")
    graph.add_edge("merge_research_notes", "generate_recommendations")

    # Category-by-category recommendation loop — refine within a category until
    # the user approves it (or the round limit is hit), then advance; after the
    # last category the itinerary is built automatically.
    graph.add_edge("generate_recommendations", "wait_for_user")
    graph.add_conditional_edges(
        "wait_for_user",
        should_continue_chat,
        {
            "generate_recommendations": "generate_recommendations",
            "advance_category": "advance_category",
            "synthesize_itinerary": "synthesize_itinerary",
            "__end__": END,
        },
    )
    graph.add_conditional_edges(
        "advance_category",
        has_more_categories,
        {
            "generate_recommendations": "generate_recommendations",
            "synthesize_itinerary": "synthesize_itinerary",
        },
    )

    # Post-confirmation: build full itinerary and handle reservations
    graph.add_edge("synthesize_itinerary", "enrich_reservations")
    graph.add_edge("enrich_reservations", "reservations_agent")
    graph.add_conditional_edges(
        "reservations_agent",
        tools_condition,
        {"tools": "reservation_tools", "__end__": "finalize_agent"},
    )
    graph.add_edge("reservation_tools", "reservations_agent")
    graph.add_edge("finalize_agent", END)

    return graph.compile(checkpointer=checkpointer)


def run_travel_agent(user_message: str, get_user_input=None) -> dict:
    """Run the travel agent as an open-ended recommendation chat.

    Args:
        user_message: Natural language trip request to start the conversation.
        get_user_input: Optional callable(interrupt_data: dict) -> str.
            Receives {"display": str, "prompt": str} and must return the user's
            reply. Defaults to reading from stdin interactively.
            Pass ``lambda data: "exit"`` to run headless / in tests.
    """
    checkpointer = MemorySaver()
    app = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print("Researching your trip...\n")
    result = app.invoke({"messages": [HumanMessage(content=user_message)]}, config)

    # Chat loop — runs until the user says exit/quit/bye or the graph reaches END
    while result.get("__interrupt__"):
        interrupt_data = result["__interrupt__"][0].value

        if get_user_input is not None:
            user_response = get_user_input(interrupt_data)
        else:
            print("\n" + interrupt_data["display"])
            print("\n" + interrupt_data["prompt"])
            print("(Type 'exit' to end the conversation.)\n")
            user_response = input("You: ").strip() or "Tell me more"

        result = app.invoke(Command(resume=user_response), config)

    return result


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    result = run_travel_agent(
        "Kyoto, April 3-9 2026, $2800 total, 2 travelers, temples and kaiseki, balanced pace"
    )
    print(result["itinerary"].model_dump_json(indent=2))