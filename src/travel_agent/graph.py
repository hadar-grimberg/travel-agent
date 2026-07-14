"""LangGraph workflow for the travel agent."""

from __future__ import annotations

import os
import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from travel_agent.models import (
    INTEREST_GROUP,
    Itinerary,
    RecommendationResponse,
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


def _model(max_tokens=4096, reasoning_effort="low") -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("TRAVEL_AGENT_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("NEBIUS_BASE_URL"),
        api_key=os.getenv("NEBIUS_API_KEY"),
        temperature=0.3,
        max_tokens=max_tokens,
        # gpt-oss reasoning models: pass provider-specific params via extra_body
        # so they land in the raw request body regardless of ChatOpenAI's schema.
        extra_body={"reasoning_effort": reasoning_effort},
    )


def _parse_trip_request(text: str) -> TripRequest:
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

    llm = _model(max_tokens=1500, reasoning_effort="low")
    raw = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=text),
        ]
    )
    parsed = parser.parse(raw.content)
    return TripRequest.model_validate(parsed)


def intake_node(state: TravelAgentState) -> dict:
    """Parse the latest user message into a TripRequest.

    If the caller already supplied a structured trip_request (e.g. the
    orchestrator via ActivitiesAgent), skip the LLM parse and use it as-is.
    """
    trip = state.get("trip_request")
    if trip is not None:
        skill_names, _ = load_skills_for_phase("research")
        return {
            "trip_request": trip,
            "phase": "research",
            "active_skill_names": skill_names,
        }

    messages = state["messages"]
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    trip = _parse_trip_request(user_text)
    skill_names, _ = load_skills_for_phase("research")
    return {
        "trip_request": trip,
        "phase": "research",
        "active_skill_names": skill_names,
    }


def bootstrap_research_node(state: TravelAgentState) -> dict:
    """Deterministic research pass: geocode, activities, budget — feeds the LLM."""
    trip = state["trip_request"]
    assert trip is not None

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
    # One round per category GROUP of the checked items, in checklist order —
    # granular picks (buddhist_temples, view_points, ...) share their group's round.
    categories = list(
        dict.fromkeys(INTEREST_GROUP[opt].value for opt in trip.interests)
    ) or ["culture"]
    return {
        "research": bundle,
        "categories": categories,
        "current_category_index": 0,
        "category_refinement_count": 0,
    }


def _agent_node_factory(phase: str, tools: list | None = None):
    """Create a phase-specific agent node (ReAct when tools provided)."""

    def node(state: TravelAgentState) -> dict:
        trip = state["trip_request"]
        research = state.get("research")
        skill_names, skills_text = load_skills_for_phase(phase)
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

        llm = _model(max_tokens=4096, reasoning_effort="medium")
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


research_agent = _agent_node_factory("research", RESEARCH_TOOLS)
reservations_agent = _agent_node_factory("reservations", RESEARCH_TOOLS)
finalize_agent = _agent_node_factory("finalize")


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


def _current_category(state: TravelAgentState) -> str:
    categories = state.get("categories") or ["general"]
    index = min(state.get("current_category_index", 0), len(categories) - 1)
    return categories[index]


def generate_recommendations_node(state: TravelAgentState) -> dict:
    """Recommend activities for the current category, as structured JSON.

    First round presents the category's options from research; later rounds
    recommend more activities similar to what the user just picked.
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

    parser = JsonOutputParser(pydantic_object=RecommendationResponse)
    llm = _model(max_tokens=2000, reasoning_effort="low")
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
        "interests. Checked items like free_tours, guided_tours, have no POI data — "
        "cover them from the research notes and your knowledge of the destination. "
        "When recommending food or dining, strictly honor the trip's dietary_restrictions "
        "(e.g. vegan, vegetarian, kosher). "
        "In the question field, ask the user to pick favorites, request more like these, "
        "or say 'approve' to move to the next category.\n\n"
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
        {**parsed, "category": category, "round": round_number, "categories_remaining": remaining}
    )
    text = recommendation.model_dump_json(indent=2)

    return {
        "current_recommendations": text,
        "messages": [AIMessage(content=text)],
        "phase": "recommendation",
        "category_refinement_count": round_number,
    }


def wait_for_user_node(state: TravelAgentState) -> dict:
    """Display the current recommendations and pause for user input."""
    recommendations = state.get("current_recommendations", "")
    category = _current_category(state)

    user_input = interrupt({
        "display": recommendations,
        "prompt": (
            f"Pick your favorite '{category}' activities, ask for more like these, "
            "or say 'approve' to move to the next category:"
        ),
    })
    user_input_str = str(user_input)

    updates: dict = {
        "last_user_input": user_input_str,
        "messages": [HumanMessage(content=user_input_str)],
    }

    # Record actual picks only — approval/quit commands aren't preferences
    lowered = user_input_str.lower().strip()
    is_command = (
        any(p in lowered for p in APPROVE_PHRASES + FINISH_PHRASES)
        or lowered in QUIT_WORDS
    )
    if not is_command:
        updates["user_preferences"] = (state.get("user_preferences") or []) + [
            f"[{category}] {user_input_str}"
        ]
    return updates


def should_continue_chat(
    state: TravelAgentState,
) -> Literal["generate_recommendations", "advance_category", "synthesize_itinerary", "__end__"]:
    """Route after each user reply:
    - finish phrase → build the full itinerary immediately
    - exit          → end without an itinerary
    - approval, or refinement limit reached → advance to the next category
    - else          → refine recommendations within the current category
    """
    last = (state.get("last_user_input") or "").lower().strip()

    if any(phrase in last for phrase in FINISH_PHRASES):
        return "synthesize_itinerary"
    if any(word in last for word in QUIT_WORDS):
        return "__end__"
    if (
        any(phrase in last for phrase in APPROVE_PHRASES)
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


def synthesize_itinerary_node(state: TravelAgentState) -> dict:
    """Build the full structured itinerary from the user's confirmed activity preferences."""
    trip = state["trip_request"]
    research = state.get("research")
    user_preferences = state.get("user_preferences") or []
    assert trip is not None

    _, skills_text = load_skills_for_phase("planning")
    parser = JsonOutputParser(pydantic_object=Itinerary)
    llm = _model(max_tokens=8000, reasoning_effort="medium")

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


def build_graph(checkpointer=None):
    """Compile the travel agent graph.

    After research the graph walks the user through activities one category at
    a time: it presents options, refines with similar suggestions after each
    pick (up to 5 refinements per category), and advances when the user
    approves. Agent replies are structured JSON (RecommendationResponse). Once
    all categories are covered — or the user asks to finish — it builds a full
    itinerary with reservations. A checkpointer is required.
    """
    graph = StateGraph(TravelAgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("bootstrap_research", bootstrap_research_node)
    graph.add_node("research_agent", research_agent)
    graph.add_node("research_tools", ToolNode(RESEARCH_TOOLS))
    graph.add_node("merge_research_notes", merge_research_notes_node)
    graph.add_node("generate_recommendations", generate_recommendations_node)
    graph.add_node("wait_for_user", wait_for_user_node)
    graph.add_node("advance_category", advance_category_node)
    graph.add_node("synthesize_itinerary", synthesize_itinerary_node)
    graph.add_node("enrich_reservations", enrich_reservations_node)
    graph.add_node("reservations_agent", reservations_agent)
    graph.add_node("reservation_tools", ToolNode(RESEARCH_TOOLS))
    graph.add_node("finalize_agent", finalize_agent)

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