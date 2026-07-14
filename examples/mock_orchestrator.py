"""Mock orchestrator: drives ActivitiesAgent exactly like the real orchestrator would.

Starts a session with a structured trip (granular checklist interests), prints
every JSON request/response pair, relays feedback round by round, and ends with
the final itinerary + reservation checklist.

NOTE: this makes real LLM calls (NEBIUS_* keys required) and real POI/web
lookups (GEOAPIFY_API_KEY / TAVILY_API_KEY optional — falls back without them).

Usage:
    python examples/mock_orchestrator.py                 # scripted conversation
    python examples/mock_orchestrator.py --interactive   # you type the feedback
    python examples/mock_orchestrator.py --out run.json  # save the transcript
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from travel_agent.agent import ActivitiesAgent

# What the real orchestrator would send after the user filled the checklist
SAMPLE_START_REQUEST = {
    "action": "start",
    "trip": {
        "destination": "Kyoto",
        "start_date": "2026-04-03",
        "end_date": "2026-04-09",
        "budget_usd": 2800,
        "travelers": 2,
        "interests": [
            "buddhist_temples",
            "monasteries",
            "free_tours",
            "view_points",
            "gardens_and_parks",
            "restaurants",
            "cafes",
        ],
        "travel_style": "balanced",
        "dietary_restrictions": ["vegetarian"],
        "notes": "temples and kaiseki",
    },
}

MAX_TURNS = 25  # safety net against infinite loops


def scripted_feedback(response: dict) -> dict:
    """Simulate the user: keep the top 2 activities on the first round of each
    category and ask for more; on later rounds keep 1 more and approve."""
    recommendations = response.get("recommendations", {})
    names = [r.get("name") for r in recommendations.get("recommendations", []) if r.get("name")]
    if recommendations.get("round", 1) == 1:
        return {"selected": names[:2], "approve": False}
    return {"selected": names[:1], "approve": True}


def interactive_feedback() -> dict:
    raw = input(
        "\nActivity names to keep (comma-separated), or 'approve' / 'finish' / 'exit': "
    ).strip()
    lowered = raw.lower()
    if lowered == "approve":
        return {"approve": True}
    if lowered == "finish":
        return {"finish": True}
    if lowered in {"exit", "quit"}:
        return {"quit": True}
    return {"selected": [s.strip() for s in raw.split(",") if s.strip()]}


def show(label: str, payload: dict) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive ActivitiesAgent like the orchestrator")
    parser.add_argument("--interactive", action="store_true", help="type feedback yourself")
    parser.add_argument("--out", type=Path, help="write the full transcript JSON to this path")
    args = parser.parse_args()

    load_dotenv()
    agent = ActivitiesAgent()
    transcript: list[dict] = []

    def call(request: dict) -> dict:
        show(">>> ORCHESTRATOR -> AGENT", request)
        response = agent.handle(request)
        show("<<< AGENT -> ORCHESTRATOR", response)
        transcript.append({"request": request, "response": response})
        return response

    response = call(SAMPLE_START_REQUEST)
    session_id = response.get("session_id")

    turn = 0
    while response.get("status") == "awaiting_feedback" and turn < MAX_TURNS:
        turn += 1
        if args.interactive:
            feedback = interactive_feedback()
        else:
            feedback = scripted_feedback(response)
            print(f"\n[scripted user selection] {json.dumps(feedback, ensure_ascii=False)}")
        response = call({"action": "feedback", "session_id": session_id, "feedback": feedback})

    # Demonstrate the status action the orchestrator can use for debugging
    call({"action": "status", "session_id": session_id})

    print(f"\nSession finished with status: {response.get('status')}")
    if response.get("status") == "completed":
        days = response.get("itinerary", {}).get("days", [])
        checklist = response.get("reservation_checklist", [])
        print(f"Itinerary: {len(days)} days, {len(checklist)} reservation items.")

    if args.out:
        args.out.write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Transcript written to {args.out}")


if __name__ == "__main__":
    main()
