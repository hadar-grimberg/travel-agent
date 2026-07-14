"""CLI entry point for the travel agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from travel_agent.graph import run_travel_agent


def _format_final_output(result: dict) -> str:
    lines: list[str] = []

    itinerary = result.get("itinerary")
    if itinerary:
        lines.append("\n" + "=" * 60)
        lines.append("YOUR ITINERARY")
        lines.append("=" * 60)
        lines.append(itinerary.model_dump_json(indent=2)[:8000])

    research = result.get("research")
    if research and research.reservation_candidates:
        lines.append("\n" + "=" * 60)
        lines.append("RESERVATION CHECKLIST")
        lines.append("=" * 60)
        for item in research.reservation_candidates[:15]:
            flag = "REQUIRED" if item.reservation_required else "optional"
            lines.append(
                f"  [{flag}] {item.name} ({item.category})"
                f" — book {item.lead_time_days or '?'} days ahead"
                f" — {item.booking_url}"
            )

    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="LangGraph travel agent — activity recommender chat",
    )
    parser.add_argument(
        "request",
        nargs="?",
        help='Natural language trip request, e.g. "Paris June 10-15 2026, $3000, food and culture"',
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Read trip request from a text file",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write conversation history JSON to this path when the chat ends",
    )
    args = parser.parse_args()

    if args.file:
        user_message = args.file.read_text(encoding="utf-8")
    elif args.request:
        user_message = args.request
    else:
        print("Enter your trip request (destination, dates, budget, preferences). End with Ctrl-D / Ctrl-Z:")
        user_message = sys.stdin.read()

    if not user_message.strip():
        parser.error("A trip request is required.")

    result = run_travel_agent(user_message.strip())

    final_output = _format_final_output(result)
    if final_output:
        print(final_output)

    if args.json_out:
        conversation = []
        for msg in result.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                conversation.append({"role": "assistant", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage) and msg.content:
                conversation.append({"role": "user", "content": str(msg.content)})

        serializable = {
            "trip_request": result.get("trip_request").model_dump(mode="json")
            if result.get("trip_request")
            else None,
            "itinerary": result.get("itinerary").model_dump(mode="json")
            if result.get("itinerary")
            else None,
            "user_preferences": result.get("user_preferences", []),
            "conversation": conversation,
        }
        args.json_out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"\nWrote conversation to {args.json_out}")


if __name__ == "__main__":
    main()
