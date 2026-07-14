"""JSON-in / JSON-out facade for the orchestrator.

The orchestrator owns the conversation with the user. This class exposes the
activities agent as a plain object: pass a JSON request, get a JSON response.
Each `interrupt()` pause in the graph becomes an "awaiting_feedback" response;
the orchestrator relays the user's reply via the "feedback" action.

Actions:
    start    — begin a session from a structured trip or free-text request
    feedback — relay the user's selections into an awaiting session
    status   — inspect a session (debugging)

Feedback is structured, not free text: {"selected": [names to keep],
"approve": bool, "finish": bool, "quit": bool}.

Example:
    agent = ActivitiesAgent()
    out = agent.handle({"action": "start", "trip": {...}})
    out = agent.handle({"action": "feedback", "session_id": out["session_id"],
                        "feedback": {"selected": ["Kinkaku-ji Temple"], "approve": True}})
"""

from __future__ import annotations

import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from travel_agent.graph import build_graph
from travel_agent.models import FeedbackPayload, TripRequest

AWAITING = "awaiting_feedback"
COMPLETED = "completed"
ENDED = "ended"


class ActivitiesAgent:
    """Session-based wrapper around the travel agent graph."""

    def __init__(self):
        self._checkpointer = MemorySaver()
        self._app = build_graph(checkpointer=self._checkpointer)
        self._sessions: dict[str, str] = {}  # session_id -> status

    # -- public API ---------------------------------------------------------

    def handle(self, request: dict | str) -> dict:
        """Process one orchestrator request and return a JSON-serializable dict."""
        try:
            if isinstance(request, str):
                request = json.loads(request)
        except json.JSONDecodeError as exc:
            return self._error(f"Invalid JSON: {exc}")
        if not isinstance(request, dict):
            return self._error("Request must be a JSON object")

        action = request.get("action")
        try:
            if action == "start":
                return self._start(request)
            if action == "feedback":
                return self._feedback(request)
            if action == "status":
                return self._status(request)
            return self._error(f"Unknown action: {action!r}. Use start, feedback, or status.")
        except Exception as exc:  # surface graph/tool failures as JSON, not tracebacks
            return self._error(f"{type(exc).__name__}: {exc}", session_id=request.get("session_id"))

    def handle_json(self, request: str) -> str:
        """Strict string-in / string-out variant of handle()."""
        return json.dumps(self.handle(request), default=str)

    # -- actions ------------------------------------------------------------

    def _start(self, request: dict) -> dict:
        trip_data = request.get("trip")
        text = request.get("request")
        if not trip_data and not text:
            return self._error("'start' requires 'trip' (structured) or 'request' (free text)")

        initial_state: dict = {"messages": [HumanMessage(content=text or json.dumps(trip_data))]}
        if trip_data:
            try:
                initial_state["trip_request"] = TripRequest.model_validate(trip_data)
            except ValidationError as exc:
                return self._error(f"Invalid trip: {exc}")

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = AWAITING
        result = self._app.invoke(initial_state, self._config(session_id))
        return self._response(session_id, result)

    def _feedback(self, request: dict) -> dict:
        session_id = request.get("session_id")
        feedback = request.get("feedback")
        if not session_id or feedback is None:
            return self._error("'feedback' requires 'session_id' and 'feedback'")

        status = self._sessions.get(session_id)
        if status is None:
            return self._error(f"Unknown session: {session_id}", session_id=session_id)
        if status != AWAITING:
            return self._error(
                f"Session is '{status}' and no longer accepts feedback", session_id=session_id
            )

        if isinstance(feedback, str):
            try:
                feedback = json.loads(feedback)
            except json.JSONDecodeError:
                feedback = None
        if not isinstance(feedback, dict):
            return self._error(
                "'feedback' must be a JSON object, e.g. "
                '{"selected": ["Activity name"], "approve": false}',
                session_id=session_id,
            )
        try:
            payload = FeedbackPayload.model_validate(feedback)
        except ValidationError as exc:
            return self._error(f"Invalid feedback: {exc}", session_id=session_id)

        result = self._app.invoke(
            Command(resume=payload.model_dump()), self._config(session_id)
        )
        return self._response(session_id, result)

    def _status(self, request: dict) -> dict:
        session_id = request.get("session_id")
        status = self._sessions.get(session_id)
        if status is None:
            return self._error(f"Unknown session: {session_id}", session_id=session_id)

        state = self._app.get_state(self._config(session_id)).values
        categories = state.get("categories") or []
        index = state.get("current_category_index", 0)
        return {
            "session_id": session_id,
            "status": status,
            "phase": state.get("phase"),
            "categories": categories,
            "current_category": categories[index] if index < len(categories) else None,
            "round": state.get("category_refinement_count", 0),
            "user_preferences": state.get("user_preferences") or [],
        }

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    @staticmethod
    def _error(message: str, session_id: str | None = None) -> dict:
        response: dict = {"status": "error", "error": message}
        if session_id:
            response["session_id"] = session_id
        return response

    def _response(self, session_id: str, result: dict) -> dict:
        """Translate a graph invoke() result into the orchestrator response contract."""
        if result.get("__interrupt__"):
            data = result["__interrupt__"][0].value
            try:
                recommendations = json.loads(data.get("display") or "{}")
            except json.JSONDecodeError:
                recommendations = {"raw": data.get("display")}
            return {
                "session_id": session_id,
                "status": AWAITING,
                "recommendations": recommendations,
                "prompt": data.get("prompt"),
            }

        itinerary = result.get("itinerary")
        if itinerary is None:
            self._sessions[session_id] = ENDED
            return {
                "session_id": session_id,
                "status": ENDED,
                "user_preferences": result.get("user_preferences") or [],
            }

        self._sessions[session_id] = COMPLETED
        research = result.get("research")
        checklist = (
            [item.model_dump(mode="json") for item in research.reservation_candidates]
            if research
            else []
        )
        return {
            "session_id": session_id,
            "status": COMPLETED,
            "itinerary": itinerary.model_dump(mode="json"),
            "reservation_checklist": checklist,
            "user_preferences": result.get("user_preferences") or [],
        }


def main() -> None:
    """Standalone debug loop: exercise the JSON contract from the terminal."""
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    agent = ActivitiesAgent()

    text = " ".join(sys.argv[1:]) or input("Trip request: ").strip()
    response = agent.handle({"action": "start", "request": text})
    print(json.dumps(response, indent=2, default=str))

    while response.get("status") == AWAITING:
        raw = input(
            "\nActivity names to keep (comma-separated), or 'approve' / 'finish' / 'exit': "
        ).strip()
        lowered = raw.lower()
        if lowered == "approve":
            feedback: dict = {"approve": True}
        elif lowered == "finish":
            feedback = {"finish": True}
        elif lowered in {"exit", "quit"}:
            feedback = {"quit": True}
        else:
            feedback = {"selected": [s.strip() for s in raw.split(",") if s.strip()]}
        response = agent.handle(
            {"action": "feedback", "session_id": response["session_id"], "feedback": feedback}
        )
        print(json.dumps(response, indent=2, default=str))


if __name__ == "__main__":
    main()
