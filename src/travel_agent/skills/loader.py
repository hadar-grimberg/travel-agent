"""Load skill playbooks (markdown) into agent prompts."""

from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Phase → skill files to inject
PHASE_SKILLS: dict[str, list[str]] = {
    "research": [
        "destination_research.md",
        "drive_radius_planning.md",
    ],
    "planning": [
        "itinerary_planning.md",
        "budget_management.md",
    ],
    "reservations": [
        "reservation_guidance.md",
    ],
    "finalize": [
        "client_experience.md",
    ],
}


def load_skill(name: str) -> str:
    path = SKILLS_DIR / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_skills_for_phase(phase: str) -> tuple[list[str], str]:
    names = PHASE_SKILLS.get(phase, [])
    sections = []
    for name in names:
        content = load_skill(name)
        if content:
            sections.append(f"## Skill: {name}\n\n{content}")
    return names, "\n\n---\n\n".join(sections)


def all_skill_names() -> list[str]:
    if not SKILLS_DIR.exists():
        return []
    return sorted(p.name for p in SKILLS_DIR.glob("*.md"))
