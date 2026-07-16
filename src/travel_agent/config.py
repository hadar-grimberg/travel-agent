"""Injectable configuration for the activities agent.

All credentials and provider choices come from here — either read from the
environment (``Settings.from_env()``) or passed in explicitly by the host
(e.g. Tripper constructs ``Settings(...)`` and hands it to ``ActivitiesAgent``).
Nothing in this module reads a private ``.env`` or performs I/O at import.
"""

from __future__ import annotations

import os
from typing import Mapping

from pydantic import BaseModel

# LLM backends the agent can drive.
PROVIDER_TOKEN = "token"          # Nebius per-token OpenAI-compatible API
PROVIDER_SERVERLESS = "serverless"  # Nebius serverless dedicated endpoint
PROVIDER_MOCK = "mock"            # keyless deterministic mode (no LLM, no network)
PROVIDER_AUTO = "auto"            # pick the best available from what's configured


class Settings(BaseModel):
    """Runtime configuration. Construct directly for injection, or ``from_env()``."""

    llm_provider: str = PROVIDER_AUTO
    model: str = "gpt-4o-mini"

    # Nebius per-token API
    nebius_api_key: str | None = None
    nebius_base_url: str | None = None

    # Nebius serverless dedicated endpoint
    nebius_endpoint_url: str | None = None
    nebius_endpoint_id: str | None = None

    # Optional data providers (POI discovery, live web search)
    geoapify_api_key: str | None = None
    tavily_api_key: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from environment variables (defaults to ``os.environ``)."""
        env = os.environ if env is None else env

        def get(key: str) -> str | None:
            value = env.get(key)
            return value if value not in (None, "") else None

        return cls(
            llm_provider=get("TRAVEL_AGENT_LLM_PROVIDER") or PROVIDER_AUTO,
            model=get("TRAVEL_AGENT_MODEL") or "gpt-4o-mini",
            nebius_api_key=get("NEBIUS_API_KEY"),
            nebius_base_url=get("NEBIUS_BASE_URL"),
            nebius_endpoint_url=get("NEBIUS_ENDPOINT_URL"),
            nebius_endpoint_id=get("NEBIUS_ENDPOINT_ID"),
            geoapify_api_key=get("GEOAPIFY_API_KEY"),
            tavily_api_key=get("TAVILY_API_KEY"),
        )

    @property
    def resolved_provider(self) -> str:
        """The provider actually used, resolving ``auto`` against configured creds."""
        provider = (self.llm_provider or PROVIDER_AUTO).strip().lower()
        if provider in (PROVIDER_TOKEN, PROVIDER_SERVERLESS, PROVIDER_MOCK):
            return provider
        # auto: prefer per-token, then serverless, else keyless mock
        if self.nebius_api_key and self.nebius_base_url:
            return PROVIDER_TOKEN
        if self.nebius_endpoint_url and self.nebius_endpoint_id:
            return PROVIDER_SERVERLESS
        return PROVIDER_MOCK

    @property
    def use_mock(self) -> bool:
        """True when the agent runs keyless — no LLM call, deterministic output."""
        return self.resolved_provider == PROVIDER_MOCK
