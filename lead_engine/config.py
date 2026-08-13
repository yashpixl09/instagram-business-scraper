"""Environment configuration -- and the reason a key can never leave through the API.

Everything a deployment has to be told lives here, read from the process environment and
from `.env`. That part is unremarkable. What is deliberate is the shape of what comes back
out.

WHY THE SECRETS ARE `SecretStr` AND THE DUMP IS BOOLEANS
--------------------------------------------------------
`/api/config/status` answers one question -- "which providers are wired up" -- and the
honest answer is a row of booleans. The tempting implementation is `return settings`, and
the tempting debugging aid is `{"searchapi_key": key[:4] + "..."}`. A four-character prefix
confirms a guess, an eight-character one is most of a lookup, and this endpoint is
reachable from a browser.

So there are three layers between a key and a response, and each one is enough on its own:

  1. every secret is a `SecretStr`. `repr()`, `str()` and pydantic's JSON serialisation all
     render it `**********`, so a key cannot reach a log line or a traceback by accident;
  2. `model_dump()` is overridden to emit `capabilities()` -- booleans and nothing else.
     `return settings` from a route therefore ships a capability report, not a credential.
     There is no serialisation of this object that contains a key, in any mode;
  3. reading a secret's value requires naming it: `settings.searchapi_key_value`. That is
     one grep for every place in the codebase that touches a credential.

The DSN is a secret too, and for the same reason: it carries the database password.

`GEMINI_MODEL` is not a secret and is a plain string. It is here rather than beside the
key because a model name is configuration, and splitting configuration across two
mechanisms is how a deployment ends up with a key for a model it is not asking for.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, SecretStr, model_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict

#: The metered provider. Named here so the API, the CLI and the ledger agree on the string
#: that keys a budget row.
SEARCHAPI = "searchapi"

#: SearchAPI's free searches are granted once per key and never renew. Fifty is what the
#: operator holds; the ledger in Postgres is the accounting, and this only seeds it.
DEFAULT_SEARCH_BUDGET = 50

#: Every provider this system can be configured with, in the order a status report reads
#: best: the billed one first, then enrichment, then the interchangeable LLMs.
PROVIDER_NAMES: tuple[str, ...] = (
    SEARCHAPI,
    "tinyfish",
    "firecrawl",
    "groq",
    "gemini",
    "nvidia",
)

#: Providers that can each answer the same question, so any one of them is enough. A run
#: completes with none of them using deterministic copy, which is a guarantee rather than a
#: fallback -- see `lead_engine/copy.py`.
LLM_PROVIDERS: tuple[str, ...] = ("groq", "gemini", "nvidia")


def _value(secret: SecretStr | None) -> str | None:
    """Unwrap a secret, treating a blank one as absent.

    `SEARCHAPI_KEY=` in a `.env` file is how a key is *removed*, and it arrives here as an
    empty string. Treating that as configured would produce a 401 from the vendor instead
    of the 503 that tells the operator a key is missing.
    """
    if secret is None:
        return None
    text = secret.get_secret_value().strip()
    return text or None


class Settings(BaseSettings):
    """Everything the process is told, with the credentials sealed in."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # The `.env` in this repository also carries LEAD_ENGINE_TEST_DSN and comments for
        # the operator. An unknown key there is not a configuration error.
        extra="ignore",
    )

    lead_engine_dsn: SecretStr | None = None
    searchapi_key: SecretStr | None = None
    lead_engine_search_budget: int = Field(default=DEFAULT_SEARCH_BUDGET, ge=0)
    tinyfish_api_key: SecretStr | None = None
    firecrawl_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    gemini_model: str | None = None
    nvidia_api_key: SecretStr | None = None

    # --- the values, each one named at the point of use ---------------------------------

    @property
    def dsn(self) -> str | None:
        return _value(self.lead_engine_dsn)

    @property
    def searchapi_key_value(self) -> str | None:
        return _value(self.searchapi_key)

    @property
    def tinyfish_key_value(self) -> str | None:
        return _value(self.tinyfish_api_key)

    @property
    def search_budget(self) -> int:
        return int(self.lead_engine_search_budget)

    # --- capabilities, which is all a caller may ever see --------------------------------

    @property
    def database_configured(self) -> bool:
        return self.dsn is not None

    def provider_status(self) -> dict[str, bool]:
        """Which providers hold a key. Booleans, in registry order, never a value."""
        configured = {
            SEARCHAPI: self.searchapi_key,
            "tinyfish": self.tinyfish_api_key,
            "firecrawl": self.firecrawl_api_key,
            "groq": self.groq_api_key,
            "gemini": self.gemini_api_key,
            "nvidia": self.nvidia_api_key,
        }
        return {name: _value(configured[name]) is not None for name in PROVIDER_NAMES}

    @property
    def llm_configured(self) -> bool:
        """True when any LLM provider can answer. They are alternatives, not a stack."""
        status = self.provider_status()
        return any(status[name] for name in LLM_PROVIDERS)

    def capabilities(self) -> dict[str, Any]:
        """The whole object as a caller may see it: what is configured, never with what."""
        return {
            "database_configured": self.database_configured,
            "providers": self.provider_status(),
            "llm_configured": self.llm_configured,
            "search_budget": self.search_budget,
            "gemini_model": self.gemini_model,
        }

    @model_serializer(mode="plain")
    def _capabilities_only(self) -> dict[str, Any]:
        """Serialising this object yields capabilities, in every mode.

        This is the structural half of the promise in the module docstring. `SecretStr`
        already protects a field that is serialised by name; this protects the object, so
        that the one-line mistake -- `return settings` from a route, `logger.info(settings)`
        -- cannot produce a key even if a future field forgets to be a `SecretStr`.
        """
        return self.capabilities()
