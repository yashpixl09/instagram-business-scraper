"""Thin chat clients for the three interchangeable LLM providers.

Written against the live documentation, read on 2026-08-14:

    Groq    POST https://api.groq.com/openai/v1/chat/completions
            header `Authorization: Bearer <key>`     OpenAI-compatible envelope
    Gemini  POST https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent
            header `x-goog-api-key: <key>`           contents/parts envelope
    NVIDIA  POST https://integrate.api.nvidia.com/v1/chat/completions
            header `Authorization: Bearer <key>`     OpenAI-compatible envelope

Groq is the default first link because it is genuinely free -- no card, 30 requests per
minute, 14,400 per day -- so the cheapest possible run is the one that never leaves it.

THE KEY IS NEVER IN THE URL
---------------------------
Gemini's own quickstart passes the key as `?key=...`. This client does not, and that is a
deliberate divergence from the docs rather than an oversight. Every error path in this
project has to be safe to log, and `lead_engine/providers/errors.py` exists because a URL
in a message is a credential in a logfile. A header cannot end up in `httpx`'s
`request.url`, so the key cannot ride out on an exception nobody inspected. The header form
is documented and supported; only the sample happens to use the query string.

Everything else here is the house pattern from `providers/tinyfish.py`:

* `transport` is injectable, so no test in this repo opens a socket;
* every failure becomes a `ProviderError` from the shared five-code taxonomy and nothing
  else -- an `httpx.ReadTimeout` escaping this module would push a vendor's failure modes
  into the router, which then has to know what one means;
* transport exceptions are re-raised with `from None`, because the httpx exception carries
  `.request.url` and a traceback prints the whole chain;
* every message is static prose with at most an HTTP status interpolated. Nothing here
  interpolates a response body, a URL or a key. `errors._safe_message` would throw such a
  message away rather than leak it, so the visible symptom of getting this wrong is a
  useless error, not a leak -- but the discipline is here so it never gets that far.

No client retries. Retrying is the router's decision to make, once, with a circuit breaker
and a persisted token bucket in front of it; a retry loop hidden down here would multiply
by whatever the router does above and is exactly the retry storm a 429 must not cause.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from lead_engine.providers.errors import (
    ProviderError,
    auth_failed,
    bad_response,
    rate_limited,
    unavailable,
)

GROQ = "groq"
GEMINI = "gemini"
NVIDIA = "nvidia"

#: The chain, in order. `config.LLM_PROVIDERS` carries the same order for the capability
#: report; this is the one the router walks.
CHAIN: tuple[str, ...] = (GROQ, GEMINI, NVIDIA)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"
NVIDIA_MODEL = "meta/llama-3.3-70b-instruct"

#: Free-tier request ceilings, per minute. These seed the router's token buckets, and the
#: cheapest 429 is the one that is never sent. A paid plan raises them through the
#: constructor rather than by editing these.
GROQ_REQUESTS_PER_MINUTE = 30
GEMINI_REQUESTS_PER_MINUTE = 10
NVIDIA_REQUESTS_PER_MINUTE = 40

#: Longer than the search clients' 30s: prose generation on a 70B model is slower than a
#: lookup, and a timeout here spends a retry-free attempt for nothing.
DEFAULT_TIMEOUT = 45.0

#: Enough for the longest artifact in `outreach/generate.py` (a visit brief) with room for
#: a thinking model's preamble, and small enough that a runaway generation is capped.
DEFAULT_MAX_TOKENS = 800

#: Low, not zero. The task is prose over rows that already exist; invention is the failure
#: mode this whole phase is built against, and temperature is the dial that buys it.
DEFAULT_TEMPERATURE = 0.2


@dataclass(frozen=True)
class Completion:
    """One answer from one provider.

    Token counts are optional because not every provider reports them on every response,
    and a zero written where a count is unknown would be an invented number in the ledger --
    the same mistake this phase exists to prevent, one layer down.
    """

    provider: str
    model: str
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class ChatClient(Protocol):
    """What the router needs from a provider, and nothing more."""

    name: str
    model: str
    requests_per_minute: int

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Completion: ...

    def close(self) -> None: ...


class _HttpChatClient:
    """Shared transport, auth-header assembly and status mapping.

    Subclasses supply a URL, a request body and a response reader. They never touch httpx.
    """

    name = ""
    model = ""
    requests_per_minute = 0

    def __init__(
        self,
        api_key: str | None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout,
            headers={"Accept": "application/json", **self._auth_headers(api_key)},
        )

    # -- subclass hooks -------------------------------------------------------------

    @staticmethod
    def _auth_headers(api_key: str | None) -> dict[str, str]:
        raise NotImplementedError

    def _url(self) -> str:
        raise NotImplementedError

    def _payload(
        self, prompt: str, system: str | None, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        raise NotImplementedError

    def _read(self, payload: dict[str, Any]) -> tuple[str, int | None, int | None]:
        raise NotImplementedError

    # -- lifecycle ------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        """Whether this client holds a key. An unconfigured provider is simply not built."""
        return bool(self.api_key)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> _HttpChatClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- the one public method ------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Completion:
        """Generate once. Raises `ProviderError` and nothing else."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        body = self._payload(prompt, system, max_tokens, temperature)
        payload = self._post(body)
        text, input_tokens, output_tokens = self._read(payload)
        if not text.strip():
            # An empty completion is not a completion. Returning "" would let the caller
            # store an empty pitch as a successful generation and cache it forever.
            raise bad_response(
                f"{self.name} returned an empty completion.", provider=self.name
            )
        return Completion(
            provider=self.name,
            model=self.model,
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # -- transport ------------------------------------------------------------------

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(self._url(), json=body)
        except httpx.TimeoutException:
            raise unavailable(f"{self.name} timed out.", provider=self.name) from None
        except httpx.HTTPError:
            raise unavailable(f"{self.name} could not be reached.", provider=self.name) from None

        if response.status_code >= 400:
            raise status_error(self.name, response.status_code)

        try:
            payload = response.json()
        except ValueError:
            raise self._bad("returned a body that was not JSON") from None
        if not isinstance(payload, dict):
            raise self._bad("returned a JSON body that was not an object")
        return payload

    def _bad(self, what: str) -> ProviderError:
        return bad_response(f"{self.name} {what}.", provider=self.name)


def status_error(provider: str, status_code: int) -> ProviderError:
    """Map an HTTP status onto the shared taxonomy.

    The status code is the one piece of the exchange that is always safe to name -- it is
    neither the key, the body nor the URL -- and it is the single most useful thing to have
    in a run log when a chain fell through to the template.

    Follows `providers/tinyfish.py` rather than `errors.from_status`: 5xx is
    `provider_unavailable` (upstream is down) rather than `provider_bad_response` (upstream
    sent us nonsense). Both are retryable and both count as one failure against the circuit
    breaker, so nothing downstream behaves differently -- but the run log reads honestly.
    """
    if status_code == 429:
        return rate_limited(
            f"{provider} rate limit reached (HTTP {status_code}).", provider=provider
        )
    if status_code in (401, 403):
        return auth_failed(
            f"{provider} rejected the configured key (HTTP {status_code}).", provider=provider
        )
    if status_code >= 500:
        return unavailable(f"{provider} is unavailable (HTTP {status_code}).", provider=provider)
    return bad_response(f"{provider} refused the request (HTTP {status_code}).", provider=provider)


class _OpenAICompatibleClient(_HttpChatClient):
    """Groq and NVIDIA both speak the OpenAI chat-completions envelope."""

    url = ""
    #: Groq takes the current `max_completion_tokens`; NVIDIA's NIM gateway takes the
    #: original `max_tokens`. One field name, one attribute, no branching in `_payload`.
    max_tokens_field = "max_completion_tokens"

    @staticmethod
    def _auth_headers(api_key: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _url(self) -> str:
        return self.url

    def _payload(
        self, prompt: str, system: str | None, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            self.max_tokens_field: max_tokens,
            "stream": False,
        }

    def _read(self, payload: dict[str, Any]) -> tuple[str, int | None, int | None]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise self._bad("returned no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise self._bad("returned a malformed choice")
        message = first.get("message")
        if not isinstance(message, dict):
            raise self._bad("returned a choice with no message")
        content = message.get("content")
        if not isinstance(content, str):
            raise self._bad("returned a message with no text content")
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return content, _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))


class GroqClient(_OpenAICompatibleClient):
    """The default first link: free, no card, 30 RPM, 14,400 requests a day."""

    name = GROQ
    url = GROQ_URL
    max_tokens_field = "max_completion_tokens"

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = GROQ_MODEL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        requests_per_minute: int = GROQ_REQUESTS_PER_MINUTE,
    ) -> None:
        self.model = model
        self.requests_per_minute = requests_per_minute
        super().__init__(api_key, transport=transport, timeout=timeout)


class NvidiaClient(_OpenAICompatibleClient):
    """The last provider before the deterministic template."""

    name = NVIDIA
    url = NVIDIA_URL
    max_tokens_field = "max_tokens"

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str = NVIDIA_MODEL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        requests_per_minute: int = NVIDIA_REQUESTS_PER_MINUTE,
    ) -> None:
        self.model = model
        self.requests_per_minute = requests_per_minute
        super().__init__(api_key, transport=transport, timeout=timeout)


class GeminiClient(_HttpChatClient):
    """`generateContent`, with the key in a header rather than the query string."""

    name = GEMINI

    def __init__(
        self,
        api_key: str | None,
        *,
        model: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = GEMINI_BASE_URL,
        requests_per_minute: int = GEMINI_REQUESTS_PER_MINUTE,
    ) -> None:
        # `None` and `""` both mean "not configured" -- see `config._value`, where an empty
        # env var is how a setting is removed. Neither may become a model named "".
        self.model = model or GEMINI_MODEL
        self.base_url = base_url.rstrip("/")
        self.requests_per_minute = requests_per_minute
        super().__init__(api_key, transport=transport, timeout=timeout)

    @staticmethod
    def _auth_headers(api_key: str | None) -> dict[str, str]:
        return {"x-goog-api-key": api_key} if api_key else {}

    def _url(self) -> str:
        return f"{self.base_url}/models/{self.model}:generateContent"

    def _payload(
        self, prompt: str, system: str | None, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _read(self, payload: dict[str, Any]) -> tuple[str, int | None, int | None]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            # Also what a safety block looks like: `promptFeedback` and no candidates. That
            # is a refusal, not an outage, but it is still "no answer from this provider",
            # and the router's next link -- or the template -- is the right response.
            raise self._bad("returned no candidates")
        first = candidates[0]
        if not isinstance(first, dict):
            raise self._bad("returned a malformed candidate")
        content = first.get("content")
        if not isinstance(content, dict):
            raise self._bad("returned a candidate with no content")
        text = _joined_parts(content.get("parts"))
        if text is None:
            raise self._bad("returned content with no text part")
        usage = payload.get("usageMetadata")
        usage = usage if isinstance(usage, dict) else {}
        return (
            text,
            _count(usage.get("promptTokenCount")),
            _count(usage.get("candidatesTokenCount")),
        )


def _joined_parts(parts: object) -> str | None:
    """Concatenate a Gemini candidate's text parts, skipping its thinking.

    2.5 Flash is a thinking model: a response may carry parts flagged `thought: true`, and
    they are reasoning, not the answer. Splicing them into the artifact would put the
    model's deliberation in front of a business owner.
    """
    if not isinstance(parts, list):
        return None
    texts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    ]
    return "".join(texts) if texts else None


def _count(value: object) -> int | None:
    """A token count, or None. Never a zero standing in for "not reported"."""
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def secret_value(secret: object) -> str | None:
    """Unwrap a `SecretStr`, a plain string, or nothing at all.

    Duck-typed rather than importing pydantic, so this package stays importable without the
    settings layer -- and so `config.Settings` remains the only place that decides what a
    blank key means. A blank value is `None`: `GROQ_API_KEY=` in a `.env` is how a key is
    removed, and treating it as configured buys a 401 instead of a clean skip.
    """
    if secret is None:
        return None
    reader = getattr(secret, "get_secret_value", None)
    text = reader() if callable(reader) else secret
    if not isinstance(text, str):
        return None
    return text.strip() or None


def build_clients(
    *,
    groq_key: object = None,
    gemini_key: object = None,
    nvidia_key: object = None,
    gemini_model: str | None = None,
    transport: httpx.BaseTransport | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    only: Iterable[str] | None = None,
) -> list[ChatClient]:
    """Build the chain, in order, skipping every provider without a key.

    Returns `[]` when nothing is configured, which is a supported configuration and not an
    error: the router falls straight through to the deterministic template and the run
    completes. See `lead_engine/copy.py`.

    `only` narrows the chain by name, for an operator who holds three keys but wants one
    provider used -- and for tests that need a chain of exactly one.
    """
    allowed = None if only is None else {str(name) for name in only}
    builders: Sequence[tuple[str, Any]] = (
        (GROQ, lambda key: GroqClient(key, transport=transport, timeout=timeout)),
        (
            GEMINI,
            lambda key: GeminiClient(
                key, model=gemini_model, transport=transport, timeout=timeout
            ),
        ),
        (NVIDIA, lambda key: NvidiaClient(key, transport=transport, timeout=timeout)),
    )
    keys = {
        GROQ: secret_value(groq_key),
        GEMINI: secret_value(gemini_key),
        NVIDIA: secret_value(nvidia_key),
    }

    clients: list[ChatClient] = []
    for name, build in builders:
        if allowed is not None and name not in allowed:
            continue
        key = keys[name]
        if key is None:
            continue
        clients.append(build(key))
    return clients


def clients_from_settings(
    settings: Any,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    only: Iterable[str] | None = None,
) -> list[ChatClient]:
    """`build_clients` fed from a `config.Settings`, without importing it.

    `Settings` exposes the LLM keys as `SecretStr` fields with no unwrapping property of
    their own, so `secret_value` does the unwrapping here -- one place, greppable, and it
    treats a blank exactly as `config._value` does.
    """
    return build_clients(
        groq_key=getattr(settings, "groq_api_key", None),
        gemini_key=getattr(settings, "gemini_api_key", None),
        nvidia_key=getattr(settings, "nvidia_api_key", None),
        gemini_model=getattr(settings, "gemini_model", None),
        transport=transport,
        timeout=timeout,
        only=only,
    )
