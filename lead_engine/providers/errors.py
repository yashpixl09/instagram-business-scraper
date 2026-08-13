"""The provider error taxonomy: five codes, one HTTP status each, ported from the prototype.

Every outbound integration in this system fails in the same five ways, and the API layer
has to turn each of them into a status code without knowing which vendor was involved. So
the classification happens once, here, and the HTTP clients raise these and nothing else.

    location_not_found   422  the caller asked for a place that does not exist
    provider_rate_limited 429 we are being throttled; back off and retry
    provider_auth_failed  503 our key is wrong or revoked; a caller retry cannot help
    provider_bad_response 502 upstream answered, but not with something we can parse
    provider_unavailable  503 upstream did not answer at all

`retryable` is about the *upstream* call, not the caller's HTTP request. `provider_auth_failed`
surfaces as 503 because a bad key is our outage, not the caller's mistake -- returning 401
would tell the caller to fix credentials they do not have.

THE ONE RULE THAT MATTERS
------------------------
A ProviderError must never carry the raw upstream response body or the request URL.

Every provider in this project authenticates by query string -- Geoapify's `?apiKey=`,
SearchAPI's `?api_key=`. A URL pasted into an error message is a leaked credential the
moment that message reaches a log aggregator, an HTTP response body, or a bug report. The
body is no safer: upstream error payloads routinely echo the request back.

Two defences, because one is not enough:

  1. There is no field to put a body in. The constructor takes a message, not a payload,
     and nothing here stores a response object.
  2. The constructor refuses to keep a message that looks like upstream data. It runs
     `redact()` as a *detector*: if scrubbing would change anything, the whole message is
     dropped in favour of the safe default for that code. A client that interpolates an
     httpx response body still ships nothing but generic text.

A third defence is the caller's job and cannot be enforced from here: raise these with
`from None`, not `from exc`, when the cause is an httpx or urllib exception. Those carry
`.request.url` -- key and all -- and an exception chain is printed in full by every
traceback formatter. Chain only from causes you have looked at.
"""

from __future__ import annotations

import re

REDACTED = "[redacted]"

#: Every code in the taxonomy, mapped to the HTTP status the API layer should return.
STATUS_BY_CODE: dict[str, int] = {
    "location_not_found": 422,
    "provider_rate_limited": 429,
    "provider_auth_failed": 503,
    "provider_bad_response": 502,
    "provider_unavailable": 503,
}

#: Whether retrying the *upstream* call could plausibly succeed without a code change.
RETRYABLE_BY_CODE: dict[str, bool] = {
    "location_not_found": False,
    "provider_rate_limited": True,
    "provider_auth_failed": False,
    "provider_bad_response": True,
    "provider_unavailable": True,
}

#: Safe, vendor-neutral default text. Chosen so a client that supplies no message of its
#: own still cannot leak anything, because it never had to write one.
DEFAULT_MESSAGE: dict[str, str] = {
    "location_not_found": "No supported location matched the supplied city or area.",
    "provider_rate_limited": "The provider rate limit was reached. Try again later.",
    "provider_auth_failed": "The provider rejected the configured API key.",
    "provider_bad_response": "The provider returned an invalid response.",
    "provider_unavailable": "The provider is temporarily unavailable.",
}

CODES: frozenset[str] = frozenset(STATUS_BY_CODE)

# Anything shaped like a URL, taken to the next delimiter so the query string -- where the
# key lives -- goes with it. The class stops at quotes and braces so a URL embedded in a
# JSON error body does not swallow the rest of the document.
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s\"'<>{}\\]+")

# `Authorization: Bearer sk-...` reduced to its scheme.
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+\S+")

# A labelled secret outside a URL, e.g. copied out of a header dict. The `[=:]` is
# required: making the separator optional would eat the word after every "secret" and
# "token" in ordinary prose, and a redactor that mangles readable messages gets deleted.
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|secret|password"
    r"|authorization|x-api-key)\b\s*[=:]\s*\S+"
)

# The one that actually does the work, because labels are unreliable. An upstream body
# says `"quota exceeded for key sk-live-9f3c..."` -- no `=`, no `:`, nothing to anchor on.
# So match the SHAPE of a credential instead: twenty or more characters from the key
# alphabet containing BOTH a letter and a digit.
#
# Requiring a digit is what keeps this off English. No word is twenty characters long and
# has a digit in it; long snake_case identifiers ("automation_opportunities") have no
# digit and survive. Real keys -- 32 hex characters, `sk-live-...`, base64 -- all match.
_CREDENTIAL_SHAPED = re.compile(
    r"\b(?=[A-Za-z0-9_\-]*[0-9])(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{20,}\b"
)


def redact(text: str) -> str:
    """Strip URLs, bearer tokens, `key=value` pairs and credential-shaped runs from text.

    Deliberately over-eager. A message that loses a harmless URL is a mildly worse log
    line; a message that keeps a real one is a credential in a logfile. The cost is real
    though -- a long opaque provider id in a message will be redacted too -- so the rules
    are tuned to leave ordinary prose untouched, and a test pins that.
    """
    text = _URL.sub(REDACTED, text)
    text = _BEARER.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    text = _SECRET_ASSIGNMENT.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    return _CREDENTIAL_SHAPED.sub(REDACTED, text)


def _safe_message(code: str, message: object) -> str:
    """Accept a caller's message, or throw all of it away.

    Not `redact(text)`. Scrubbing a URL out of an interpolated JSON error body leaves the
    rest of the body behind -- quota numbers, account fields, the provider's internal
    field names -- and that residue is still upstream data we chose not to inspect. Worse,
    it reads as sanitised, so nobody looks again.

    So the test is used as a detector rather than a fixer: if `redact()` wanted to change
    anything, the message is not something this codebase wrote, and none of it is kept.
    Clean prose passes through untouched, which is every message the clients actually
    write, including all of the prototype's.

    The cost is that a developer who interpolates a body while debugging sees the generic
    text instead of theirs. That is the intended lesson: put the detail in a log line you
    control, not in an object that gets serialised to whoever called the API.
    """
    text = str(message)
    return DEFAULT_MESSAGE[code] if redact(text) != text else text


class ProviderError(RuntimeError):
    """A classified failure from an external provider.

    The four positional arguments are the prototype's, unchanged, so ported call sites
    keep working:

        raise ProviderError("provider_rate_limited", "Rate limit reached.", 429, True)

    `status` and `retryable` may be omitted, in which case they are read from the tables
    above -- the preferred form, since a hand-typed status is a chance to get it wrong:

        raise ProviderError("provider_rate_limited")
    """

    def __init__(
        self,
        code: str,
        message: str | None = None,
        status: int | None = None,
        retryable: bool | None = None,
        *,
        provider: str | None = None,
    ) -> None:
        if code not in STATUS_BY_CODE:
            raise ValueError(f"unknown provider error code: {code!r}")
        self.code = code
        self.status = STATUS_BY_CODE[code] if status is None else int(status)
        self.retryable = RETRYABLE_BY_CODE[code] if retryable is None else bool(retryable)
        self.provider = provider
        self.message = DEFAULT_MESSAGE[code] if message is None else _safe_message(code, message)
        super().__init__(self.message)

    def as_dict(self) -> dict[str, object]:
        """The JSON body for an API response. Contains nothing the caller may not see."""
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code!r}, status={self.status!r}, "
            f"retryable={self.retryable!r}, message={self.message!r})"
        )


def location_not_found(message: str | None = None, *, provider: str | None = None) -> ProviderError:
    """The caller named a city or area we could not resolve. Not retryable."""
    return ProviderError("location_not_found", message, provider=provider)


def rate_limited(message: str | None = None, *, provider: str | None = None) -> ProviderError:
    """Upstream is throttling us."""
    return ProviderError("provider_rate_limited", message, provider=provider)


def auth_failed(message: str | None = None, *, provider: str | None = None) -> ProviderError:
    """Upstream rejected our key. Retrying the same request cannot fix it."""
    return ProviderError("provider_auth_failed", message, provider=provider)


def bad_response(message: str | None = None, *, provider: str | None = None) -> ProviderError:
    """Upstream answered with something we could not parse or did not expect."""
    return ProviderError("provider_bad_response", message, provider=provider)


def unavailable(message: str | None = None, *, provider: str | None = None) -> ProviderError:
    """Upstream did not answer: connection refused, DNS failure, timeout."""
    return ProviderError("provider_unavailable", message, provider=provider)


def from_status(status: int, *, provider: str | None = None) -> ProviderError:
    """Classify an upstream HTTP status.

    The ladder is the prototype's, verbatim: 429 throttles, 401/403 is our key, everything
    else -- 4xx and 5xx alike -- is a bad response. An upstream 503 landing on
    `provider_bad_response` rather than `provider_unavailable` looks arbitrary, but the
    latter is reserved for *no answer at all* (connection refused, DNS, timeout), and both
    are retryable, so nothing downstream behaves differently.

    Every client repeats this ladder, so it lives once. It takes the status code only --
    not the response -- which is what makes it impossible to use unsafely:

        response = client.get(url)
        if response.status_code >= 400:
            raise errors.from_status(response.status_code, provider="searchapi") from None
    """
    if status == 429:
        return rate_limited(provider=provider)
    if status in (401, 403):
        return auth_failed(provider=provider)
    return bad_response(provider=provider)
