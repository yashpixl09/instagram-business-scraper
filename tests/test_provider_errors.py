"""The provider error taxonomy, and the one thing about it that can leak a credential.

Two questions are being asked here, and only one of them is about correctness in the usual
sense.

The first is the boring one: does every code map to the status and the retryable flag the
API layer was promised. That is a table, so it is tested as a table.

The second is the one this file exists for. Every provider in this project authenticates by
query string -- `?apiKey=`, `?api_key=` -- so the request URL IS a credential, and upstream
error bodies routinely echo the request back. A `ProviderError` that carries either one
turns a rate-limit blip into a key in a logfile, a Sentry event, and a 429 response body
sent to whoever asked. The prototype had a test pinning this. So does this.

The leak test does not check `str(exc)` and stop. A secret hiding in `exc.args[1]`, or in an
attribute nobody prints today, is a secret that leaks the first time someone adds
`logger.exception` or a `repr()` to a debug line. So it sweeps every string reachable from
the exception object.
"""

from __future__ import annotations

import json
import traceback
import unittest

from lead_engine.providers import errors
from lead_engine.providers.errors import ProviderError

# A key shaped like the real thing, and distinctive enough that a substring search for it
# cannot match by accident.
API_KEY = "sk_live_9f2c1a7b3e4d5f60ZZQ"

REQUEST_URL = (
    "https://www.searchapi.io/api/v1/search"
    f"?engine=google_maps&q=cafe+in+Indiranagar&api_key={API_KEY}"
)

# What a throttling provider actually sends back: the quota story, plus the request echoed
# in full. Both halves are hostile -- the top-level message is safe to show a user, the
# `request` block underneath is not.
RATE_LIMIT_BODY = json.dumps(
    {
        "error": "Rate limit exceeded. 100 searches per month on the Free plan.",
        "request": {"url": REQUEST_URL, "headers": {"Authorization": f"Bearer {API_KEY}"}},
        "account": {"api_key": API_KEY, "searches_left": 0},
    }
)


def reachable_strings(exc: BaseException) -> list[str]:
    """Every string this exception can put in front of a human or a log aggregator."""
    found = [str(exc), repr(exc), f"{exc}", f"{exc!r}"]
    found.extend(str(arg) for arg in exc.args)
    found.extend(f"{name}={value!r}" for name, value in vars(exc).items())
    if isinstance(exc, ProviderError):
        found.append(json.dumps(exc.as_dict()))
    found.extend(traceback.format_exception_only(type(exc), exc))
    return found


class TaxonomyTests(unittest.TestCase):
    """The table itself. Ported from the prototype and pinned so a refactor cannot drift."""

    EXPECTED = {
        # code                     status  retryable
        "location_not_found": (422, False),
        "provider_rate_limited": (429, True),
        "provider_auth_failed": (503, False),
        "provider_bad_response": (502, True),
        "provider_unavailable": (503, True),
    }

    def test_every_code_maps_to_its_documented_status_and_retryability(self):
        for code, (status, retryable) in self.EXPECTED.items():
            with self.subTest(code=code):
                error = ProviderError(code)
                self.assertEqual(error.code, code)
                self.assertEqual(error.status, status)
                self.assertEqual(error.retryable, retryable)

    def test_the_taxonomy_has_exactly_these_five_codes(self):
        # A sixth code added without a status is a KeyError in the API layer at request
        # time. Adding one has to break this test first.
        self.assertEqual(set(self.EXPECTED), set(errors.STATUS_BY_CODE))
        self.assertEqual(set(self.EXPECTED), set(errors.RETRYABLE_BY_CODE))
        self.assertEqual(set(self.EXPECTED), set(errors.DEFAULT_MESSAGE))
        self.assertEqual(set(self.EXPECTED), set(errors.CODES))

    def test_auth_failure_is_a_503_not_a_401(self):
        # Worth its own test because 401 is the obvious wrong answer. Our key being wrong
        # is our outage; telling the caller to fix credentials they do not hold sends them
        # chasing a problem they cannot reach.
        error = errors.auth_failed()
        self.assertEqual(error.status, 503)
        self.assertFalse(error.retryable)

    def test_an_unknown_code_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            ProviderError("provider_on_fire", "something new")

    def test_it_is_still_a_runtime_error(self):
        # The prototype's callers catch RuntimeError in a couple of places.
        self.assertIsInstance(ProviderError("provider_unavailable"), RuntimeError)

    def test_the_prototypes_four_positional_arguments_still_work(self):
        # Verbatim port: the call shape from `lead_finder/providers.py` must keep working,
        # including an explicit status that disagrees with the table.
        error = ProviderError("provider_bad_response", "Geoapify returned garbage.", 502, True)
        self.assertEqual(error.code, "provider_bad_response")
        self.assertEqual(str(error), "Geoapify returned garbage.")
        self.assertEqual(error.status, 502)
        self.assertTrue(error.retryable)

        override = ProviderError("provider_rate_limited", "slow down", 503, False)
        self.assertEqual(override.status, 503)
        self.assertFalse(override.retryable)

    def test_the_factories_agree_with_the_table(self):
        pairs = [
            (errors.location_not_found, "location_not_found"),
            (errors.rate_limited, "provider_rate_limited"),
            (errors.auth_failed, "provider_auth_failed"),
            (errors.bad_response, "provider_bad_response"),
            (errors.unavailable, "provider_unavailable"),
        ]
        for factory, code in pairs:
            with self.subTest(code=code):
                error = factory(provider="searchapi")
                self.assertEqual(error.code, code)
                self.assertEqual(error.status, errors.STATUS_BY_CODE[code])
                self.assertEqual(error.retryable, errors.RETRYABLE_BY_CODE[code])
                self.assertEqual(error.provider, "searchapi")
                self.assertEqual(str(error), errors.DEFAULT_MESSAGE[code])

    def test_from_status_reproduces_the_prototype_ladder(self):
        expected = {
            429: "provider_rate_limited",
            401: "provider_auth_failed",
            403: "provider_auth_failed",
            400: "provider_bad_response",
            404: "provider_bad_response",
            418: "provider_bad_response",
            500: "provider_bad_response",
            503: "provider_bad_response",
        }
        for status, code in expected.items():
            with self.subTest(upstream=status):
                self.assertEqual(errors.from_status(status).code, code)

    def test_as_dict_is_the_whole_wire_format(self):
        # Pinned exactly: an extra key here is an extra key in every API error response,
        # and the obvious extra key to add is the one carrying the upstream detail.
        error = errors.rate_limited(provider="searchapi")
        self.assertEqual(
            error.as_dict(),
            {
                "code": "provider_rate_limited",
                "message": errors.DEFAULT_MESSAGE["provider_rate_limited"],
                "retryable": True,
            },
        )


class SecretLeakTests(unittest.TestCase):
    """The rule: a ProviderError never carries the upstream body or the request URL.

    Defence one -- there is no field for a payload, and the factories produce fixed text --
    is absolute. On that path the error provably contains NOTHING from upstream, and
    `test_the_429_carries_nothing_from_the_upstream_body` asserts exactly that by pinning
    the whole message.

    Defence two is the net under a careless caller, and it is coarser than it first looks:
    the constructor does not scrub a bad message, it REJECTS it. `redact()` runs as a
    detector, and if it wanted to change anything the message is judged to be upstream data
    rather than something this codebase wrote, so all of it is dropped for the canned text.

    That is worth stating precisely, because the obvious design -- scrub the message and
    keep the remains -- is the weaker one. Surgical scrubbing removes the key and leaves
    the quota numbers, the account fields and the provider's internal field names behind,
    and the result reads as sanitised, so nobody looks at it twice. These errors are
    serialised into HTTP responses; residue is not acceptable there.

    The cost is real and is pinned too, in `test_a_clean_message_from_a_client_is_kept`: a
    developer's careless message vanishes entirely, and only clean prose survives.
    """

    def assertNoCredential(self, exc: BaseException) -> None:
        """Nothing reachable from `exc` contains a key or a request URL."""
        for text in reachable_strings(exc):
            self.assertNotIn(API_KEY, text, f"api key leaked via: {text!r}")
            self.assertNotIn("searchapi.io", text, f"request url leaked via: {text!r}")
            self.assertNotIn("api.geoapify.com", text, f"request url leaked via: {text!r}")

    def test_the_429_carries_nothing_from_the_upstream_body(self):
        # The headline case, on the path clients are meant to use. Not merely "no key" --
        # the message is pinned to the canned string, so no fragment of the body, no quota
        # numbers, and no account detail can be present. There is nowhere for them to be.
        for error in (
            errors.rate_limited(provider="searchapi"),
            errors.from_status(429, provider="searchapi"),
        ):
            with self.subTest(error=error.code):
                self.assertEqual(error.status, 429)
                self.assertNoCredential(error)
                self.assertEqual(str(error), errors.DEFAULT_MESSAGE["provider_rate_limited"])
                self.assertNotIn("searches_left", json.dumps(error.as_dict()))
                self.assertNotIn("100 searches", json.dumps(error.as_dict()))

    def test_a_client_that_interpolates_the_body_still_leaks_no_credential(self):
        # Defence two. Someone will eventually write this line at 2am while debugging a
        # different problem, and it must not be the line that leaks the key.
        careless = ProviderError("provider_rate_limited", f"upstream said: {RATE_LIMIT_BODY}")
        self.assertNoCredential(careless)

    def test_a_careless_message_is_dropped_whole_not_scrubbed(self):
        # The guarantee is stronger than "no key survives", so assert the stronger thing.
        # Left unpinned, someone could later swap the reject for a surgical scrub, keep
        # every assertion in this file green, and start shipping quota numbers and account
        # fields to API callers.
        careless = ProviderError("provider_rate_limited", f"upstream said: {RATE_LIMIT_BODY}")

        self.assertEqual(str(careless), errors.DEFAULT_MESSAGE["provider_rate_limited"])
        for fragment in ("upstream said", "Rate limit exceeded", "searches_left", "Free plan"):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, str(careless))

    def test_a_clean_message_from_a_client_is_kept(self):
        # The other half, and the reason this is a detector and not a blanket replacement.
        # If every custom message were discarded the clients would stop writing them and
        # every provider failure in the logs would read identically.
        error = ProviderError("location_not_found", "No supported location matched 'Whitefield'.")
        self.assertEqual(str(error), "No supported location matched 'Whitefield'.")
        self.assertEqual(error.as_dict()["message"], "No supported location matched 'Whitefield'.")

    def test_a_client_that_interpolates_the_url_still_leaks_no_credential(self):
        careless = ProviderError("provider_unavailable", f"timeout calling {REQUEST_URL}")
        self.assertNoCredential(careless)

    def test_every_code_survives_a_careless_message(self):
        for code in errors.STATUS_BY_CODE:
            with self.subTest(code=code):
                self.assertNoCredential(
                    ProviderError(code, f"{REQUEST_URL} -> {RATE_LIMIT_BODY}")
                )

    def test_the_error_has_nowhere_to_put_a_body(self):
        # Defence one, and the only one that cannot be defeated by a clever caller: the
        # object has no field for a payload. If one of these names ever appears, this test
        # fails before the field has a chance to be populated with a URL.
        error = errors.bad_response(provider="geoapify")
        for name in ("body", "response", "url", "request", "payload", "raw", "detail"):
            with self.subTest(attribute=name):
                self.assertFalse(hasattr(error, name), f"ProviderError grew a {name!r} field")

    def test_a_bearer_token_in_a_header_dump_is_scrubbed(self):
        error = ProviderError(
            "provider_auth_failed",
            f"rejected with headers {{'Authorization': 'Bearer {API_KEY}'}}",
        )
        self.assertNoCredential(error)

    def test_a_json_body_hides_the_key_behind_a_quote(self):
        # Worth its own case because it is the one a `key=value` scrubber misses. In JSON
        # the separator is `": "`, not `=`, so a pattern anchored on `\s*=` walks straight
        # past `"api_key": "sk_live_..."` and the key reaches the log intact.
        self.assertNotIn(API_KEY, errors.redact(f'{{"api_key": "{API_KEY}"}}'))
        self.assertNotIn(API_KEY, errors.redact(f'quota exceeded for key {API_KEY}'))

    def test_redact_leaves_ordinary_prose_alone(self):
        # A scrubber that eats every message is a scrubber someone switches off. Pin the
        # blast radius: no scheme, no `key=`, no credential-shaped run, no change.
        for clean in (
            "The provider rate limit was reached. Try again later.",
            "No supported location matched the supplied city or area.",
            "No results for 'Indiranagar, Bangalore' within 18000 metres.",
            "automation_opportunities has no row for this business",
        ):
            with self.subTest(text=clean):
                self.assertEqual(errors.redact(clean), clean)

    def test_redact_handles_each_shape_a_key_arrives_in(self):
        cases = [
            f"https://api.geoapify.com/v2/places?apiKey={API_KEY}",
            f"http://x/y?api_key={API_KEY}&limit=20",
            f"api_key={API_KEY}",
            f"apiKey={API_KEY}",
            f"X-API-Key: {API_KEY}",
            f"token={API_KEY}",
            f"Authorization: Bearer {API_KEY}",
            f"secret = {API_KEY}",
        ]
        for case in cases:
            with self.subTest(case=case):
                scrubbed = errors.redact(case)
                self.assertNotIn(API_KEY, scrubbed)
                self.assertIn(errors.REDACTED, scrubbed)


class ExceptionChainTests(unittest.TestCase):
    """`raise ... from exc` is the hole this taxonomy cannot close by itself.

    `ProviderError` controls what IT carries. It cannot control `__cause__`. An `httpx` or
    `urllib` exception holds `.request.url`, and every traceback formatter prints the whole
    chain -- so chaining from a transport exception re-introduces exactly the leak the rest
    of this file prevents.

    Both halves are pinned below: the safe form is proven safe, and the unsafe form is
    proven unsafe. The second assertion looks perverse, but it is the evidence for the rule
    in the module docstring. If someone later teaches this package to scrub the cause, that
    test fails and they delete it deliberately, having actually fixed the problem.
    """

    def _formatted(self, exc: BaseException) -> str:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    def test_from_none_keeps_the_transport_exception_out_of_the_traceback(self):
        try:
            try:
                raise OSError(f"connection failed for {REQUEST_URL}")
            except OSError:
                raise errors.unavailable(provider="searchapi") from None
        except ProviderError as exc:
            # Rebound deliberately: Python deletes the `as exc` name when the except block
            # ends, so anything needed afterwards has to be copied out here.
            raised, rendered = exc, self._formatted(exc)

        self.assertNotIn(API_KEY, str(raised))
        self.assertNotIn(API_KEY, rendered)
        self.assertNotIn("searchapi.io", rendered)

    def test_from_exc_would_leak_which_is_why_the_rule_exists(self):
        try:
            try:
                raise OSError(f"connection failed for {REQUEST_URL}")
            except OSError as cause:
                raise errors.unavailable(provider="searchapi") from cause
        except ProviderError as exc:
            raised, rendered = exc, self._formatted(exc)

        # The ProviderError itself is still clean...
        self.assertNotIn(API_KEY, str(raised))
        # ...but the chain it was raised from is not. Clients must use `from None`.
        self.assertIn(API_KEY, rendered)


if __name__ == "__main__":
    unittest.main()
