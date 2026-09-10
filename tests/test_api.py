"""The HTTP surface, against a TestClient and -- where it matters -- a real Postgres.

Split the way the risk is split. Most of what an API layer gets wrong is arithmetic-free
and needs no database: the shape of a failure, the order of a list, whether a key can reach
a response body, whether a request that must not spend money spends money. Those tests run
everywhere, in milliseconds, and they are the bulk of this file.

The integration section at the bottom needs Postgres and is skipped without
`LEAD_ENGINE_TEST_DSN`, following `tests/test_migrations.py`: a throwaway schema per test,
dropped afterwards, so nothing leaks between tests and the suite is parallel-safe.

WHAT THESE TESTS ARE ACTUALLY DEFENDING
---------------------------------------
*The allowance.* Fifty searches exist and never renew. `POST /api/search` is wired to a
provider that raises on any call, so "this endpoint enqueues rather than searches" is proved
by construction rather than asserted by reading the code.

*The key.* A sentinel credential is planted in the settings the app is built with, and every
response this suite produces -- including `/openapi.json` and a deliberate 500 -- is searched
for it. A four-character prefix confirms a guess, so nothing may leak, not even a fragment.

*One error shape.* Four handlers and FastAPI's own 404 all have to produce
`{"error": {code, message, details}}`. A client that meets two shapes grows two code paths
and tests one.

NO TEST HERE REACHES THE NETWORK. The maps provider raises if called and the location
resolver is the offline seed, which is also why `Nowhereville` is a clean miss rather than a
geocoder request.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from lead_engine.api.app import create_app
from lead_engine.api.deps import Engine
from lead_engine.api.schemas import ApiProblem
from lead_engine.config import Settings
from lead_engine.geo.resolver import SeedResolver
from lead_engine.niches import niche_payload
from lead_engine.providers.errors import STATUS_BY_CODE, ProviderError

try:
    import psycopg
    from psycopg_pool import ConnectionPool

    from lead_engine.db.migrate import apply_migrations
except ImportError:  # pragma: no cover - the pure suite runs without a database driver
    psycopg = None

DSN = os.environ.get("LEAD_ENGINE_TEST_DSN")

#: Planted in every credential-shaped setting the app is built with. Shaped like a real key
#: -- long, mixed alphabet, with digits -- so a leak of any part of it is unmistakable.
SENTINEL = "sk-live-9f3c1d2b4a6e8c0f5171"

#: The DSN carries a password, which is a credential like any other.
SENTINEL_DSN = f"postgresql://lead_engine:{SENTINEL}@127.0.0.1:5433/lead_engine"

CITY = "Bangalore"
AREA = "Indiranagar"


def make_settings(**overrides) -> Settings:
    """Settings built from nothing but these arguments.

    `_env_file=None` and an explicit value for every field: this repository has a real `.env`
    and the developer running the suite may have real keys exported, and a test that reported
    "configured: True" because of the machine it ran on would be worthless.
    """
    base = dict(
        lead_engine_dsn=None,
        searchapi_key=None,
        lead_engine_search_budget=50,
        tinyfish_api_key=None,
        firecrawl_api_key=None,
        groq_api_key=None,
        gemini_api_key=None,
        gemini_model=None,
        nvidia_api_key=None,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def configured_settings(**overrides) -> Settings:
    """A fully wired deployment, every credential the sentinel."""
    base = dict(
        lead_engine_dsn=SENTINEL_DSN,
        searchapi_key=SENTINEL,
        tinyfish_api_key=SENTINEL,
        firecrawl_api_key=SENTINEL,
        groq_api_key=SENTINEL,
        gemini_api_key=SENTINEL,
        gemini_model="gemini-2.5-flash",
        nvidia_api_key=SENTINEL,
    )
    base.update(overrides)
    return make_settings(**base)


class RefusingProvider:
    """A maps provider that fails the test if anything asks it to search.

    This is how "`POST /api/search` does not spend a credit" is proved. An assertion that the
    call count is zero would pass just as well against an endpoint that called a provider
    which happened to be mocked; this one cannot.
    """

    api_key = "configured-but-must-not-be-called"

    def search_places(self, *args, **kwargs):
        raise AssertionError("the API issued a billed search; it must only enqueue one")


class RecordingBudget:
    """A ledger that answers reads and screams at a write."""

    def __init__(self, remaining: int = 50) -> None:
        self.remaining = remaining
        self.spends: list[tuple] = []

    def remaining_total(self, provider: str, *, key_fingerprint: str = "") -> int:
        return self.remaining

    def spend(self, provider: str, n: int = 1, **kwargs) -> int:  # pragma: no cover
        self.spends.append((provider, n))
        raise AssertionError("the API spent a search credit")

    def ensure_allowance(self, *args, **kwargs) -> dict[str, int]:
        return {"discover": self.remaining}


def build_client(settings: Settings | None = None, **engine_kwargs) -> TestClient:
    settings = settings if settings is not None else configured_settings()
    engine_kwargs.setdefault("maps_provider", RefusingProvider())
    engine_kwargs.setdefault("resolver", SeedResolver())
    engine_kwargs.setdefault("budget", RecordingBudget())
    engine = Engine(settings, **engine_kwargs)
    # `raise_server_exceptions=False`: the point of the 500 test is the response body, and
    # the default TestClient re-raises before there is one.
    return TestClient(create_app(settings, engine=engine), raise_server_exceptions=False)


@pytest.fixture
def client() -> TestClient:
    with build_client() as client:
        yield client


def search_body(**overrides) -> dict:
    body = {
        "location": {"city": CITY, "areas": [AREA]},
        "niches": ["salon"],
        "limit": 10,
        "use_ai": True,
    }
    body.update(overrides)
    return body


def error(response) -> dict:
    """The envelope, asserted to BE the envelope. Every failure in this API is this shape."""
    payload = response.json()
    assert set(payload) == {"error"}, payload
    assert set(payload["error"]) == {"code", "message", "details"}, payload
    return payload["error"]


# --- the shape of the surface ------------------------------------------------------------


def test_health_answers_without_a_database(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_niches_are_in_registry_order(client):
    response = client.get("/api/niches")
    assert response.status_code == 200
    returned = [niche["id"] for niche in response.json()["niches"]]
    # Registry order, not sorted and not the database's. The list is a catalogue a caller
    # renders; re-ordering it whenever a niche is searched would shuffle a UI for no reason.
    assert returned == [profile["id"] for profile in niche_payload()]
    assert len(returned) == 24


def test_every_niche_carries_its_registry_payload_and_a_verification_state(client):
    niches = {niche["id"]: niche for niche in client.get("/api/niches").json()["niches"]}
    expected = {profile["id"]: profile for profile in niche_payload()}

    for niche_id, profile in expected.items():
        returned = niches[niche_id]
        # `unverified` with no row is the honest default: nobody has looked yet, which is a
        # different statement from `starved`, where 20 places came back and none matched.
        assert returned["verification"] == "unverified"
        assert {key: returned[key] for key in profile} == profile


def test_config_status_is_booleans_and_a_budget(client):
    payload = client.get("/api/config/status").json()

    assert payload["database_configured"] is True
    assert payload["llm_configured"] is True
    assert set(payload["providers"]) == {
        "searchapi",
        "tinyfish",
        "firecrawl",
        "groq",
        "gemini",
        "nvidia",
    }
    assert all(isinstance(value, bool) for value in payload["providers"].values())
    assert payload["search_budget"] == {
        "provider": "searchapi",
        "configured": True,
        "remaining": 50,
    }


def test_config_status_reports_an_unconfigured_deployment_as_false():
    with build_client(make_settings(), maps_provider=None, budget=None) as client:
        payload = client.get("/api/config/status").json()

    assert payload["database_configured"] is False
    assert payload["llm_configured"] is False
    assert payload["providers"] == dict.fromkeys(payload["providers"], False)
    # None, not 0. "Nobody has seeded an allowance" and "the allowance is gone" call for
    # opposite reactions, and a caller shown 0 would abort a run that could go ahead.
    assert payload["search_budget"]["remaining"] is None


# --- the key never leaves ------------------------------------------------------------------


def test_no_response_carries_key_material(client):
    """The whole surface, searched for the sentinel. Including the schema and a 500.

    A prefix is enough to confirm a guess, so the assertion is on any fragment of the key,
    not on the whole string.
    """
    responses = [
        client.get("/health"),
        client.get("/api/niches"),
        client.get("/api/config/status"),
        client.get("/openapi.json"),
        client.post("/api/search", json=search_body()),
        client.post("/api/search", json={"city": 42}),
        client.get("/api/runs"),
        client.get("/api/leads"),
        client.get("/nowhere"),
    ]

    fragments = [SENTINEL, SENTINEL[:8], "9f3c1d2b"]
    for response in responses:
        body = response.text
        for fragment in fragments:
            assert fragment not in body, (response.request.url, fragment)


def test_the_settings_object_itself_cannot_be_serialised_into_a_key():
    """The structural half: `return settings` from a route would ship booleans.

    Belt and braces on purpose. `SecretStr` protects a field that is dumped by name; this
    protects the object, so a future field that forgets to be a `SecretStr` still cannot be
    serialised into a response.
    """
    settings = configured_settings()

    for dumped in (settings.model_dump(), settings.model_dump(mode="json")):
        assert SENTINEL not in str(dumped)
        assert dumped["providers"]["searchapi"] is True
    assert SENTINEL not in repr(settings)
    assert SENTINEL not in str(settings.model_dump_json())
    # And the value is still reachable, by naming it.
    assert settings.searchapi_key_value == SENTINEL


# --- validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "code"),
    [
        pytest.param([1, 2], "invalid_request", id="body-is-not-an-object"),
        pytest.param({"location": "Bangalore"}, "invalid_request", id="location-is-not-an-object"),
        pytest.param(search_body(location={"city": CITY, "areas": AREA}), "invalid_request",
                     id="areas-is-a-bare-string"),
        pytest.param(search_body(location={"city": CITY, "areas": [f"a{i}" for i in range(26)]}),
                     "invalid_request", id="too-many-areas"),
        pytest.param(search_body(location={"city": CITY, "areas": [""]}), "invalid_request",
                     id="blank-area"),
        pytest.param(search_body(location={"city": CITY, "country": "x" * 161}),
                     "invalid_request", id="country-too-long"),
        pytest.param({"niches": ["salon"], "limit": 10, "use_ai": True}, "invalid_city",
                     id="no-city"),
        pytest.param(search_body(location={"city": "   "}), "invalid_city", id="blank-city"),
        pytest.param(search_body(location={"city": "x" * 161}), "invalid_city", id="city-too-long"),
        pytest.param(search_body(location={"city": 7}), "invalid_city", id="city-is-not-a-string"),
        pytest.param(search_body(niches=[]), "invalid_niches", id="no-niches"),
        pytest.param(search_body(niches="salon"), "invalid_niches", id="niches-is-a-bare-string"),
        pytest.param(search_body(niches=[7]), "invalid_niches", id="niche-is-not-a-string"),
        pytest.param(search_body(niches=[f"n{i}" for i in range(25)]), "invalid_niches",
                     id="too-many-niches"),
        pytest.param(search_body(niches=["nail bar"]), "unsupported_niche", id="unknown-niche"),
        pytest.param(search_body(limit=0), "invalid_limit", id="limit-below-range"),
        pytest.param(search_body(limit=501), "invalid_limit", id="limit-above-range"),
        pytest.param(search_body(limit="10"), "invalid_limit", id="limit-is-a-string"),
        # `True` is an `int` in Python and would otherwise be a legal limit of 1.
        pytest.param(search_body(limit=True), "invalid_limit", id="limit-is-a-bool"),
        pytest.param(search_body(use_ai=None), "invalid_use_ai", id="use-ai-missing"),
        pytest.param(search_body(use_ai="yes"), "invalid_use_ai", id="use-ai-is-a-string"),
    ],
)
def test_every_validation_code_reaches_the_caller(client, body, code):
    response = client.post("/api/search", json=body)

    assert response.status_code == 422
    assert error(response)["code"] == code


def test_an_unsupported_niche_travels_with_the_supported_list(client):
    response = client.post("/api/search", json=search_body(niches=["nail bar"]))

    assert response.status_code == 422
    body = error(response)
    assert body["code"] == "unsupported_niche"
    assert "nail bar" in body["message"]
    # The list travels with the rejection: a caller that has to make a second request to
    # find out what it may ask for will hard-code the answer instead.
    assert body["details"]["supported_niches"] == [p["id"] for p in niche_payload()]


def test_a_plan_prices_a_search_without_a_database_or_a_provider_call(client):
    """`--dry-run` over HTTP: the price, before anyone agrees to pay it.

    The provider raises on any call and the budget raises on any write, so a plan that
    reached either would fail rather than quietly cost a credit.
    """
    response = client.post(
        "/api/search/plan",
        json=search_body(
            location={"city": CITY, "areas": [AREA, "Koramangala"]}, niches=["salon", "cafe"]
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["niches"] == ["salon", "cafe"]
    assert [location["label"] for location in payload["locations"]] == [
        "Indiranagar, Bangalore, Karnataka, India",
        "Koramangala, Bangalore, Karnataka, India",
    ]
    assert payload["locations"][0]["radius_meters"] == 3000
    # 2 areas x 2 niches, one billed search apiece under the breadth-first default.
    assert payload["planned_searches"] == 4
    assert payload["search_budget_remaining"] == 50


def test_a_plan_is_validated_exactly_as_a_search_is(client):
    response = client.post("/api/search/plan", json=search_body(niches=["nail bar"]))

    assert response.status_code == 422
    assert error(response)["code"] == "unsupported_niche"


def test_a_verdict_is_validated_against_the_supported_list(client):
    response = client.patch(
        f"/api/leads/{uuid.uuid4()}/status", json={"my_verdict": "maybe later"}
    )

    assert response.status_code == 422
    body = error(response)
    assert body["code"] == "invalid_status"
    assert body["details"]["supported_verdicts"] == ["high", "low", "medium", "skip"]


def test_a_verdict_is_a_potential_tier_not_a_pipeline_state(client):
    """Two orthogonal facts, two columns.

    `my_verdict` is whether the lead is worth pursuing -- the half only the operator can
    supply, since the engine bands audience size from evidence but cannot know whose
    afternoon is worth spending. `outcome` is where the approach got to. A lead can be
    high-potential AND already contacted; one vocabulary for both loses whichever is asked
    for second.

    This was briefly the prototype's OUTREACH_STATUSES, which meant Phase 8's gate on
    `high`/`medium` could never open through the only write path there is.
    """
    pipeline = client.patch(
        f"/api/leads/{uuid.uuid4()}/status", json={"my_verdict": "contacted"}
    )
    assert pipeline.status_code == 422, "a pipeline state is not a verdict"

    bad_outcome = client.patch(
        f"/api/leads/{uuid.uuid4()}/status",
        json={"my_verdict": "high", "outcome": "high"},
    )
    assert bad_outcome.status_code == 422, "a verdict is not an outcome"
    assert "contacted" in error(bad_outcome)["details"]["supported_outcomes"]


def test_fastapis_own_validation_uses_the_same_envelope(client):
    """A bad query parameter must not produce FastAPI's `detail` list.

    This is the shape that appears first in practice -- `?run_id=` from a stale bookmark --
    and it is the one most likely to teach a client a second parser.
    """
    response = client.get("/api/leads", params={"run_id": "not-a-uuid"})

    assert response.status_code == 422
    body = error(response)
    assert body["code"] == "invalid_request"
    assert body["details"]["fields"][0]["field"] == "query.run_id"


def test_a_body_that_is_not_json_is_the_same_envelope(client):
    response = client.post(
        "/api/search", content="{not json", headers={"content-type": "application/json"}
    )

    assert response.status_code == 422
    assert error(response)["code"] == "invalid_request"


def test_an_unknown_path_is_the_same_envelope(client):
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert error(response)["code"] == "not_found"


# --- failure handling -----------------------------------------------------------------------


class BoomEngine(Engine):
    """An Engine whose reporting call fails, so the handlers can be reached from a route."""

    def __init__(self, settings: Settings, exception: Exception) -> None:
        super().__init__(settings, maps_provider=RefusingProvider(), resolver=SeedResolver())
        self._exception = exception

    def niches(self):
        raise self._exception


def boom_client(exception: Exception) -> TestClient:
    settings = configured_settings()
    app = create_app(settings, engine=BoomEngine(settings, exception))
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("code", sorted(STATUS_BY_CODE))
def test_provider_errors_map_to_their_status(code):
    with boom_client(ProviderError(code)) as client:
        response = client.get("/api/niches")

    assert response.status_code == STATUS_BY_CODE[code]
    body = error(response)
    assert body["code"] == code
    # `retryable` is about the upstream call and is the field a client acts on.
    assert isinstance(body["details"]["retryable"], bool)


def test_an_api_problem_carries_its_own_status():
    problem = ApiProblem(409, "run_in_flight", "That run is already running.", {"run": "x"})
    with boom_client(problem) as client:
        response = client.get("/api/niches")

    assert response.status_code == 409
    assert error(response) == {
        "code": "run_in_flight",
        "message": "That run is already running.",
        "details": {"run": "x"},
    }


def test_an_unhandled_exception_reports_nothing_about_itself(caplog):
    """The 500 body is a constant, and the detail goes to the log.

    An exception message is written by whoever raised it -- psycopg puts the failing
    statement and its parameters into one -- so a handler that echoed `str(exc)` would
    publish a DSN, a key, or a customer's phone number to whoever made the request.
    """
    secret = f"password={SENTINEL} while executing INSERT INTO businesses"

    with caplog.at_level("ERROR"), boom_client(RuntimeError(secret)) as client:
        response = client.get("/api/niches")

    assert response.status_code == 500
    body = error(response)
    assert body["code"] == "internal_error"
    for fragment in (SENTINEL, SENTINEL[:8], "password", "INSERT INTO", "RuntimeError"):
        assert fragment not in response.text, fragment
    # ...and it is not merely swallowed: the operator can still find out what happened.
    logged = [
        str(record.exc_info[1]) if record.exc_info else record.getMessage()
        for record in caplog.records
    ]
    assert any(secret in line for line in logged), logged


def test_a_missing_searchapi_key_is_503_and_says_so():
    """Not 500. The operator needs to know it is a missing key, not a broken server."""
    settings = make_settings(lead_engine_dsn=SENTINEL_DSN)
    with build_client(settings, maps_provider=None, budget=None) as client:
        response = client.post("/api/search", json=search_body())

    assert response.status_code == 503
    body = error(response)
    assert body["code"] == "provider_not_configured"
    assert body["details"]["provider"] == "searchapi"


def test_a_search_without_a_database_says_which_piece_is_missing(client):
    """The provider is configured here; the database is not. The 503 has to name that."""
    response = client.post("/api/search", json=search_body())

    assert response.status_code == 503
    assert error(response)["code"] == "database_not_configured"


def test_validation_happens_before_configuration_is_checked():
    """A malformed request is malformed whatever the deployment is missing.

    Otherwise an operator with no key would be told to fix the key, fix it, and only then
    discover that the niche they asked for does not exist.
    """
    settings = make_settings()
    with build_client(settings, maps_provider=None, budget=None) as client:
        response = client.post("/api/search", json=search_body(niches=["nail bar"]))

    assert response.status_code == 422
    assert error(response)["code"] == "unsupported_niche"


def test_configuration_is_checked_before_the_geocoder_is_touched(client):
    """Order of refusals: provider, then database, then geography.

    Geocoding is the only step here that can reach a network, and doing it for a request
    that cannot proceed anyway buys a Nominatim call and a rate-limit wait to produce a 503
    that was already knowable.
    """
    response = client.post(
        "/api/search", json=search_body(location={"city": CITY, "areas": ["Nowhereville"]})
    )

    assert response.status_code == 503
    assert error(response)["code"] == "database_not_configured"


# --- integration ---------------------------------------------------------------------------

integration = pytest.mark.integration

needs_postgres = pytest.mark.skipif(
    psycopg is None or not DSN,
    reason="needs psycopg and LEAD_ENGINE_TEST_DSN (see env.example)",
)


@pytest.fixture
def pool():
    """A pool bound to a schema that exists only for this test.

    The injected pool IS the seam -- `Repository(pool)` and every store in this project take
    what they are given -- so no environment variable reaches the code under test.
    """
    schema = "test_" + uuid.uuid4().hex
    with psycopg.connect(DSN, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(f'CREATE SCHEMA "{schema}"')
    with psycopg.connect(DSN) as connection:
        connection.execute(f"SET search_path = {schema}, public")
        apply_migrations(connection)
        connection.commit()

    pool = ConnectionPool(
        DSN,
        min_size=1,
        max_size=4,
        open=True,
        kwargs={"options": f"-c search_path={schema},public"},
    )
    try:
        yield pool
    finally:
        pool.close()
        with psycopg.connect(DSN, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def live(pool):
    """A client over a real schema, with a provider that raises if anything searches."""
    settings = configured_settings(lead_engine_dsn=DSN)
    engine = Engine(
        settings, pool=pool, maps_provider=RefusingProvider(), resolver=SeedResolver()
    )
    with TestClient(create_app(settings, engine=engine), raise_server_exceptions=False) as client:
        yield client


def rows(pool, sql, params=None):
    with pool.connection() as connection:
        return connection.execute(sql, params).fetchall()


def add_business(pool, *, name="Blush Salon", niche="salon", city=CITY, area=AREA, run_id=None):
    """Seed a business, optionally attributed to the run that found it.

    `run_id` writes the enrichment row a discovery pass would have written. It is what makes
    the business belong to a run: `businesses` carries no run column, because a business is
    discovered once and lives on, so a column there could only record which run found it
    FIRST. What a run owns is the observations it paid for.
    """
    business_id = uuid.uuid4()
    with pool.connection() as connection:
        connection.execute(
            "INSERT INTO businesses (id, name, niche_id, city, search_area, phone)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            (business_id, name, niche, city, area, "9123456789"),
        )
        if run_id is not None:
            connection.execute(
                "INSERT INTO enrichments (business_id, source, status, data, run_id)"
                " VALUES (%s, 'google_maps', 'ok', '{\"reviews\": 210}'::jsonb, %s)",
                (business_id, run_id),
            )
    return business_id


@integration
@needs_postgres
def test_search_enqueues_and_does_not_search(live, pool):
    """The whole point of the endpoint: a run exists, work is queued, nothing was bought.

    The provider raises on any call, so this passing means no HTTP request went out -- which
    is what stops a browser refresh from spending two of fifty non-renewing credits.
    """
    response = live.post(
        "/api/search",
        json=search_body(
            location={"city": CITY, "state": "Karnataka", "areas": [AREA, "Koramangala"]},
            niches=["salon", "cafe"],
        ),
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["niches"] == ["salon", "cafe"]
    assert len(payload["locations"]) == 2
    # 2 areas x 2 niches, one billed search apiece under the breadth-first default.
    assert payload["enqueued_tasks"] == 4
    assert payload["planned_searches"] == 4
    assert payload["search_budget_remaining"] == 50

    tasks = rows(pool, "SELECT type, status, idem_key, payload FROM tasks ORDER BY idem_key")
    assert len(tasks) == 4
    assert {task[0] for task in tasks} == {"discover"}
    assert {task[1] for task in tasks} == {"pending"}
    # The idem key names the unit of work, so a resumed run recognises what it queued.
    assert len({task[2] for task in tasks}) == 4
    # The resolved point travels with the task: a worker that geocoded again could search
    # somewhere other than the receipt promised.
    assert tasks[0][3]["location"]["radius_meters"] == 3000

    run = rows(pool, "SELECT status, goal_id FROM runs")[0]
    assert run[0] == "queued"
    assert str(run[1]) == payload["goal_id"]
    spec = rows(pool, "SELECT spec FROM goals")[0][0]
    assert spec["niche_ids"] == ["salon", "cafe"]
    assert spec["areas"] == [AREA, "Koramangala"]
    events = rows(pool, "SELECT type, payload FROM events ORDER BY seq")
    assert [event[0] for event in events] == ["run.queued"]
    assert events[0][1]["enqueued_tasks"] == 4


@integration
@needs_postgres
def test_an_unresolvable_area_aborts_before_anything_is_recorded(live, pool):
    """`location_not_found` is a 422 through the shared taxonomy, and costs nothing.

    Half a sweep is not a smaller sweep; it is a sweep with a hole nothing downstream can
    see. So an area that will not resolve aborts the whole request -- and it must do so
    before a goal, a run or a task exists, or a later worker would search the areas that
    did resolve and report a complete run.

    The seed resolver is injected, so `Nowhereville` is a genuine miss rather than a
    geocoder request.
    """
    response = live.post(
        "/api/search",
        json=search_body(location={"city": CITY, "areas": [AREA, "Nowhereville"]}),
    )

    assert response.status_code == 422
    assert error(response)["code"] == "location_not_found"
    assert rows(pool, "SELECT count(*) FROM goals")[0][0] == 0
    assert rows(pool, "SELECT count(*) FROM runs")[0][0] == 0
    assert rows(pool, "SELECT count(*) FROM tasks")[0][0] == 0


@integration
@needs_postgres
def test_a_plan_records_nothing(live, pool):
    """A price is not a commitment. Nothing queued, nothing recorded, nothing spent."""
    response = live.post("/api/search/plan", json=search_body())

    assert response.status_code == 200
    assert rows(pool, "SELECT count(*) FROM goals")[0][0] == 0
    assert rows(pool, "SELECT count(*) FROM runs")[0][0] == 0
    assert rows(pool, "SELECT count(*) FROM tasks")[0][0] == 0
    assert rows(pool, "SELECT coalesce(sum(used), 0) FROM search_budget")[0][0] == 0


@integration
@needs_postgres
def test_the_search_budget_is_seeded_once_and_read_back(live, pool):
    payload = live.get("/api/config/status").json()
    assert payload["search_budget"]["remaining"] == 50

    ledger = rows(
        pool,
        "SELECT provider, purpose, limit_total, used FROM search_budget ORDER BY purpose",
    )
    # The default split gives discovery everything: re-checking ground already swept is
    # only worth paying for once there is no new ground left.
    assert ledger == [("searchapi", "discover", 50, 0), ("searchapi", "refresh", 0, 0)]


@integration
@needs_postgres
def test_runs_are_listed_newest_first_and_fetched_by_id(live):
    first = live.post("/api/search", json=search_body()).json()
    second = live.post("/api/search", json=search_body(niches=["cafe"])).json()

    listed = live.get("/api/runs").json()["runs"]
    assert [run["id"] for run in listed] == [second["run_id"], first["run_id"]]

    detail = live.get(f"/api/runs/{first['run_id']}").json()
    assert detail["id"] == first["run_id"]
    assert detail["status"] == "queued"
    assert detail["trigger"] == "api"
    assert detail["stats"] == {}


@integration
@needs_postgres
def test_an_unknown_run_is_a_404_in_the_envelope(live):
    response = live.get(f"/api/runs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert error(response)["code"] == "run_not_found"


@integration
@needs_postgres
def test_leads_default_to_the_latest_run_and_are_scoped_by_it(live, pool):
    """A run owns the businesses it paid to discover, not everything in its scope.

    Attribution is through `enrichments.run_id`, the row a discovery pass writes for every
    business it stores. So a business seeded without one belongs to no run -- which is also
    why a second run over the same city does not re-list the first run's leads: boundary
    dedup means it never stored them, so it never bought them.
    """
    started = live.post("/api/search", json=search_body(niches=["salon"])).json()
    run_id = started["run_id"]

    wanted = add_business(pool, name="Blush Salon", niche="salon", run_id=run_id)
    other_city = add_business(pool, name="Delhi Salon", niche="salon", city="Delhi",
                              run_id=run_id)
    other_niche = add_business(pool, name="Third Wave", niche="cafe", run_id=run_id)

    payload = live.get("/api/leads").json()

    ids = [lead["id"] for lead in payload["leads"]]
    assert ids == [str(wanted)]
    assert str(other_city) not in ids
    assert str(other_niche) not in ids
    assert payload["count"] == 1
    lead = payload["leads"][0]
    assert lead["name"] == "Blush Salon"
    assert lead["city"] == CITY
    # No score row yet, so the band is 'unknown' rather than a guess, and every scoring
    # field is null rather than zero.
    assert lead["audience_band"] == "unknown"
    assert lead["score_total"] is None
    assert lead["my_verdict"] is None


@integration
@needs_postgres
def test_leads_can_be_asked_for_by_run(live, pool):
    salons = live.post("/api/search", json=search_body(niches=["salon"])).json()
    add_business(pool, name="Blush Salon", niche="salon", run_id=salons["run_id"])
    live.post("/api/search", json=search_body(niches=["cafe"])).json()

    payload = live.get("/api/leads", params={"run_id": salons["run_id"]}).json()

    assert payload["run_id"] == salons["run_id"]
    assert [lead["name"] for lead in payload["leads"]] == ["Blush Salon"]
    # The latest run is the cafe one, and it is scoped to cafes, so it lists nothing.
    assert live.get("/api/leads").json()["leads"] == []


@integration
@needs_postgres
def test_leads_with_no_run_at_all_is_a_404(live):
    response = live.get("/api/leads")

    assert response.status_code == 404
    assert error(response)["code"] == "run_not_found"


@integration
@needs_postgres
def test_a_lead_is_fetched_by_id_with_its_band(live, pool):
    business_id = add_business(pool)
    with pool.connection() as connection:
        connection.execute(
            "INSERT INTO scores (business_id, scorer_version, total, signals, evidence,"
            " audience_index) VALUES (%s, 'google-only', 71, '[]', '{}', 0.42)",
            (business_id,),
        )
        connection.execute(
            "INSERT INTO enrichments (business_id, source, status, data)"
            " VALUES (%s, 'google_maps', 'ok', '{\"reviews\": 240}')",
            (business_id,),
        )

    payload = live.get(f"/api/leads/{business_id}").json()

    assert payload["id"] == str(business_id)
    assert payload["score_total"] == 71
    assert payload["reviews"] == 240
    # A cohort of one is banded on absolute review thresholds, and says which rule it used.
    assert payload["audience_band"] == "medium"
    assert payload["banding_method"] == "absolute"
    # numeric in Postgres, float on the wire: JSON has one number type, and a
    # string-encoded decimal is a trap for every client.
    assert payload["audience_index"] == pytest.approx(0.42)


@integration
@needs_postgres
def test_lead_detail_carries_contact_summary_and_pitches(live, pool):
    """The list view stays cheap; the detail view is where this is affordable -- see
    LeadOut's own note on why these fields are None on GET /api/leads but populated here."""
    business_id = add_business(pool)
    with pool.connection() as connection:
        connection.execute(
            "INSERT INTO scores (business_id, scorer_version, total, signals, evidence,"
            " audience_index) VALUES (%s, 'enriched', 71, %s, %s, 0.42)",
            (business_id, '["no website", "public phone"]', '{"ai_summary": "A real lead."}'),
        )
        connection.execute(
            "INSERT INTO contacts (business_id, name, role, phone, email, source, confidence)"
            " VALUES (%s, 'Priya Sharma', 'owner', '9123456789', 'priya@x.in', 'website', 0.8)",
            (business_id,),
        )
        connection.execute(
            "INSERT INTO outreach (business_id, kind, channel, body, evidence)"
            " VALUES (%s, 'website_pitch', 'email', 'Hi there', '{}')",
            (business_id,),
        )
        connection.execute(
            "INSERT INTO automation_opportunities (business_id, opportunity_id, confidence,"
            " trigger_signals, evidence) VALUES (%s, 'appointment_booking', 1.0, '[]', '{}')",
            (business_id,),
        )

    payload = live.get(f"/api/leads/{business_id}").json()

    assert payload["contact_name"] == "Priya Sharma"
    assert payload["contact_role"] == "owner"
    assert payload["signals"] == ["no website", "public phone"]
    assert payload["ai_summary"] == "A real lead."
    assert payload["website_pitch"] == "Hi there"
    assert payload["automation_opportunities"] == ["appointment_booking"]


@integration
@needs_postgres
def test_an_unknown_lead_is_a_404(live):
    response = live.get(f"/api/leads/{uuid.uuid4()}")

    assert response.status_code == 404
    assert error(response)["code"] == "lead_not_found"


@integration
@needs_postgres
def test_a_verdict_is_written_and_then_replaced(live, pool):
    business_id = add_business(pool)

    first = live.patch(
        f"/api/leads/{business_id}/status",
        json={"my_verdict": "high", "notes": "walk-in Tuesday"},
    )
    assert first.status_code == 204
    assert first.content == b""

    lead = live.get(f"/api/leads/{business_id}").json()
    assert lead["my_verdict"] == "high"
    assert lead["notes"] == "walk-in Tuesday"
    assert lead["verdict_updated_at"] is not None

    # Recording the approach does not require restating the opinion. `outreach_status` is
    # the prototype's field name and aliases `outcome` -- it always held pipeline states.
    second = live.patch(
        f"/api/leads/{business_id}/status",
        json={
            "outreach_status": "contacted",
            "contacted_on": "2026-08-13",
            "channel": "visit",
        },
    )
    assert second.status_code == 204

    updated = live.get(f"/api/leads/{business_id}").json()
    assert updated["outcome"] == "contacted"
    assert updated["contacted_on"] == "2026-08-13"
    assert updated["channel"] == "visit"
    # A second write replaces the first rather than inserting beside it.
    assert rows(pool, "SELECT count(*) FROM verdicts")[0][0] == 1


@integration
@needs_postgres
def test_a_verdict_on_a_lead_that_does_not_exist_is_a_404(live):
    response = live.patch(
        f"/api/leads/{uuid.uuid4()}/status", json={"my_verdict": "high"}
    )

    assert response.status_code == 404
    assert error(response)["code"] == "lead_not_found"


@integration
@needs_postgres
def test_niche_verification_is_read_from_the_table(live, pool):
    with pool.connection() as connection:
        connection.execute(
            "INSERT INTO niche_status (niche_id, state, qualified, rejected)"
            " VALUES ('salon', 'starved', 0, 24), ('cafe', 'verified', 3, 1)"
        )

    listed = live.get("/api/niches").json()["niches"]
    niches = {niche["id"]: niche["verification"] for niche in listed}

    assert niches["salon"] == "starved"
    assert niches["cafe"] == "verified"
    # Everything else defaults, and the order still comes from the registry.
    assert niches["manufacturer"] == "unverified"
    assert list(niches) == [profile["id"] for profile in niche_payload()]
