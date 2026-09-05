"""`POST /api/enrich` -- request parsing and response shape.

Mirrors `POST /api/search`'s own testing convention: the route and its schema are a few
lines of adaptation over `Engine.execute_enrichment`, so what is worth proving here is the
wire contract -- which bodies are accepted, which are refused and with what code, and that
the response reproduces the engine's report exactly -- not the enrichment pipeline itself,
which `tests/test_pipeline.py` already covers with fakes.

Unlike `POST /api/search`, this endpoint is not enqueued -- `Engine.execute_enrichment`'s
own docstring, and the route's, say why: TinyFish is free, Firecrawl is a monthly pool
merely reported on rather than a non-renewing allowance, and a repeated prompt is served
from the LLM cache rather than re-billed. `test_the_endpoint_runs_inline_rather_than_being_enqueued`
is the proof: nothing enqueues in this system without a `Repository`, and `RecordingEngine`
below never touches one, yet the call still lands, synchronously, within the request.

NO TEST HERE REACHES THE NETWORK OR A DATABASE.
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from lead_engine.api.app import create_app
from lead_engine.api.deps import Engine, EnrichmentReport
from lead_engine.config import Settings


def make_settings(**overrides) -> Settings:
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


class RecordingEngine(Engine):
    """`execute_enrichment` records its arguments and returns a canned report. This file
    tests routing and schema validation, not the pipeline -- see `tests/test_pipeline.py`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[dict] = []

    def execute_enrichment(self, **kwargs):
        self.calls.append(kwargs)
        return EnrichmentReport(
            business_ids=tuple(kwargs.get("business_ids") or (uuid.uuid4(),)),
            enriched=1,
            scored=1,
            offers_detected=2,
            outreach_written=3,
            usage={"primary_calls": 4, "fallback_calls": 0, "fallback_unavailable": 0},
        )


def build_client(engine: Engine | None = None) -> TestClient:
    settings = make_settings()
    engine = engine if engine is not None else RecordingEngine(settings)
    return TestClient(create_app(settings, engine=engine), raise_server_exceptions=False)


def error(response) -> dict:
    """The envelope, asserted to BE the envelope -- every failure in this API is this shape."""
    payload = response.json()
    assert set(payload) == {"error"}, payload
    assert set(payload["error"]) == {"code", "message", "details"}, payload
    return payload["error"]


# --- accepted shapes -----------------------------------------------------------------------


def test_a_run_id_only_body_is_accepted_and_forwarded():
    engine = RecordingEngine(make_settings())
    run_id = uuid.uuid4()
    with build_client(engine) as client:
        response = client.post("/api/enrich", json={"run_id": str(run_id)})

    assert response.status_code == 200, response.text
    assert len(engine.calls) == 1
    assert engine.calls[0]["run_id"] == run_id
    assert engine.calls[0]["business_ids"] is None
    assert engine.calls[0]["use_ai"] is True


def test_a_business_ids_only_body_is_accepted_and_forwarded():
    engine = RecordingEngine(make_settings())
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    with build_client(engine) as client:
        response = client.post("/api/enrich", json={"business_ids": ids})

    assert response.status_code == 200, response.text
    assert engine.calls[0]["run_id"] is None
    assert [str(item) for item in engine.calls[0]["business_ids"]] == ids


def test_both_run_id_and_business_ids_may_be_sent_together():
    engine = RecordingEngine(make_settings())
    run_id = uuid.uuid4()
    business_id = uuid.uuid4()
    with build_client(engine) as client:
        response = client.post(
            "/api/enrich", json={"run_id": str(run_id), "business_ids": [str(business_id)]}
        )

    assert response.status_code == 200
    assert engine.calls[0]["run_id"] == run_id
    assert list(engine.calls[0]["business_ids"]) == [business_id]


def test_use_ai_false_is_threaded_through():
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post(
            "/api/enrich", json={"run_id": str(uuid.uuid4()), "use_ai": False}
        )

    assert response.status_code == 200
    assert engine.calls[0]["use_ai"] is False


def test_use_ai_defaults_true_when_omitted():
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        client.post("/api/enrich", json={"run_id": str(uuid.uuid4())})

    assert engine.calls[0]["use_ai"] is True


# --- refusals, each with its own code -------------------------------------------------------


def test_neither_run_id_nor_business_ids_is_refused():
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post("/api/enrich", json={})

    assert response.status_code == 422
    assert error(response)["code"] == "invalid_request"
    assert engine.calls == []


def test_an_empty_request_body_is_refused_the_same_way():
    # No JSON body at all -- `parse_enrich_request` treats `None` as `{}` rather than
    # crashing, so this meets the same "nothing to enrich" refusal as an explicit `{}`.
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post("/api/enrich")

    assert response.status_code == 422
    assert error(response)["code"] == "invalid_request"
    assert engine.calls == []


def test_a_non_object_body_is_refused():
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post("/api/enrich", json=["not", "an", "object"])

    assert response.status_code == 422
    assert error(response)["code"] == "invalid_request"


def test_a_malformed_run_id_is_refused_with_its_own_code():
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post("/api/enrich", json={"run_id": "not-a-uuid"})

    assert response.status_code == 422
    assert error(response)["code"] == "invalid_run_id"
    assert engine.calls == []


def test_a_malformed_business_id_is_refused_with_its_own_code():
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post("/api/enrich", json={"business_ids": ["not-a-uuid"]})

    assert response.status_code == 422
    assert error(response)["code"] == "invalid_business_ids"
    assert engine.calls == []


def test_business_ids_must_be_an_array():
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post("/api/enrich", json={"business_ids": str(uuid.uuid4())})

    assert response.status_code == 422
    assert error(response)["code"] == "invalid_business_ids"


def test_a_non_boolean_use_ai_is_refused():
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post(
            "/api/enrich", json={"run_id": str(uuid.uuid4()), "use_ai": "yes"}
        )

    assert response.status_code == 422
    assert error(response)["code"] == "invalid_use_ai"


# --- the response, and that this runs inline ------------------------------------------------


def test_the_response_reproduces_the_engines_report_exactly():
    engine = RecordingEngine(make_settings())
    run_id = uuid.uuid4()
    with build_client(engine) as client:
        response = client.post("/api/enrich", json={"run_id": str(run_id)})

    body = response.json()
    assert body["run_id"] == str(run_id)
    assert body["enriched"] == 1
    assert body["scored"] == 1
    assert body["offers_detected"] == 2
    assert body["outreach_written"] == 3
    assert body["usage"]["primary_calls"] == 4


def test_the_endpoint_runs_inline_rather_than_being_enqueued():
    # `RecordingEngine` never touches a `Repository` or a queue -- there is nothing here
    # THAT COULD enqueue -- and the recorded call and the 200 both exist by the time this
    # request returns, which is the whole proof that the pass ran synchronously.
    engine = RecordingEngine(make_settings())
    with build_client(engine) as client:
        response = client.post("/api/enrich", json={"run_id": str(uuid.uuid4())})

    assert response.status_code == 200
    assert len(engine.calls) == 1
