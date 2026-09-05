"""The HTTP surface: paths, status codes, and nothing else.

Every function here is three lines or fewer of adaptation over `deps.Engine`, and that is
the point rather than a coincidence. The CLI calls the same methods, so a capability cannot
exist on one surface and not the other; when the frontend arrives, a missing endpoint is a
route to write, never a behaviour to reimplement.

Two conventions worth stating:

*Requests are parsed, not modelled.* `POST /api/search` and `PATCH /api/leads/{id}/status`
take the raw body and hand it to `schemas.parse_search_request` /
`schemas.parse_verdict_update`, which raise `ApiProblem` with a *specific code per field*.
Declaring a pydantic request model instead would answer every one of those with one
`RequestValidationError` and a `loc` tuple, and the caller would have to reconstruct
`invalid_limit` from a path. See the long note at the top of `schemas.py`.

*Dependencies arrive through `Annotated`.* `engine: EngineDep` rather than
`engine=Depends(get_engine)`, because a call in a default argument is a real bug in almost
every other context and the linter is right to say so.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Query, Response

from .deps import DEFAULT_LEAD_LIMIT, DEFAULT_RUN_LIMIT, MAX_LEAD_LIMIT, MAX_RUN_LIMIT, Engine
from .deps import get_engine as _get_engine
from .schemas import (
    ConfigStatusOut,
    EnrichReportOut,
    HealthOut,
    LeadListOut,
    LeadOut,
    NicheListOut,
    RunListOut,
    RunOut,
    SearchAcceptedOut,
    SearchPlanOut,
    parse_enrich_request,
    parse_search_request,
    parse_verdict_update,
)

EngineDep = Annotated[Engine, Depends(_get_engine)]

router = APIRouter()


@router.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    """Liveness only.

    Deliberately does not touch the database. A health check that fails when Postgres is
    unreachable cannot be used to tell "the process is up and the database is down" from
    "the process is down", which is the one distinction it exists to make.
    """
    return HealthOut(status="ok")


@router.get("/api/niches", response_model=NicheListOut, tags=["catalogue"])
def niches(engine: EngineDep) -> NicheListOut:
    """The 24 registry profiles in registry order, each with its verification state."""
    return engine.niches()


@router.get("/api/config/status", response_model=ConfigStatusOut, tags=["meta"])
def config_status(engine: EngineDep) -> ConfigStatusOut:
    """Which providers are configured -- booleans -- and what is left of the allowance.

    No key, no prefix, no length, no fingerprint. See `ConfigStatusOut` and `config.py`.
    """
    return engine.config_status()


@router.post(
    "/api/search",
    response_model=SearchAcceptedOut,
    status_code=202,
    tags=["search"],
)
def start_search(engine: EngineDep, payload: Annotated[Any, Body()] = None) -> SearchAcceptedOut:
    """Accept a search and queue the work. 202, because nothing has been searched yet.

    The response is a receipt: what was queued, where it will look, and what it will cost
    if every task runs. No SearchAPI call happens in this request -- a browser refresh on an
    endpoint that spent a credit would spend it twice, out of an allowance of fifty that
    never renews.
    """
    return engine.start_search(parse_search_request(payload))


@router.post("/api/search/plan", response_model=SearchPlanOut, tags=["search"])
def plan_search(engine: EngineDep, payload: Annotated[Any, Body()] = None) -> SearchPlanOut:
    """What that search WOULD cost, without queueing or spending anything.

    The HTTP half of the CLI's `--dry-run`, and the reason it exists on both surfaces: the
    operator holds fifty non-renewing searches, and "how much will this cost" is the
    question they must be able to ask before answering "yes". A frontend without this would
    have to compute the cell count itself -- which is the moment two definitions of the
    price appear and one of them starts being wrong.

    Resolves geography, which is free and cached. Creates no goal, no run and no task.
    """
    return engine.plan(parse_search_request(payload))


@router.get("/api/runs", response_model=RunListOut, tags=["runs"])
def list_runs(
    engine: EngineDep,
    limit: Annotated[int, Query(ge=1, le=MAX_RUN_LIMIT)] = DEFAULT_RUN_LIMIT,
) -> RunListOut:
    return engine.runs(limit)


@router.get("/api/runs/{run_id}", response_model=RunOut, tags=["runs"])
def get_run(engine: EngineDep, run_id: UUID) -> RunOut:
    return engine.run(run_id)


@router.get("/api/leads", response_model=LeadListOut, tags=["leads"])
def list_leads(
    engine: EngineDep,
    run_id: Annotated[UUID | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LEAD_LIMIT)] = DEFAULT_LEAD_LIMIT,
) -> LeadListOut:
    """Leads in a run's scope; the latest run when `run_id` is omitted."""
    return engine.leads(run_id, limit=limit)


@router.get("/api/leads/{lead_id}", response_model=LeadOut, tags=["leads"])
def get_lead(engine: EngineDep, lead_id: UUID) -> LeadOut:
    return engine.lead(lead_id)


@router.post("/api/enrich", response_model=EnrichReportOut, tags=["enrichment"])
def enrich(engine: EngineDep, payload: Annotated[Any, Body()] = None) -> EnrichReportOut:
    """Enrich, re-score, detect automation opportunities, and draft outreach -- inline.

    Unlike `POST /api/search`, this is not enqueued for a worker: TinyFish is free and
    capped only by request rate, Firecrawl is a monthly credit pool this call merely draws
    from and reports usage on rather than a non-renewing allowance, and the LLM call for a
    prompt already answered is served back out of `llm_calls` rather than re-billed. None of
    that is the one-shot, unrecoverable spend `execute_run` is kept off this layer for.
    """
    spec = parse_enrich_request(payload)
    report = engine.execute_enrichment(
        run_id=spec.run_id, business_ids=spec.business_ids or None, use_ai=spec.use_ai
    )
    return EnrichReportOut(
        run_id=spec.run_id,
        business_ids=list(report.business_ids),
        enriched=report.enriched,
        scored=report.scored,
        offers_detected=report.offers_detected,
        outreach_written=report.outreach_written,
        usage=report.usage,
    )


@router.patch("/api/leads/{lead_id}/status", status_code=204, tags=["leads"])
def set_verdict(
    engine: EngineDep,
    lead_id: UUID,
    payload: Annotated[Any, Body()] = None,
) -> Response:
    """Record the operator's verdict. 204: the caller already knows what it wrote."""
    engine.set_verdict(lead_id, parse_verdict_update(payload))
    return Response(status_code=204)
