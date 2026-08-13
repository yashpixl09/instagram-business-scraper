"""The operator's terminal: argument parsing, printing, and exit codes.

    python -m lead_engine.cli --city Bangalore --area Indiranagar --niche salon --dry-run
    python -m lead_engine.cli --city Bangalore --niche salon --fixtures tests/fixtures/searchapi
    python -m lead_engine.cli --city Bangalore --area Indiranagar --niche salon --niche cafe

Everything below is presentation. What a search *is* -- validation, geography, the goal and
run records, the discovery pass, the score, the sheet -- lives in `lead_engine.api.deps`,
and the HTTP routes call the same methods with the same arguments. That is not tidiness: a
web frontend arrives later and must never need a capability that only this file has, and the
only way to guarantee that is for this file to hold none.

The request is validated by `schemas.parse_search_request`, the same function the API's
`POST /api/search` uses, so `--niche nail-bar` is rejected here exactly as it is over HTTP,
with the same code and the same supported list.

`--fixtures DIR` IS A FIRST-CLASS FLAG
--------------------------------------
It swaps the recorded-response provider in for the billed one. Fifty searches exist, ever,
and they do not renew, so the whole pipeline -- discovery, scoring, the workbook -- is
developed and demonstrated against `tests/fixtures/searchapi` for free. Only a deliberate
run without `--fixtures` spends anything.

BEFORE ANYTHING BILLABLE
------------------------
The resolved coordinates and the remaining allowance are printed before the first search is
issued, and a plan that would cost more than remains is refused outright rather than
started and stopped halfway. `--dry-run` prints the same block and exits without touching a
provider at all.

EXIT CODES
----------
    0   the run found something
    1   the run completed and found nothing new, or a provider failed mid-run
    2   this run cannot start: bad arguments, an unsupported niche, missing configuration,
        a location that will not resolve, or a plan that costs more than the allowance holds
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .api.deps import Engine, build_engine, fixture_provider
from .api.schemas import ApiProblem, SearchPlanOut, SearchSpec, parse_search_request
from .config import Settings
from .providers.errors import ProviderError

EXIT_OK = 0
EXIT_NO_RESULTS = 1
EXIT_CANNOT_START = 2

#: Leads across every niche, split evenly. The prototype's default.
DEFAULT_LIMIT = 60

DEFAULT_OUTPUT_DIR = "exports"

PROGRAM = "lead-engine"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Find local businesses with no effective web presence, score them, and write "
            "the operator's sheet."
        ),
    )
    parser.add_argument("--city", required=True, help="City to search. Anchors everything.")
    parser.add_argument("--state", default=None, help="Disambiguates the city; never narrows.")
    parser.add_argument("--country", default=None, help="Disambiguates the city, and sets `gl`.")
    parser.add_argument(
        "--area",
        action="append",
        default=[],
        metavar="AREA",
        help=(
            "Neighbourhood to sweep; repeatable. Each area is at least one billed search "
            "per niche. With none given the whole city is one 15km search per niche."
        ),
    )
    parser.add_argument(
        "--niche",
        action="append",
        default=[],
        metavar="NICHE",
        help="Registry niche id, label or alias; repeatable. See GET /api/niches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Leads wanted across every niche, split evenly (default {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Use deterministic copy only. A run completes either way; this pins it.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where the workbook is written (default {DEFAULT_OUTPUT_DIR}/).",
    )
    parser.add_argument(
        "--fixtures",
        default=None,
        metavar="DIR",
        help=(
            "Serve recorded responses from DIR instead of calling SearchAPI. Spends "
            "nothing; this is how the pipeline is developed and demonstrated."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the geography, plan the cells, print the cost, and stop.",
    )
    return parser


def build_spec(args: argparse.Namespace) -> SearchSpec:
    """Arguments -> the same validated `SearchSpec` the API builds from a JSON body.

    Going through `parse_search_request` rather than constructing a `SearchSpec` directly is
    what keeps the two surfaces honest: the 160-character ceiling, the area cap, the niche
    resolution and the limit range are enforced once, and `--limit 0` is refused here with
    the code an HTTP caller would receive.
    """
    payload: dict[str, Any] = {
        "location": {
            "city": args.city,
            "state": args.state,
            "country": args.country,
            "areas": list(args.area or ()),
        },
        "niches": list(args.niche or ()),
        "limit": args.limit,
        "use_ai": not args.no_ai,
    }
    return parse_search_request(payload)


def configuration_problem(args: argparse.Namespace, settings: Settings) -> ApiProblem | None:
    """What this run is missing, before anything is built.

    Checked here rather than left to the Engine so the message names the flag or the
    variable that would fix it. A missing key is not a broken server and must not read like
    one -- the same distinction `/api/search` makes with its 503.
    """
    if not args.fixtures and settings.searchapi_key_value is None:
        return ApiProblem(
            503,
            "provider_not_configured",
            "SEARCHAPI_KEY is not set, so no search can be issued. Set it in .env, or pass "
            "--fixtures DIR to run against recorded responses for free.",
        )
    if args.fixtures and not Path(args.fixtures).is_dir():
        return ApiProblem(
            422,
            "invalid_request",
            f"--fixtures {args.fixtures!r} is not a directory.",
        )
    if not args.dry_run and settings.dsn is None:
        return ApiProblem(
            503,
            "database_not_configured",
            "LEAD_ENGINE_DSN is not set. A run records goals, cells, businesses and scores, "
            "and there is nowhere to put them.",
        )
    if not args.dry_run:
        # Up front, before a credit can move. The sheet is the product, and discovering
        # that its directory is unwritable *after* the searches have been billed turns a
        # typo into a spend -- the rows survive in Postgres, but the operator paid for a
        # run that produced no file.
        try:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ApiProblem(
                422,
                "invalid_request",
                f"--output-dir {args.output_dir!r} cannot be written to: {exc.strerror or exc}.",
            )
    return None


def build_cli_engine(args: argparse.Namespace, settings: Settings) -> Engine:
    """The same Engine the API builds, with the fixture provider swapped in when asked."""
    provider = fixture_provider(args.fixtures) if args.fixtures else None
    return build_engine(settings, maps_provider=provider)


# --- printing ---------------------------------------------------------------------------


def _out(message: str = "") -> None:
    print(message)


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def print_plan(plan: SearchPlanOut, *, billed: bool) -> None:
    """The pre-flight block. Printed before a credit can move, on every path."""
    _out("Resolved locations:")
    for location in plan.locations:
        _out(
            f"  {location.label}"
            f"  ({location.latitude:.6f}, {location.longitude:.6f})"
            f"  r={location.radius_meters}m  {location.precision}"
        )
    _out(f"Niches: {', '.join(plan.niches)}")
    remaining = plan.search_budget_remaining
    _out(
        f"Planned searches: {plan.planned_searches}"
        f"  ({'billed against SearchAPI' if billed else 'served from fixtures, free'})"
    )
    _out(
        "Search budget remaining: "
        + ("unknown -- no ledger row for this key yet" if remaining is None else str(remaining))
    )


def print_report(report: Any, remaining: int | None) -> None:
    outcome = report.outcome
    _out()
    _out(f"Run {report.run_id}")
    _out(f"  new businesses : {len(outcome.businesses)}")
    _out(f"  searches spent : {outcome.searches_spent}")
    _out(f"  cells searched : {outcome.cells_searched} of {outcome.cells_planned} planned")
    _out(f"  stopped        : {outcome.stopped}")
    _out(f"  scored         : {report.scored}")
    for niche in outcome.niches:
        _out(
            f"    {niche.niche_id:<22} searches={niche.searches} candidates={niche.candidates} "
            f"qualified={niche.qualified} new={niche.new_businesses} "
            f"known={niche.already_stored} state={niche.state or '-'}"
        )
    if outcome.starved_niches:
        # The loud signal: real places came back and the registry refused all of them.
        _out(
            "  STARVED niches (the type slugs are wrong, not the neighbourhood): "
            + ", ".join(outcome.starved_niches)
        )
    if report.export_path:
        _out(f"  sheet          : {report.export_path}")
    if remaining is not None:
        _out(f"  budget left    : {remaining}")


# --- the run ------------------------------------------------------------------------------


def _fail(problem: ApiProblem) -> int:
    _err(f"{problem.code}: {problem.message}")
    supported = problem.details.get("supported_niches")
    if supported:
        _err("supported niches: " + ", ".join(str(niche) for niche in supported))
    return EXIT_CANNOT_START


def _run(args: argparse.Namespace, spec: SearchSpec, engine: Engine) -> int:
    billed = not args.fixtures
    try:
        plan = engine.plan(spec)
    except ProviderError as exc:
        # An unresolvable area aborts before anything is spent. There is no fallback to the
        # city centre: a sweep aimed at ground nobody asked for produces real businesses at
        # real phone numbers, and nothing anywhere says it went wrong.
        _err(f"{exc.code}: {exc.message}")
        return EXIT_CANNOT_START

    print_plan(plan, billed=billed)

    if args.dry_run:
        _out("Dry run: nothing was searched and no credit was spent.")
        return EXIT_OK

    remaining = plan.search_budget_remaining
    if billed and remaining is not None and plan.planned_searches > remaining:
        _err(
            f"budget_exhausted: this plan needs {plan.planned_searches} searches and "
            f"{remaining} remain. Narrow --area or --niche, or rotate the key. Nothing has "
            "been spent."
        )
        return EXIT_CANNOT_START

    goal_id, run_id = engine.open_run(spec, trigger="cli")
    try:
        report = engine.execute_run(
            spec,
            goal_id=goal_id,
            run_id=run_id,
            # A ceiling even when a ledger is present: the plan is what the operator was
            # shown, and a pass must not quietly cost more than the number they read.
            max_searches=plan.planned_searches,
            output_dir=args.output_dir,
        )
    except ProviderError as exc:
        engine.fail_run(run_id, exc.code)
        _err(f"{exc.code}: {exc.message}")
        return EXIT_NO_RESULTS

    print_report(report, engine.remaining_budget())
    if not report.found:
        _out("No new businesses. Nothing here this system did not already hold.")
        return EXIT_NO_RESULTS
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    engine: Engine | None = None,
) -> int:
    """Parse, validate, plan, run. Returns the exit code rather than raising SystemExit."""
    args = build_parser().parse_args(argv)
    settings = settings if settings is not None else Settings()

    try:
        spec = build_spec(args)
    except ApiProblem as problem:
        return _fail(problem)

    owned = engine is None
    if engine is None:
        problem = configuration_problem(args, settings)
        if problem is not None:
            return _fail(problem)
        engine = build_cli_engine(args, settings)

    try:
        return _run(args, spec, engine)
    except ApiProblem as problem:
        return _fail(problem)
    finally:
        if owned:
            engine.close()


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
