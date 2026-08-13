"""Shared test fixtures.

This module exists so that no test module has to import from another one. The prototype
had `tests/test_scoring.py` doing `from tests.test_search import qualifying_lead`, which
couples two suites together and depends on the test package importing cleanly as a
namespace package. A factory is not a test; it lives on its own.
"""

from __future__ import annotations

from lead_engine.models import Lead
from lead_engine.niches import NicheProfile


def qualifying_lead(profile: NicheProfile, provider_id: str | None = None) -> Lead:
    """Build a lead that genuinely satisfies `matches_niche(profile, lead)`.

    Both gates of the qualification rule are fed deliberately:

    * the *type* gate gets one of the profile's own Google type slugs as the category, and
      an exact include outranks every other rule, so no suffix exclusion can fire on it;
    * the *name* gate gets the profile's first qualification term in the business name,
      which is what `strict` profiles demand as corroborating evidence.

    Sorting `include_types` only makes the choice deterministic -- a frozenset has no
    first element, and a factory that picked a different type per run would make failures
    unreproducible.
    """
    term = profile.qualification_terms[0] if profile.qualification_terms else profile.label
    google_type = sorted(profile.include_types)[0]
    return Lead(
        name=f"Example {term.title()}",
        category=google_type,
        address="MG Road, Pune",
        city="Pune",
        latitude=18.52,
        longitude=73.86,
        phone="9123456789",
        website=None,
        source_url="https://example.invalid/source",
        raw_categories=[google_type],
        provider_id=provider_id or f"place-{profile.id}",
    )
