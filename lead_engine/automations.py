"""Automation offer catalogue -- the second offer.

The first offer is a website. The second is agentic automation of the workflows the
business already runs by hand, and it is only worth making once the operator has judged
the lead worth pursuing at all.

Detection is rule-based, exactly as scoring is. The LLM writes the pitch prose over the
rows this catalogue produces; it never decides which opportunities exist. An offer fires
only when every one of its `required_signals` is actually observed, and each observation
names the row it came from. An invented claim about how a business operates is worse than
saying nothing, because the operator repeats it to the owner's face.

Shape
-----
`AutomationOffer` deliberately mirrors `NicheProfile`: a frozen dataclass, a private
`_offer` constructor, one module-level dict keyed by id, and free functions that take an
offer rather than methods on it. That pattern is already proven here, and a second
unrelated shape would be a second thing to learn.

Two gates, and both must pass
-----------------------------
    niche       `niches` empty means every niche; otherwise the business's niche must
                be named. This is what stops an appointment-booking pitch reaching a
                manufacturer.
    signals     every entry in `required_signals` must be present in what was observed.
                Missing one is a miss -- there is no partial credit, no substring match
                and no "close enough".

The niche gate carries the qualitative half of each rule ("appointment-driven",
"class-based", "recurring-service"), which is why several offers require only one or two
signals. Encoding "is this an appointment business" as a signal would mean deriving from
evidence something the registry already states as fact.

The signal vocabulary
---------------------
Two families, declared separately because they have different provenance:

`SCORER_SIGNALS` are strings `lead_engine.scoring.score_lead` puts in `ScoreBreakdown`,
observed over a Google listing. `ENRICHMENT_SIGNALS` are asserted by an enrichment pass
and stored on the row that observed them, so each maps to the `enrichments.source` that
can produce it.

A `required_signals` entry in neither family is a DEAD RULE: it can never fire, and a rule
that can never fire is worse than no rule because it looks like coverage.
`tests/test_automations.py` proves the scorer really emits every string in
`SCORER_SIGNALS` by running it, rather than trusting this file.

The niche-match signals the scorer also emits (`"cafe match"`, `"salon match"`, one per
profile) are deliberately absent from the vocabulary. The niche gate above already carries
that information, and an offer requiring both would state the same condition twice.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

#: Enrichment sources that can assert a signal, as they appear in `enrichments.source`.
#:
#: `google_maps` and `instagram` are already in use (`lead_engine.export.excel`).
#: `firecrawl_site` and `meta_ads` name passes that arrive with website grading and the Ad
#: Library. These names are provenance documentation, not a filter: detection accepts a
#: declared signal from whichever row asserts it and records that row's id as the evidence,
#: so an enricher naming its source differently loses the annotation, not the signal.
ENRICHMENT_SOURCES: tuple[str, ...] = ("google_maps", "instagram", "firecrawl_site", "meta_ads")

#: Signals an enrichment pass asserts, mapped to the source that can observe them.
#:
#: Every one is an observation about the business's public surface, not an inference about
#: its internals. "no booking link" says a booking link was looked for and not found; it
#: does not say the business takes no bookings. That distinction is what keeps the pitch
#: honest: the operator can open with "there is no way to book you online", which is
#: checkable in one click, rather than "you have no booking system", which is a guess.
ENRICHMENT_SIGNALS: dict[str, str] = {
    # --- website grading: what the site does and does not do -----------------------
    "no online ordering": "firecrawl_site",
    "no booking link": "firecrawl_site",
    "no enquiry form": "firecrawl_site",
    "no landing page": "firecrawl_site",
    "manual enquiry flow": "firecrawl_site",
    # --- the listing and its reviews -----------------------------------------------
    "high review count": "google_maps",
    "owner replies absent": "google_maps",
    # --- the social profile ---------------------------------------------------------
    "active social presence": "instagram",
    "instagram-only catalogue": "instagram",
    "dm ordering": "instagram",
    # --- paid acquisition -----------------------------------------------------------
    "runs meta ads": "meta_ads",
}

#: Signals `lead_engine.scoring.score_lead` emits, excluding the per-niche match strings.
#:
#: Listed here rather than imported so this module stays data-only; the test suite runs the
#: scorer and asserts this tuple is exactly what it can produce.
SCORER_SIGNALS: tuple[str, ...] = (
    "no website",
    "social-only",
    "weak/free website",
    "has website",
    "public phone",
    "public web link",
    "detailed business listing",
    "multiple service fit",
)

#: Every signal detection will accept. Anything else observed is ignored -- a typo in an
#: enrichment payload must not become a claim in a pitch.
KNOWN_SIGNALS: frozenset[str] = frozenset(SCORER_SIGNALS) | frozenset(ENRICHMENT_SIGNALS)


@dataclass(frozen=True)
class AutomationOffer:
    id: str
    label: str
    niches: tuple[str, ...]  # empty means every niche
    required_signals: tuple[str, ...]
    pitch_line: str
    est_hours_saved_weekly: float


def _offer(
    offer_id: str,
    label: str,
    *,
    niches: tuple[str, ...] = (),
    signals: tuple[str, ...],
    pitch: str,
    hours: float,
) -> AutomationOffer:
    return AutomationOffer(
        id=offer_id,
        label=label,
        niches=niches,
        required_signals=signals,
        pitch_line=pitch,
        est_hours_saved_weekly=hours,
    )


AUTOMATION_OFFERS: dict[str, AutomationOffer] = {
    # ── Taking the order or the booking ───────────────────────────────────────────
    "order_intake": _offer(
        "order_intake", "Order intake",
        niches=("cafe", "bakery", "cake_shop", "cloud_kitchen"),
        signals=("no online ordering", "active social presence"),
        pitch="Take orders on a page instead of in DMs, with WhatsApp confirmation and "
              "payment taken when the order is placed.",
        hours=6.0,
    ),
    "appointment_booking": _offer(
        "appointment_booking", "Appointment booking and reminders",
        # The registry's appointment-driven niches. `driving_school` sells slots too but
        # buys them as a course, so it belongs to enrolment scheduling instead.
        niches=("salon", "spa", "fitness_gym", "dental_clinic", "clinic", "veterinary"),
        signals=("no booking link",),
        pitch="Let customers book a slot themselves, and send the reminder that stops the "
              "no-show.",
        hours=5.0,
    ),
    "enrolment_scheduling": _offer(
        "enrolment_scheduling", "Enrolment and batch scheduling",
        niches=("tutor_class", "preschool", "driving_school"),
        # Two signals, not one: a class business with a booking link already has the demo
        # slot handled, and the pitch would be describing something they have.
        signals=("no booking link", "manual enquiry flow"),
        pitch="Take admission enquiries on a form, book the demo class, and fill the next "
              "batch without a callback.",
        hours=4.0,
    ),
    # ── Answering the enquiry ─────────────────────────────────────────────────────
    "quotation_handling": _offer(
        "quotation_handling", "Quotation handling",
        niches=("manufacturer", "home_decor", "interior_designer", "event_planner"),
        signals=("no enquiry form",),
        pitch="Collect the specification in a structured form and turn it into a quotation "
              "without a phone call.",
        hours=4.0,
    ),
    "enquiry_routing": _offer(
        "enquiry_routing", "Enquiry capture and routing",
        # The five enquiry-driven niches no other offer reaches. Each sells a bespoke
        # engagement that begins with someone asking a question, and each is left with only
        # the two universal offers otherwise -- see the coverage test.
        niches=("catering", "photographer", "real_estate", "travel_agency",
                "professional_services"),
        signals=("manual enquiry flow", "public phone"),
        pitch="Capture every enquiry with its dates and budget, acknowledge it instantly, "
              "and route it to whoever answers.",
        hours=3.5,
    ),
    # ── Selling the catalogue ─────────────────────────────────────────────────────
    "catalog_whatsapp": _offer(
        "catalog_whatsapp", "Catalogue and WhatsApp checkout",
        niches=("boutique", "bakery", "home_decor"),
        signals=("instagram-only catalogue", "dm ordering"),
        pitch="Put the Instagram catalogue on a page and take the order over WhatsApp with "
              "price and availability already stated.",
        hours=5.0,
    ),
    # ── Keeping the customer ──────────────────────────────────────────────────────
    "service_reminders": _offer(
        "service_reminders", "Service-due reminders",
        niches=("auto_service", "dental_clinic", "veterinary"),
        # `public phone` is load-bearing rather than decorative: reminders go out on a
        # channel, and the listed number is the evidence that one exists.
        signals=("no booking link", "public phone"),
        pitch="Remember when each customer is next due and send the reminder automatically.",
        hours=2.5,
    ),
    # ── Every niche ───────────────────────────────────────────────────────────────
    "review_response": _offer(
        "review_response", "Review response",
        signals=("high review count", "owner replies absent"),
        pitch="Draft a reply to every new review for one-tap approval, so the review page "
              "stops going unanswered.",
        hours=2.0,
    ),
    "lead_capture": _offer(
        "lead_capture", "Ad lead capture",
        signals=("runs meta ads", "no landing page"),
        pitch="Point the ad spend at a page that captures the enquiry and answers it within "
              "the minute.",
        hours=3.0,
    ),
}


def applies_to_niche(offer: AutomationOffer, niche_id: str) -> bool:
    """Whether this offer is on the table for this niche. Empty `niches` means every niche.

    An unknown niche id therefore gets the universal offers and nothing else, which is the
    right answer for both readings of an unknown id: a niche this catalogue has not been
    extended to, or a corrupt `businesses.niche_id`.
    """
    return not offer.niches or niche_id in offer.niches


def missing_signals(offer: AutomationOffer, observed: Iterable[str]) -> tuple[str, ...]:
    """Which required signals were not observed, in declared order.

    Membership is exact. A signal is a token, not a phrase to be matched loosely, so
    "no booking" does not satisfy "no booking link" and neither does "no booking link "
    with a stray space. Loose matching is how a rule fires on evidence nobody gathered.
    """
    present = set(observed)
    return tuple(signal for signal in offer.required_signals if signal not in present)


def offer_fires(offer: AutomationOffer, niche_id: str, observed: Iterable[str]) -> bool:
    """Both gates. Nothing fires on an empty observation set, because every offer in the
    catalogue requires at least one signal -- which the test suite enforces, since an offer
    requiring none would fire for every business in its niches and that is guessing."""
    if not applies_to_niche(offer, niche_id):
        return False
    return not missing_signals(offer, observed)


def offers_for_niche(niche_id: str) -> list[AutomationOffer]:
    """Every offer this niche could ever receive, before any evidence is considered."""
    return [offer for offer in AUTOMATION_OFFERS.values() if applies_to_niche(offer, niche_id)]


def firing_offers(niche_id: str, observed: Iterable[str]) -> list[AutomationOffer]:
    """The offers that actually fire, in catalogue order.

    `observed` is walked once and reused, so a generator caller is safe.
    """
    present = set(observed)
    return [
        offer for offer in AUTOMATION_OFFERS.values() if offer_fires(offer, niche_id, present)
    ]


def automation_payload() -> list[dict]:
    """The catalogue as plain data, for the API and the operator's reference."""
    return [
        {
            "id": offer.id,
            "label": offer.label,
            "niches": list(offer.niches),
            "required_signals": list(offer.required_signals),
            "pitch_line": offer.pitch_line,
            "est_hours_saved_weekly": offer.est_hours_saved_weekly,
        }
        for offer in AUTOMATION_OFFERS.values()
    ]
