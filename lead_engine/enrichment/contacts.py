"""A named contact, and nothing invented to fill the gap.

Contacts are extracted under the same rule as every other fact this system records: a
validated schema or a failure. An extractor that cannot find a name records nothing. It
never guesses one, since a fabricated owner name is worse than a blank field the moment
the operator opens with it. The corollary, easy to miss, is symmetric: a contact DETAIL
with no attributable name is just as much a guess if it gets attached to whichever name
happens to be nearby. `find_contacts` below records a person only when a name is clearly
present AND clearly tied -- in the same line, or a bare name-only line paired with the
role/phone/email on the very next line or two -- to at least one of a role, a phone, or an
email. Two facts that merely share a page, but not a line, get nothing. When in real doubt,
this module records nothing: a blank field costs nothing, a wrong name attached to a
business is worse than either.

THIS MODULE ISSUES NO REQUEST OF ANY KIND -- IT IS A PURE FUNCTION OVER TEXT ALREADY PAID
FOR. `EnrichmentService._grade` fetches a business's own homepage once, to grade the site;
that text is sitting in memory. Reading a named contact out of the SAME text costs nothing
more, which is the entire reason `website` was the first source built here. Of the design
spec's other four:

    ig_bio        NOW BUILT (`find_bio_contacts`, below). It was blocked -- Instagram bio
                  text did not exist anywhere in this system -- until this session's Phase 7
                  work (`workers/instagram.py`) started reading one out of an already-fetched
                  profile response. That module's own docstring explains why the extraction
                  lives HERE and not there: "`contacts.py` is the module that turns page text
                  into a named contact... re-implementing a shrunken version of it here...
                  is exactly the kind of duplicate logic that drifts." `find_bio_contacts` is
                  the seam `workers/instagram.py` calls once IT is wired into a pipeline --
                  which it is not yet: no queue task type or `Engine` method invokes
                  `InstagramProfileLookup` today, and the real, human-built
                  `InstagramBrowser` implementation does not exist either. Building this
                  extractor now is safe and cheap regardless -- it is pure, it is fully
                  testable against fixture bios, and it costs nothing to have ready before
                  the rest of Phase 7's pipeline exists to call it.
    review_reply  needs a NEW paid SearchAPI reviews-with-replies call. Discovery owns every
                  one of that budget's 50 one-time searches; spending one here is a cost
                  decision for someone who owns that budget, not a decision this module gets
                  to make by adding a call.
    directory     (Justdial, IndiaMART) needs new scraping capability this codebase does not
                  have. `website.py` treats those hosts only as a REJECT list -- evidence the
                  business exists and none at all that a URL there is safe to scrape as
                  structured data.
    search        is a paid Firecrawl query for "owner/founder/proprietor", and the design
                  spec's own Open Questions flag it as premature before the free sources
                  have a measured hit rate. Spending a metered credit to validate a source
                  that has not been measured is the wrong order of operations.

`website` and `ig_bio` share the same extraction engine, `find_contacts`, parameterised by
`source` and `confidence` -- the "same vicinity" rule below applies identically to a
homepage and a bio, and a business's own text is a business's own text regardless of which
page it came from. Only the trust level differs, per the design spec's source table: 0.8
for a business's own site, 0.6 for a bio, because a bio is unmoderated and shorter-lived
than a homepage a business chose to publish and maintain.

WHAT IS DELIBERATELY NOT DONE
------------------------------
Only the business's already-fetched HOMEPAGE text is read. A dedicated About/Contact
sub-page fetch -- which would need a new Firecrawl call and URL-guessing logic to find that
page in the first place -- is a future enhancement, not this slice. Whatever a contact
block looks like on the homepage is what gets read; a business whose only named contact
lives on a page nobody fetched yields nothing here, honestly, rather than a call this phase
was not budgeted to make.

There is also no `status` field and no `unavailable()` here, unlike `website.py` and
`social.py`. Those modules answer a question that can go three ways -- found, genuinely
absent, or the lookup itself failed -- because they are graded on a search that ran or did
not. This module never runs anything that can fail: it either finds a name in the text it
was handed or it does not, and "did not find one" and "there was no text to look at" are
both exactly the same thing to a caller -- an empty `ContactExtraction`. The `contacts`
table has no status column either; a fact about a person is recorded when it exists, never
as a placeholder for its own absence.

THE "SAME VICINITY" HEURISTIC, AND ITS HONEST LIMIT
-----------------------------------------------------
"Same vicinity" is implemented as: the same line, or a line containing only a name (nothing
else) paired with the role/phone/email found on the very next one or two non-blank lines,
stopping the moment another bare name-only line appears (that is a second card starting,
not detail for the first). This is a real limitation, stated plainly rather than solved
half-way: a page that puts an unrelated name and an unrelated contact detail on the SAME
line -- "Priya Sharma has run the shop for years; call our manager on 080-1234567" -- will
be misread as one fact instead of two unconnected ones. Building a real coreference
resolver to close that gap is out of scope for a slice whose only input is homepage prose;
the honest trade here is the same one `website.py` makes with the acronym-domain false
negative -- a known, bounded, documented edge over an untestable heuristic that tries to be
clever about it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Role must be one of these four. Never inferred from context -- only from an explicit
#: label in the text ("Owner", "Founder", "Manager", "Marketing" and their close spellings).
ROLE_OWNER = "owner"
ROLE_MANAGER = "manager"
ROLE_MARKETING = "marketing"
ROLE_UNKNOWN = "unknown"

#: `contacts.source` values this module writes.
SOURCE_WEBSITE = "website"
SOURCE_IG_BIO = "ig_bio"

#: Fixed per the source table in the design spec: `website` and `review_reply` are worth
#: 0.8; `ig_bio` and `directory` are worth 0.6, ahead only of `search` at 0.4. This module
#: builds the two sources that need no new capability or spend -- see the module docstring.
CONFIDENCE = 0.8
CONFIDENCE_IG_BIO = 0.6

# --- role labels -----------------------------------------------------------------------

#: Checked in order. Marketing variants must be tested before the bare "manager" pattern,
#: or "Marketing Manager" would be misread as the generic manager role.
_ROLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmarketing\b", re.I), ROLE_MARKETING),
    (re.compile(r"\bco-?founders?\b", re.I), ROLE_OWNER),
    (re.compile(r"\bfounders?\b", re.I), ROLE_OWNER),
    (re.compile(r"\bowners?\b", re.I), ROLE_OWNER),
    (re.compile(r"\bproprietors?\b", re.I), ROLE_OWNER),
    (re.compile(r"\bmanagers?\b", re.I), ROLE_MANAGER),
)

#: Every word/phrase a role rule recognises, plus the "Contact:" style channel labels that
#: name a detail rather than a role. Stripped out of a line before the name pattern runs
#: over it, so a role word standing next to a name is never captured as part of the name --
#: see the "Founder Priya Sharma" example in the module docstring.
_ROLE_WORDS_PATTERN = re.compile(
    r"\b(?:"
    r"co-?founders?|founders?|owners?|proprietors?|managers?|marketing|"
    r"contact|reach\s+us|write\s+to|enquir(?:y|ies)|general"
    r")\b",
    re.I,
)


def _extract_role(line: str) -> str | None:
    for pattern, role in _ROLE_RULES:
        if pattern.search(line):
            return role
    return None


# --- emails and phones -------------------------------------------------------------------

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

#: A run of digits, spaces, hyphens and an optional leading `+`, at least 7 and at most 13
#: digits once the punctuation is stripped -- wide enough for an Indian mobile or landline
#: with a country code, narrow enough that a 3-digit house number in "100 Feet Road" or a
#: 4-digit year never qualifies.
_PHONE = re.compile(r"\+?\d[\d\-\s]{5,16}\d")


def _extract_email(line: str) -> str | None:
    match = _EMAIL.search(line)
    return match.group(0) if match else None


def _extract_phone(line: str) -> str | None:
    for match in _PHONE.finditer(line):
        digits = re.sub(r"\D", "", match.group(0))
        if 7 <= len(digits) <= 13:
            return match.group(0).strip()
    return None


# --- names -----------------------------------------------------------------------------

_TITLE = r"(?:Mr|Mrs|Ms|Dr|Mx)\.?\s+"
_NAME_CORE = r"[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,2}"
_NAME = re.compile(rf"(?:{_TITLE})?({_NAME_CORE})")

#: Common boilerplate that happens to be Capitalised at the start of a sentence or heading.
#: A candidate containing any of these tokens is rejected outright -- borrowed in spirit
#: from `website.py`'s `GENERIC_NAME_TOKENS`, for the same reason: these words identify a
#: page section, never a person.
_STOPWORDS = frozenset(
    {
        "our",
        "menu",
        "contact",
        "about",
        "home",
        "order",
        "book",
        "call",
        "visit",
        "shop",
        "team",
        "meet",
        "welcome",
        "products",
        "services",
        "gallery",
        "location",
        "hours",
        "policy",
        "privacy",
        "terms",
        "cart",
        "checkout",
        "cake",
        "cakes",
        "bakery",
        "questions",
        "general",
        "thanks",
        "thank",
        # Address furniture. It shares a line with a phone number as often as a name does
        # ("Call 080 4111 2222 or visit the shop on 100 Feet Road"), and a street is not a
        # person -- see the module docstring's note on the same-line heuristic's limits.
        "road",
        "street",
        "lane",
        "avenue",
        "feet",
        "nagar",
        "layout",
        "block",
        "sector",
        "floor",
        "cross",
        "main",
    }
)


def _looks_like_name(candidate: str) -> bool:
    tokens = [token.lower() for token in candidate.split()]
    if len(tokens) < 2:
        return False
    if any(token in _STOPWORDS for token in tokens):
        return False
    return True


def _extract_name(line: str) -> str | None:
    """The first plausible person name in `line`, with role/label words masked out first."""
    cleaned = _ROLE_WORDS_PATTERN.sub(" ", line)
    for match in _NAME.finditer(cleaned):
        candidate = match.group(1)
        if _looks_like_name(candidate):
            return candidate
    return None


def _is_name_only_line(line: str) -> str | None:
    """A short line that is a name and nothing else -- a staff card's name row."""
    stripped = line.strip()
    if not stripped or len(stripped) > 40:
        return None
    if _extract_role(stripped) or _extract_email(stripped) or _extract_phone(stripped):
        return None
    match = re.fullmatch(rf"(?:{_TITLE})?({_NAME_CORE})\.?", stripped)
    if not match:
        return None
    candidate = match.group(1)
    return candidate if _looks_like_name(candidate) else None


# --- the finding --------------------------------------------------------------------------


@dataclass(frozen=True)
class ContactCandidate:
    """One named person, read out of page text. Never a placeholder for "found nobody"."""

    name: str
    role: str
    phone: str | None
    email: str | None
    source: str
    source_url: str | None
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "phone": self.phone,
            "email": self.email,
            "source": self.source,
            "source_url": self.source_url,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ContactExtraction:
    """Zero or more `ContactCandidate`s read from one block of text.

    There is no status here. Nothing found is not a fact worth a row -- it is simply an
    empty tuple, exactly as if the text had never been fetched at all.
    """

    candidates: tuple[ContactCandidate, ...] = ()

    @property
    def found(self) -> bool:
        return bool(self.candidates)

    def as_dict(self) -> dict[str, Any]:
        return {"candidates": [candidate.as_dict() for candidate in self.candidates]}


def find_contacts(
    text: str | None,
    *,
    source_url: str | None,
    source: str = SOURCE_WEBSITE,
    confidence: float = CONFIDENCE,
) -> ContactExtraction:
    """Read named contacts out of `text`. Pure -- no I/O, and safe on any input.

    Two passes over the text's lines:

    1. A single line that carries BOTH a name and at least one of a role/phone/email is
       recorded whole -- the "Owner: Priya Sharma, priya@cakebee.in" and the
       "-- Founder Priya Sharma" shapes.
    2. A line that is a name and only a name is paired with the role/phone/email on the
       next one or two non-blank lines -- the staff-card shape where each fact gets its own
       line -- and the search stops the moment another bare name-only line appears, since
       that is a second card starting, not more detail for the first.

    Either way, a name is never recorded without at least one attached detail, and a detail
    is never recorded without an attached name. Duplicate mentions of the same name keep
    only the first (case-insensitively), matching `social.py`'s "same handle, one row" rule.

    `source`/`confidence` default to the `website` source's values so every existing caller
    is unaffected; `find_bio_contacts` below is the only other caller today, passing
    `ig_bio`/0.6. The extraction rule itself does not change per source -- a name is a name
    and a bio is prose exactly like a homepage is, just shorter and less durable.
    """
    if not text or not isinstance(text, str):
        return ContactExtraction(candidates=())

    lines = text.splitlines()
    count = len(lines)
    consumed: set[int] = set()
    seen_names: set[str] = set()
    candidates: list[ContactCandidate] = []

    def add(name: str, role: str | None, phone: str | None, email: str | None) -> None:
        key = name.strip().lower()
        if not key or key in seen_names:
            return
        seen_names.add(key)
        candidates.append(
            ContactCandidate(
                name=name.strip(),
                role=role or ROLE_UNKNOWN,
                phone=phone,
                email=email,
                source=source,
                source_url=source_url,
                confidence=confidence,
            )
        )

    # Pass 1: everything needed is on one line.
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        role = _extract_role(line)
        email = _extract_email(line)
        phone = _extract_phone(line)
        if not (role or email or phone):
            continue
        name = _extract_name(line)
        if not name:
            continue
        add(name, role, phone, email)
        consumed.add(index)

    # Pass 2: a bare name line, paired with detail on the next line or two.
    for index, raw_line in enumerate(lines):
        if index in consumed:
            continue
        line = raw_line.strip()
        if not line:
            continue
        name = _is_name_only_line(line)
        if not name:
            continue

        role: str | None = None
        phone: str | None = None
        email: str | None = None
        used: list[int] = []
        seen_detail_lines = 0
        cursor = index + 1
        while cursor < count and seen_detail_lines < 2:
            candidate_line = lines[cursor].strip()
            if not candidate_line:
                cursor += 1
                continue
            seen_detail_lines += 1
            if cursor not in consumed and _is_name_only_line(candidate_line):
                # A second card starting, not more detail for this one.
                break
            found_role = _extract_role(candidate_line)
            found_phone = _extract_phone(candidate_line)
            found_email = _extract_email(candidate_line)
            if found_role or found_phone or found_email:
                role = role or found_role
                phone = phone or found_phone
                email = email or found_email
                used.append(cursor)
            cursor += 1

        if role or phone or email:
            add(name, role, phone, email)
            consumed.add(index)
            consumed.update(used)

    return ContactExtraction(candidates=tuple(candidates))


def find_bio_contacts(bio: str | None, *, source_url: str | None = None) -> ContactExtraction:
    """`find_contacts`, at the `ig_bio` source and its 0.6 confidence.

    The caller is `workers/instagram.py`'s `ProfileFinding.bio` -- once something wires
    `InstagramProfileLookup` into a pipeline, which nothing does yet (see the module
    docstring). Until then this is a ready, independently-tested seam, not a live path: it
    takes a bio string and returns candidates exactly as `find_contacts` would, with no
    knowledge of, or dependency on, `workers/instagram.py` itself -- consistent with that
    module's own instruction that a shrunken duplicate extractor must not live there.
    """
    return find_contacts(
        bio, source_url=source_url, source=SOURCE_IG_BIO, confidence=CONFIDENCE_IG_BIO
    )


__all__ = [
    "CONFIDENCE",
    "CONFIDENCE_IG_BIO",
    "ROLE_MANAGER",
    "ROLE_MARKETING",
    "ROLE_OWNER",
    "ROLE_UNKNOWN",
    "SOURCE_IG_BIO",
    "SOURCE_WEBSITE",
    "ContactCandidate",
    "ContactExtraction",
    "find_bio_contacts",
    "find_contacts",
]
