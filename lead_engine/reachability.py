"""Which channel to actually use to reach a lead, in the order that gets a reply.

Priority, per the operator's own stated ordering: a verified mobile number, then a social
handle (DM), then email last -- email is the weakest channel for this use case and is
reached for only when nothing else exists.

WHY THIS DOES NOT TRY TO DETECT "MOBILE" FROM THE DIGITS
-----------------------------------------------------------
The obvious approach -- a number is a mobile if it starts with 6/7/8/9 -- is wrong on real
data from this exact project, not just imprecise. `businesses.phone` for Corner House Ice
Cream (a real Bangalore landline, "080 2520 3364") strips to "8025203364", which starts
with 8: indistinguishable in shape from a real mobile number, because Bangalore's own STD
code is "80". Bangalore is most of this dataset. A wrong "this is a mobile" claim is worse
than no claim at all here -- it sends the operator to WhatsApp a landline that cannot
receive a WhatsApp message, which is a more embarrassing failure than an honestly unranked
number would have been.

So this module never inspects a phone number's digits to guess its type. It ranks by
PROVENANCE instead, which is a signal this codebase already has and already trusts:
`contacts.phone` is a number `contacts.py` found attached to a NAMED PERSON on the
business's own site -- published separately from the shop's main line specifically so
that person can be reached directly, which is exactly why a business publishes an owner's
own number in the first place. `businesses.phone` is whatever Google Maps listed for the
business generally, with no way to tell whether it rings a mobile in someone's pocket or a
landline on a shop counter. The first is genuinely more likely to be a reachable mobile;
the second is treated as "a phone number of unknown type", ranked below a social handle
that is, unambiguously, a channel to reach a specific person's account.
"""

from __future__ import annotations

from dataclasses import dataclass

MOBILE = "mobile"
INSTAGRAM = "instagram"
PHONE_UNKNOWN_TYPE = "phone"
EMAIL = "email"
NONE = "none"


@dataclass(frozen=True)
class ReachChannel:
    """The single best channel to try first, or `NONE` when nothing was found at all."""

    channel: str
    value: str | None = None
    #: What this value is asserted to be, for a caller (or an operator) to judge for
    #: themselves rather than trust a label this module cannot actually verify.
    note: str | None = None

    @property
    def found(self) -> bool:
        return self.channel != NONE


def best_reach_channel(
    *,
    contact_phone: str | None = None,
    contact_name: str | None = None,
    instagram_handle: str | None = None,
    business_phone: str | None = None,
    contact_email: str | None = None,
    business_email: str | None = None,
) -> ReachChannel:
    """Rank whatever this lead already has, honestly, in the operator's own stated order.

    Never guesses a channel that was not actually found -- a blank field stays blank, and
    a `business_phone` with no attributed person is reported as "phone, type unknown"
    rather than promoted to "mobile" on a guess this module has no way to stand behind.
    """
    if contact_phone:
        who = f" (for {contact_name})" if contact_name else ""
        return ReachChannel(
            MOBILE, contact_phone, note=f"published for a named contact{who} -- likely personal"
        )
    if instagram_handle:
        return ReachChannel(INSTAGRAM, instagram_handle, note="DM the business's own account")
    if business_phone:
        return ReachChannel(
            PHONE_UNKNOWN_TYPE,
            business_phone,
            note="the business's listed number -- mobile or landline is not knowable "
            "from this alone",
        )
    if contact_email:
        return ReachChannel(EMAIL, contact_email, note="a named contact's own address")
    if business_email:
        return ReachChannel(EMAIL, business_email, note="a general inbox, not a named person")
    return ReachChannel(NONE)


__all__ = [
    "EMAIL",
    "INSTAGRAM",
    "MOBILE",
    "NONE",
    "PHONE_UNKNOWN_TYPE",
    "ReachChannel",
    "best_reach_channel",
]
