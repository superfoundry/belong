"""Narrowing (SPEC §8.2): hard filters plus text similarity, giving a shortlist per person.

Uses only intent text and *shown* fields. Hidden fields never affect who gets
shortlisted, so they can't become the reason two people meet.
"""
from __future__ import annotations

import unicodedata

from ..schema import Listing, value_strings
from .intent import REMOTE_KINDS, ParsedIntent
from .text import TfIdf, cosine


class ShownView:
    """Read-only view of a listing that exposes shown fields only.

    Matching logic that decides *whether* a pair qualifies receives this, never the
    full listing, so it cannot read a hidden value even by mistake.
    """

    def __init__(self, listing: Listing):
        self._shown = listing.shown()
        self.person_id = listing.person_id
        self.intent_text = listing.intent_text
        self.exactness = listing.exactness

    def get(self, key: str):
        return self._shown.get(key)

    def profile_text(self) -> str:
        parts = [self.intent_text]
        for value in self._shown.values():
            parts.extend(value_strings(value))
        return " ".join(parts)


def _place(value):
    """Compare places ignoring case, spacing and accents ("Málaga" == "malaga")."""
    if not value:
        return None
    plain = "".join(c for c in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(c))
    return " ".join(plain.lower().split())


def same_place(av: ShownView, bv: ShownView):
    """"city" if both are in the same city, "country" if they share a country and at
    least one is open to anywhere in it (no city given), else None. Shown fields only."""
    a_city, b_city = _place(av.get("city")), _place(bv.get("city"))
    a_country, b_country = _place(av.get("country")), _place(bv.get("country"))
    if a_country and b_country and a_country != b_country:
        return None
    if a_city and b_city:
        return "city" if a_city == b_city else None
    if a_country and b_country:
        return "country"
    return None


def location_ok(a: ParsedIntent, av: ShownView, b: ParsedIntent, bv: ShownView) -> bool:
    if same_place(av, bv):
        return True
    remote_kinds = REMOTE_KINDS | ({a.kind, b.kind} if _cross_kind_ok(a, av, b, bv) else set())
    return a.kind in remote_kinds and b.kind in remote_kinds and a.remote_ok and b.remote_ok


def _cross_kind_ok(a: ParsedIntent, av: ShownView, b: ParsedIntent, bv: ShownView) -> bool:
    """Listings filed under different kinds can still be compared by their phrases."""
    phrases = all(v.get(k) is not None for v in (av, bv) for k in ("seeking", "offering"))
    return phrases and "dating" not in (a.kind, b.kind)


def shortlists(parsed: dict, views: dict, size: int) -> tuple[dict, TfIdf]:
    """For each person: the `size` most similar candidates that pass the hard filters
    (same kind of match, compatible location)."""
    index = TfIdf({pid: v.profile_text() for pid, v in views.items()})
    intent_vecs = {pid: index.embed(v.intent_text) for pid, v in views.items()}
    out = {}
    for a, ap in parsed.items():
        if ap.kind is None:
            out[a] = []
            continue
        cands = [b for b, bp in parsed.items()
                 if b != a and (bp.kind == ap.kind or _cross_kind_ok(ap, views[a], bp, views[b]))
                 and location_ok(ap, views[a], bp, views[b])]
        cands.sort(key=lambda b: -cosine(intent_vecs[a], index.vectors[b]))
        out[a] = cands[:size]
    return out, index
