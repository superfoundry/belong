"""Narrowing (SPEC §8.2): hard filters plus text similarity, giving a shortlist per person.

Uses only intent text and *shown* fields. Hidden fields never affect who gets
shortlisted, so they can't become the reason two people meet.
"""
from __future__ import annotations

import unicodedata
from collections import defaultdict

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


_PLACES: dict = {}  # place text -> its plain form, so each place is normalised once


def _place(value):
    """Compare places ignoring case, spacing and accents ("Málaga" == "malaga")."""
    if not value:
        return None
    text = str(value)
    plain = _PLACES.get(text)
    if plain is None:
        if len(_PLACES) > 100_000:
            _PLACES.clear()
        decomposed = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
        plain = _PLACES[text] = " ".join(decomposed.lower().split())
    return plain


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


def _has_phrases(view: ShownView) -> bool:
    return view.get("seeking") is not None and view.get("offering") is not None


PHRASES = "*phrases"  # the group of listings that can be compared with other kinds by their phrases


def shortlists(parsed: dict, views: dict, size: int, focus=None) -> tuple[dict, TfIdf]:
    """For each person (or only those in `focus`): the `size` most similar candidates that pass
    the hard filters (same kind of match, compatible location).

    Top down: listings are grouped by place and kind once, and each listing is only checked against
    its own groups (its city, its country if either is open to anywhere in it, and listings open to
    meeting online), never against everyone. The groups always contain every listing that could pass
    the filters, which are then applied exactly as before, in the same order, so the result is the same."""
    index = TfIdf({pid: v.profile_text() for pid, v in views.items()})
    order = {pid: i for i, pid in enumerate(parsed)}
    by_city, by_country, by_country_only, online = (defaultdict(list) for _ in range(4))
    for b, bp in parsed.items():
        bv = views[b]
        city, country = _place(bv.get("city")), _place(bv.get("country"))
        for group in [bp.kind] + ([PHRASES] if _has_phrases(bv) else []):
            if city:
                by_city[(city, group)].append(b)
            if country:
                by_country[(country, group)].append(b)
                if not city:
                    by_country_only[(country, group)].append(b)
            if bp.remote_ok:
                online[group].append(b)
    out = {}
    for a, ap in parsed.items():
        if focus is not None and a not in focus:
            continue
        if ap.kind is None:
            out[a] = []
            continue
        av = views[a]
        city, country = _place(av.get("city")), _place(av.get("country"))
        pool = set()
        for group in [ap.kind] + ([PHRASES] if _has_phrases(av) else []):
            if city:
                pool.update(by_city[(city, group)])
                if country:
                    pool.update(by_country_only[(country, group)])
            elif country:
                pool.update(by_country[(country, group)])
            if ap.remote_ok:
                pool.update(online[group])
        pool.discard(a)
        cands = [b for b in sorted(pool, key=order.__getitem__)
                 if (parsed[b].kind == ap.kind or _cross_kind_ok(ap, av, parsed[b], views[b]))
                 and location_ok(ap, av, parsed[b], views[b])]
        intent_vec = index.embed(av.intent_text)
        cands.sort(key=lambda b: -cosine(intent_vec, index.vectors[b]))
        out[a] = cands[:size]
    return out, index
