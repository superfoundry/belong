"""Narrowing (SPEC §8.2): a shortlist per listing, found top down in the agreed order.

  1. How they meet. Two listings can meet online only if both allow it, and then place
     doesn't matter. Otherwise both must allow meeting in person, and step 2 applies. The
     shown "format" says which ("In person", "Online", "In person or online", in any
     wording). Without one, a listing whose text says online or remote is fine allows both,
     and any other only in person. Dating is always in person.
  2. Where, by travel distance (in person only). Two cities are in reach when the distance
     between them is within *both* people's shown "travel" (same city = 0 km; no travel
     given = the same city only). A distance is used only between two cities in the country
     both people show, each city's centre read in that country: a city whose centre isn't
     known there (or a listing that shows no country) is only in reach of a city of the same
     name. In the same country, someone who gives no city is open to anywhere in it.
  3. What, by meaning. Dating, which has rules of its own, is compared with dating (the kind the
     listing shows, else the one read from its description: matcher.understood, the same reading
     every decision uses). Every other listing is compared by its "looking for" / "you are"
     phrases, found through an inverted index of the words in its phrases, its topics and the
     values of the details it shows about itself (matcher.about), inside the place groups of
     steps 1-2: a beekeeper is never compared with an accountant, nor anyone with people they could
     never meet. What a listing would like the other person's details to be (its "them" and
     "both" details) only ever removes a pair, so it needs no key in the index.

The groups always contain every pair the decisions that follow (matcher.intent_met, and
matcher.nearly for close matches and questions) could accept. Then the hard filters apply as
before, in a stable order, ranked by text similarity.

Uses only intent text and *shown* fields. Hidden fields never affect who gets
shortlisted, so they can't become the reason two people meet.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from ..schema import (ANY, CITY_COORDS, MAX_TRAVEL_KM, TRAVEL, TRAVEL_DEFAULT, UNDERSTOOD_FIELDS, Field, Listing,
                      canonical, km_between, meeting_format, own, plain_place, split_key, travel_choice, value_strings)
from .intent import ParsedIntent
from .text import TfIdf, cosine

ONLINE, IN_PERSON = "online", "in person"
_MODES = {"In person": frozenset({IN_PERSON}), "Online": frozenset({ONLINE}),
          "In person or online": frozenset({IN_PERSON, ONLINE})}
# The kind with rules of its own (matcher.intent_met), compared with its own kind. Everything else
# ("other") is compared by its phrases.
STRUCTURED_KINDS = frozenset({"dating"})


class ShownView:
    """Read-only view of a listing that exposes shown fields only.

    Matching logic that decides *whether* a pair qualifies receives this, never the
    full listing, so it cannot read a hidden value even by mistake. That includes its open
    details: only the shown ones are here.
    """

    def __init__(self, listing: Listing):
        self._shown = listing.shown()
        self._fields = listing.shown_fields()
        by_about = defaultdict(list)
        for key in sorted(self._fields):
            by_about[split_key(key)[0]].append(self._fields[key])
        self._own = tuple(by_about["me"] + by_about["both"])
        self._wishes = tuple(by_about["them"] + by_about["both"])
        self._own_values = tuple(v for f in self._own for v in f.strings() if canonical(v) != ANY)
        self.person_id = listing.person_id
        self.intent_text = listing.intent_text
        self.exactness = listing.exactness

    def get(self, key: str):
        return self._shown.get(key)

    @property
    def fields(self) -> dict:
        """The shown fields themselves, by key (what schema's detail comparisons read)."""
        return dict(self._fields)

    def own(self, name: str) -> Optional[Field]:
        """This listing's own shown value of an open detail (its me: or both: detail), or None."""
        return own(self._fields, name)

    def own_details(self) -> tuple:
        """Its shown open details about the person or the request itself (me: and both:)."""
        return self._own

    def own_values(self) -> tuple:
        """The values of those details, but "any", which is no value of its own (matcher.about)."""
        return self._own_values

    def wishes(self) -> tuple:
        """What it would like the other person's own values to be: its shown them: and both: details."""
        return self._wishes

    def profile_text(self) -> str:
        """The description and shown values, as words for text similarity: open details' values come
        in too, but never a detail's name or scale. What the person's agent understood
        (UNDERSTOOD_FIELDS) is left out: it already decides who qualifies, and as words ("dating",
        "other") it would only blur how alike two listings read."""
        parts = [self.intent_text]
        for key, value in self._shown.items():
            if key not in UNDERSTOOD_FIELDS:
                parts.extend(value_strings(value))
        return " ".join(parts)


def _cached(cache: dict, value, read):
    """read(value), remembered: the same few places, formats and distances come up again and again."""
    out = cache.get(value, cache)
    if out is cache:
        if len(cache) > 100_000:
            cache.clear()
        out = cache[value] = read(value)
    return out


_PLACES: dict = {}
_FORMATS: dict = {}
_TRAVEL: dict = {}
_KM: dict = {}
_NEAR: dict = {}


def _place(value):
    """Compare places ignoring case, spacing and accents ("Málaga" == "malaga")."""
    return _cached(_PLACES, str(value), plain_place) if value else None


def meeting_modes(parsed: ParsedIntent, view: ShownView) -> frozenset:
    """How this listing can meet: in person, online, or both (step 1). The shown format decides;
    without a readable one, the intent text does. Dating is always in person."""
    if parsed.kind == "dating":
        return _MODES["In person"]
    shown = view.get("format")
    fmt = _cached(_FORMATS, shown, meeting_format) if isinstance(shown, str) else None
    if fmt is None:
        return _MODES["In person or online"] if parsed.remote_ok else _MODES["In person"]
    return _MODES[fmt]


def travel_km(view: ShownView) -> float:
    """How far this person will go to meet in person, from their shown "travel" (0 = same city)."""
    shown = view.get("travel")
    choice = _cached(_TRAVEL, shown, travel_choice) if isinstance(shown, str) else None
    return TRAVEL[choice or TRAVEL_DEFAULT]


def _km(a: str, b: str, country: str):
    """The distance between two cities in one country (plain names), or None if either isn't known there."""
    key = (a, b, country) if a <= b else (b, a, country)
    return _cached(_KM, key, lambda k: km_between(k[0], k[2], k[1], k[2]))


def _near(city: str, country) -> list:
    """Known cities in the same country within the longest travel distance of this one, with their
    distance: worked out once, so a listing only ever looks at the cities it could reach. A city whose
    centre isn't known in the country shown with it (or with no country) has none."""
    if not _NEAR:
        for a, a_country in CITY_COORDS:
            _NEAR[(a, a_country)] = sorted(
                (km, b) for b, b_country in CITY_COORDS if b_country == a_country and b != a
                for km in [km_between(a, a_country, b, b_country)] if km <= MAX_TRAVEL_KM)
    return _NEAR.get((city, country), [])


def within_reach(av: ShownView, bv: ShownView):
    """Where two people could meet in person (step 2): "city" if their cities are in reach (the same
    city, or within both people's travel distance), "country" if they share a country and at least
    one is open to anywhere in it (no city given), else None. Shown fields only.

    A distance needs both people to show the same country, and each city's centre is read in that
    country, so a Cambridge in one country is never near a London in another. Without both
    countries, a city is only in reach of a city of the same name."""
    a_city, b_city = _place(av.get("city")), _place(bv.get("city"))
    a_country, b_country = _place(av.get("country")), _place(bv.get("country"))
    if a_country and b_country and a_country != b_country:
        return None
    if a_city and b_city:
        if a_city == b_city:
            return "city"
        if not (a_country and b_country):
            return None  # no shown country, so no known centre: only the same city
        km = _km(a_city, b_city, a_country)
        return "city" if km is not None and km <= min(travel_km(av), travel_km(bv)) else None
    if a_country and b_country:
        return "country"
    return None


def how_they_meet(a: ParsedIntent, av: ShownView, b: ParsedIntent, bv: ShownView):
    """ONLINE if both allow meeting online (then place doesn't matter); otherwise, if both allow
    meeting in person, where they could (within_reach); else None."""
    am, bm = meeting_modes(a, av), meeting_modes(b, bv)
    if ONLINE in am and ONLINE in bm:
        return ONLINE
    if IN_PERSON in am and IN_PERSON in bm:
        return within_reach(av, bv)
    return None


def location_ok(a: ParsedIntent, av: ShownView, b: ParsedIntent, bv: ShownView) -> bool:
    return how_they_meet(a, av, b, bv) is not None


def _by_phrase(p: ParsedIntent, view: ShownView) -> bool:
    """Compared by its "looking for" / "you are" phrases: it has both, and it isn't dating."""
    return view.get("seeking") is not None and view.get("offering") is not None and p.kind != "dating"


_POSTED: dict = {}


def _places_posted(p: ParsedIntent, view: ShownView) -> list:
    """The place groups a listing belongs to (steps 1-2)."""
    key = (p.kind == "dating", p.remote_ok, view.get("format"), view.get("city"), view.get("country"))
    try:
        return _cached(_POSTED, key, lambda _: _places_of(p, view))
    except TypeError:  # a value that can't be a key (not a string): work it out each time
        return _places_of(p, view)


def _places_of(p: ParsedIntent, view: ShownView) -> list:
    modes = meeting_modes(p, view)
    city, country = _place(view.get("city")), _place(view.get("country"))
    out = []
    if IN_PERSON in modes:
        if city:
            out.append(("city", city))
        if country:
            out.append(("country", country))
            if not city:
                out.append(("anywhere in", country))
    if ONLINE in modes:
        out.append((ONLINE,))
    return out


def _places_looked_at(p: ParsedIntent, view: ShownView) -> list:
    """The place groups a listing looks in: its own city and the cities in its shown country within
    its own travel distance (the other person's distance is checked by the filter), anyone in its country who
    is open to anywhere in it (or, with no city, everyone in its country), and online if it
    meets online. Together they hold everyone it could meet."""
    modes = meeting_modes(p, view)
    city, country = _place(view.get("city")), _place(view.get("country"))
    out = []
    if IN_PERSON in modes:
        if city:
            limit = travel_km(view)
            out.append(("city", city))
            out += [("city", other) for km, other in _near(city, country) if km <= limit]
            if country:
                out.append(("anywhere in", country))
        elif country:
            out.append(("country", country))
    if ONLINE in modes:
        out.append((ONLINE,))
    return out


OPEN = "open to company"  # listings whose "looking for" may be met without a shared word (see shortlists)


def shortlists(parsed: dict, views: dict, size: int, focus=None, words=None) -> tuple[dict, TfIdf]:
    """For each listing (or only those in `focus`): the `size` most similar candidates that pass
    the hard filters (the same kind of match; a way to meet).

    Top down: listings are put in place groups once (how and where they can meet, steps 1-2), and
    each listing only looks in the groups it could meet in. In those groups, dating is found by
    kind, and every other listing through the word index of its phrases (step 3), built for a group
    only when a phrase listing first looks in it, so a run for a few listings never indexes the whole
    board.

    words(view) -> (want, have, open): a listing's index keys, read the way the Matchmaker's own
    decisions read it (matcher.index_words): `want` from what it looks for, `have` from what it is
    and shows, and whether its "looking for" could be met without any shared word. Want keys are
    looked up among others' have keys and the other way round, so two listings meet if either
    one's want shares a key with the other's have. Without `words`, every phrase listing in reach
    is a candidate for every other (one catch-all group, as before the index): slower, and the
    same pairs qualify.

    The groups always contain every listing that could pass the filters and the decisions after
    them; the filters are then applied exactly as written, in board order, so only the ranking
    picks among them."""
    index = TfIdf({pid: v.profile_text() for pid, v in views.items()})
    order = {pid: i for i, pid in enumerate(parsed)}

    by_kind, by_phrase = defaultdict(list), defaultdict(list)  # (place, kind) / place -> listings
    for b, bp in parsed.items():
        if bp.kind is None:
            continue  # a listing nobody can read is never anyone's candidate
        bv = views[b]
        structured, phrase = bp.kind in STRUCTURED_KINDS, _by_phrase(bp, bv)
        if structured or phrase:
            for place in _places_posted(bp, bv):
                if structured:
                    by_kind[(place, bp.kind)].append(b)
                if phrase:
                    by_phrase[place].append(b)

    word_keys: dict = {}

    def keys_of(pid):
        out = word_keys.get(pid)
        if out is None:
            out = word_keys[pid] = words(views[pid])
        return out

    word_groups: dict = {}

    def word_index(place):
        """The phrase listings in one place group, by their want and have keys."""
        idx = word_groups.get(place)
        if idx is None:
            idx = word_groups[place] = defaultdict(list)
            for b in by_phrase.get(place, ()):
                want, have, is_open = keys_of(b)
                for k in want:
                    idx[("want", k)].append(b)
                for k in have:
                    idx[("have", k)].append(b)
                if is_open:
                    idx[OPEN].append(b)
        return idx

    out = {}
    for a, ap in parsed.items():
        if focus is not None and a not in focus:
            continue
        if ap.kind is None:
            out[a] = []
            continue
        av = views[a]
        places = _places_looked_at(ap, av)
        pool = set()
        if ap.kind in STRUCTURED_KINDS:
            for place in places:
                pool.update(by_kind.get((place, ap.kind), ()))
        if _by_phrase(ap, av):
            if words is None:
                for place in places:
                    pool.update(by_phrase.get(place, ()))
            else:
                want, have, is_open = keys_of(a)
                keys = [("have", k) for k in want] + [("want", k) for k in have] + [OPEN]
                for place in places:
                    if is_open:
                        pool.update(by_phrase.get(place, ()))
                        continue
                    idx = word_index(place)
                    for key in keys:
                        pool.update(idx.get(key, ()))
        pool.discard(a)
        cands = [b for b in sorted(pool, key=order.__getitem__)
                 if parsed[b].kind == ap.kind and location_ok(ap, av, parsed[b], views[b])]
        intent_vec = index.embed(av.intent_text)
        cands.sort(key=lambda b: -cosine(intent_vec, index.vectors[b]))
        out[a] = cands[:size]
    return out, index
